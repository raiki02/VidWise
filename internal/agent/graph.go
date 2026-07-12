package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/raiki02/vidwise/internal/tool"
)

const (
	VideoProcessStepDownloadAudio    = "download_audio"
	VideoProcessStepTranscribeAudio  = "transcribe_audio"
	VideoProcessStepFormatTranscript = "format_transcript"

	VideoProcessTextStageTranscribed = "transcribed"
	VideoProcessTextStageFormatted   = "formatted"
	VideoProcessTextStageReady       = "ready"
)

// VideoProcessStepNames returns the stable ordered steps for the video Agent pipeline.
func VideoProcessStepNames() []string {
	return []string{
		VideoProcessStepDownloadAudio,
		VideoProcessStepTranscribeAudio,
		VideoProcessStepFormatTranscript,
	}
}

// VideoProcessObserver receives step-level pipeline events.
type VideoProcessObserver interface {
	StepStarted(name string)
	StepDone(name string)
	StepFailed(name string, err error)
	StepSkipped(name, reason string)
}

// VideoProcessOutputObserver receives partial task output as soon as the
// pipeline has user-facing transcript text available.
type VideoProcessOutputObserver interface {
	OutputUpdated(output map[string]any)
}

// ExecuteVideoProcess runs the video processing pipeline:
// Download audio → ASR transcribe → optional LLM paragraph formatting.
func ExecuteVideoProcess(ctx context.Context, registry *tool.Registry, url, workDir, name, userID, sessionID, taskID string, language string) (string, error) {
	return ExecuteVideoProcessWithObserver(ctx, registry, url, workDir, name, userID, sessionID, taskID, language, nil)
}

// ExecuteVideoProcessWithObserver runs the video pipeline and reports each
// coarse-grained production step to the supplied observer.
func ExecuteVideoProcessWithObserver(ctx context.Context, registry *tool.Registry, url, workDir, name, userID, sessionID, taskID string, language string, observer VideoProcessObserver) (string, error) {
	observer = normalizeVideoProcessObserver(observer)

	// Step 1: Download audio via yt-dlp
	observer.StepStarted(VideoProcessStepDownloadAudio)
	slog.Info("agent.pipeline.download", "url", url)
	downloadTool, err := registry.Get(VideoProcessStepDownloadAudio)
	if err != nil {
		observer.StepFailed(VideoProcessStepDownloadAudio, err)
		return "", fmt.Errorf("get download_audio tool: %w", err)
	}
	audioArgs, _ := tool.ToJSON(map[string]string{
		"url":         url,
		"output_base": fmt.Sprintf("%s/%s", workDir, name),
	})
	_, err = downloadTool.InvokableRun(ctx, audioArgs)
	if err != nil {
		observer.StepFailed(VideoProcessStepDownloadAudio, err)
		return "", fmt.Errorf("extract audio: %w", err)
	}
	observer.StepDone(VideoProcessStepDownloadAudio)
	slog.Info("agent.pipeline.audio_done")

	audioPath := fmt.Sprintf("%s/%s.mp3", workDir, name)

	// Step 2: Transcribe via ASR
	observer.StepStarted(VideoProcessStepTranscribeAudio)
	slog.Info("agent.pipeline.transcribe", "path", audioPath)
	asrTool, err := registry.Get("transcribe_audio")
	if err != nil {
		observer.StepFailed(VideoProcessStepTranscribeAudio, err)
		return "", fmt.Errorf("get transcribe_audio tool: %w", err)
	}
	asrArgs, _ := tool.ToJSON(map[string]string{"audio_path": audioPath, "language": language})
	asrJSON, err := asrTool.InvokableRun(ctx, asrArgs)
	if err != nil {
		observer.StepFailed(VideoProcessStepTranscribeAudio, err)
		return "", fmt.Errorf("transcribe: %w", err)
	}
	observer.StepDone(VideoProcessStepTranscribeAudio)
	slog.Info("agent.pipeline.asr_done", "json_len", len(asrJSON))

	// ASR tool returns JSON: {"text":"...", "segments":[...], ...}
	// Extract just the text field for further processing.
	rawText := asrJSON
	var asrOutput struct {
		Text     string  `json:"text"`
		Language string  `json:"language"`
		Duration float64 `json:"duration"`
	}
	if err := json.Unmarshal([]byte(asrJSON), &asrOutput); err == nil && asrOutput.Text != "" {
		rawText = asrOutput.Text
		slog.Info("agent.pipeline.asr_parsed", "text_len", len(rawText), "lang", asrOutput.Language, "duration", asrOutput.Duration)
	}
	notifyVideoProcessOutput(observer, VideoProcessTextOutput(rawText, name, url, VideoProcessTextStageTranscribed))

	// Step 3: LLM paragraph formatting (optional)
	formattedText := rawText
	formatTool, formatErr := registry.Get("format_transcript")
	if formatErr != nil {
		observer.StepSkipped(VideoProcessStepFormatTranscript, "format_transcript tool unavailable")
		slog.Warn("agent.pipeline.no_format_tool", "err", formatErr)
	} else {
		observer.StepStarted(VideoProcessStepFormatTranscript)
		formatArgs, _ := tool.ToJSON(map[string]string{"raw_text": rawText})
		formatResult, err := formatTool.InvokableRun(ctx, formatArgs)
		if err != nil {
			observer.StepFailed(VideoProcessStepFormatTranscript, err)
			slog.Warn("agent.pipeline.format_failed_fallback", "err", err)
		} else {
			var fmtOutput struct {
				FormattedText string `json:"formatted_text"`
				Formatted     *bool  `json:"formatted"`
				Status        string `json:"status"`
				Reason        string `json:"reason"`
			}
			formatSucceeded := false
			if err := json.Unmarshal([]byte(formatResult), &fmtOutput); err == nil && fmtOutput.FormattedText != "" {
				if fmtOutput.Formatted == nil || *fmtOutput.Formatted {
					formattedText = fmtOutput.FormattedText
					formatSucceeded = true
				} else {
					reason := firstNonEmpty(fmtOutput.Reason, fmtOutput.Status, "format_transcript did not produce formatted text")
					observer.StepSkipped(VideoProcessStepFormatTranscript, reason)
					slog.Warn("agent.pipeline.format_skipped", "reason", reason)
				}
			} else {
				formattedText = strings.TrimSpace(formatResult)
				formatSucceeded = formattedText != ""
			}
			if formatSucceeded {
				slog.Info("agent.pipeline.format_done", "text_len", len(formattedText))
				notifyVideoProcessOutput(observer, VideoProcessTextOutput(formattedText, name, url, VideoProcessTextStageFormatted))
				observer.StepDone(VideoProcessStepFormatTranscript)
			}
		}
	}

	slog.Info("agent.pipeline.done", "raw_len", len(rawText), "formatted_len", len(formattedText))
	return formattedText, nil
}

type noopVideoProcessObserver struct{}

func (noopVideoProcessObserver) StepStarted(string)         {}
func (noopVideoProcessObserver) StepDone(string)            {}
func (noopVideoProcessObserver) StepFailed(string, error)   {}
func (noopVideoProcessObserver) StepSkipped(string, string) {}

func normalizeVideoProcessObserver(observer VideoProcessObserver) VideoProcessObserver {
	if observer == nil {
		return noopVideoProcessObserver{}
	}
	return observer
}

func VideoProcessTextOutput(text, name, sourceURL, stage string) map[string]any {
	filename := strings.TrimSpace(name)
	if filename == "" {
		filename = "transcript"
	}
	if !strings.HasSuffix(strings.ToLower(filename), ".txt") {
		filename += ".txt"
	}
	if stage == "" {
		stage = VideoProcessTextStageReady
	}
	output := map[string]any{
		"text":                text,
		"text_length":         len(text),
		"text_stage":          stage,
		"filename":            filename,
		"source_url":          strings.TrimSpace(sourceURL),
		"text_formatted":      false,
		"can_index_knowledge": false,
		"knowledge_indexed":   false,
	}
	switch stage {
	case VideoProcessTextStageTranscribed:
		output["raw_text"] = text
		output["raw_text_length"] = len(text)
	case VideoProcessTextStageFormatted:
		output["formatted_text"] = text
		output["formatted_text_length"] = len(text)
		output["text_formatted"] = true
		output["can_index_knowledge"] = true
	}
	return output
}

func VideoProcessReadyTextOutput(text, rawText, name, sourceURL string, formatted bool) map[string]any {
	output := VideoProcessTextOutput(text, name, sourceURL, VideoProcessTextStageReady)
	rawText = strings.TrimSpace(rawText)
	if rawText != "" {
		output["raw_text"] = rawText
		output["raw_text_length"] = len(rawText)
	}
	if formatted {
		output["formatted_text"] = text
		output["formatted_text_length"] = len(text)
		output["text_formatted"] = true
		output["can_index_knowledge"] = true
		return output
	}
	if rawText == "" {
		output["raw_text"] = text
		output["raw_text_length"] = len(text)
	}
	output["formatting_required"] = true
	return output
}

func notifyVideoProcessOutput(observer VideoProcessObserver, output map[string]any) {
	outputObserver, ok := observer.(VideoProcessOutputObserver)
	if !ok {
		return
	}
	outputObserver.OutputUpdated(output)
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			return value
		}
	}
	return ""
}
