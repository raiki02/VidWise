package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"

	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
	"github.com/raiki02/vidwise/internal/tool"
)

const (
	VideoProcessStepDownloadAudio      = "download_audio"
	VideoProcessStepTranscribeAudio    = "transcribe_audio"
	VideoProcessStepFormatTranscript   = "format_transcript"
	VideoProcessStepIndexKnowledgeBase = "index_knowledge_base"
)

// VideoProcessStepNames returns the stable ordered steps for the video Agent pipeline.
func VideoProcessStepNames() []string {
	return []string{
		VideoProcessStepDownloadAudio,
		VideoProcessStepTranscribeAudio,
		VideoProcessStepFormatTranscript,
		VideoProcessStepIndexKnowledgeBase,
	}
}

// VideoProcessObserver receives step-level pipeline events.
type VideoProcessObserver interface {
	StepStarted(name string)
	StepDone(name string)
	StepFailed(name string, err error)
	StepSkipped(name, reason string)
}

// ExecuteVideoProcess runs the video processing pipeline:
// Download audio → ASR transcribe → LLM format → RAG index.
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
			}
			if err := json.Unmarshal([]byte(formatResult), &fmtOutput); err == nil && fmtOutput.FormattedText != "" {
				formattedText = fmtOutput.FormattedText
			} else {
				formattedText = formatResult
			}
			slog.Info("agent.pipeline.format_done", "text_len", len(formattedText))
			observer.StepDone(VideoProcessStepFormatTranscript)
		}
	}

	// Step 4: RAG index to Qdrant
	ragTool, ragErr := registry.Get("rag_index")
	if ragErr != nil {
		observer.StepSkipped(VideoProcessStepIndexKnowledgeBase, "rag_index tool unavailable")
		slog.Warn("agent.pipeline.no_rag_tool", "err", ragErr)
	} else {
		observer.StepStarted(VideoProcessStepIndexKnowledgeBase)
		ragArgs, _ := tool.ToJSON(tool.RAGIndexInput{
			Text:        formattedText,
			Filename:    name + ".txt",
			ContentType: "text/plain",
			Format:      "plain",
			UserID:      userID,
			SessionID:   sessionID,
			Metadata: map[string]string{
				qdrantclient.FieldTaskID:    taskID,
				qdrantclient.FieldSourceURL: url,
			},
		})
		ragResult, err := ragTool.InvokableRun(ctx, ragArgs)
		if err != nil {
			observer.StepFailed(VideoProcessStepIndexKnowledgeBase, err)
			slog.Warn("agent.pipeline.rag_index_failed", "err", err)
		} else {
			slog.Info("agent.pipeline.rag_index_done", "result", ragResult)
			observer.StepDone(VideoProcessStepIndexKnowledgeBase)
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
