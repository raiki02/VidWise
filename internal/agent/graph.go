package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"

	"github.com/raiki02/video-extractor/internal/tool"
)

// ExecuteVideoProcess runs the video processing pipeline:
// Download audio → ASR transcribe → LLM format → RAG index.
func ExecuteVideoProcess(ctx context.Context, registry *tool.Registry, url, workDir, name, userID, sessionID, taskID string, language string) (string, error) {
	// Step 1: Download audio via yt-dlp
	slog.Info("agent.pipeline.download", "url", url)
	downloadTool, err := registry.Get("extract_audio")
	if err != nil {
		return "", fmt.Errorf("get extract_audio tool: %w", err)
	}
	audioArgs, _ := tool.ToJSON(map[string]string{
		"url":         url,
		"output_base": fmt.Sprintf("%s/%s", workDir, name),
	})
	_, err = downloadTool.InvokableRun(ctx, audioArgs)
	if err != nil {
		return "", fmt.Errorf("extract audio: %w", err)
	}
	slog.Info("agent.pipeline.audio_done")

	audioPath := fmt.Sprintf("%s/%s.mp3", workDir, name)

	// Step 2: Transcribe via ASR
	slog.Info("agent.pipeline.transcribe", "path", audioPath)
	asrTool, err := registry.Get("transcribe_audio")
	if err != nil {
		return "", fmt.Errorf("get transcribe_audio tool: %w", err)
	}
	asrArgs, _ := tool.ToJSON(map[string]string{"audio_path": audioPath, "language": language})
	asrJSON, err := asrTool.InvokableRun(ctx, asrArgs)
	if err != nil {
		return "", fmt.Errorf("transcribe: %w", err)
	}
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
		slog.Warn("agent.pipeline.no_format_tool", "err", formatErr)
	} else {
		formatArgs, _ := tool.ToJSON(map[string]string{"raw_text": rawText})
		formatResult, err := formatTool.InvokableRun(ctx, formatArgs)
		if err != nil {
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
		}
	}

	// Step 4: RAG index to Qdrant
	ragTool, ragErr := registry.Get("rag_index")
	if ragErr != nil {
		slog.Warn("agent.pipeline.no_rag_tool", "err", ragErr)
	} else {
		ragArgs, _ := tool.ToJSON(map[string]string{
			"text": formattedText,
		})
		ragResult, err := ragTool.InvokableRun(ctx, ragArgs)
		if err != nil {
			slog.Warn("agent.pipeline.rag_index_failed", "err", err)
		} else {
			slog.Info("agent.pipeline.rag_index_done", "result", ragResult)
		}
	}

	slog.Info("agent.pipeline.done", "raw_len", len(rawText), "formatted_len", len(formattedText))
	return formattedText, nil
}
