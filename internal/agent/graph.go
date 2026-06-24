package agent

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/cloudwego/eino/schema"
	"github.com/raiki02/video-extractor/internal/tool"
)

// ExecuteVideoProcess runs the video processing pipeline step by step:
// Download → AudioExtract → ASR → TextFormat → RAGIndex
// This is a simpler, more reliable approach than Eino Graph for linear DAGs.
func ExecuteVideoProcess(ctx context.Context, registry *tool.Registry, url, workDir, name, userID, sessionID, taskID string, language string) (transcript string, err error) {
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
	audioJSON, err := downloadTool.InvokableRun(ctx, audioArgs)
	if err != nil {
		return "", fmt.Errorf("extract audio: %w", err)
	}
	slog.Info("agent.pipeline.audio_done", "output", audioJSON)

	// Parse audio path from output
	audioPath := fmt.Sprintf("%s/%s.mp3", workDir, name)

	// Step 2: Transcribe
	slog.Info("agent.pipeline.transcribe", "path", audioPath)
	asrTool, err := registry.Get("transcribe_audio")
	if err != nil {
		return "", fmt.Errorf("get transcribe_audio tool: %w", err)
	}
	asrArgs, _ := tool.ToJSON(map[string]string{
		"audio_path": audioPath,
		"language":   language,
	})
	asrJSON, err := asrTool.InvokableRun(ctx, asrArgs)
	if err != nil {
		return "", fmt.Errorf("transcribe: %w", err)
	}
	slog.Info("agent.pipeline.asr_done")

	// Step 3: Format text (optional, depending on LLM config)
	slog.Info("agent.pipeline.format")
	formatTool, err := registry.Get("format_transcript")
	if err != nil {
		slog.Warn("agent.pipeline.no_format_tool", "err", err)
		// Return raw ASR text if formatter not available
		return asrJSON, nil
	}
	formatArgs, _ := tool.ToJSON(map[string]string{"raw_text": asrJSON})
	formattedJSON, err := formatTool.InvokableRun(ctx, formatArgs)
	if err != nil {
		slog.Warn("agent.pipeline.format_failed_fallback", "err", err)
		return asrJSON, nil // Fallback to raw text
	}

	// Step 4: Index to RAG (background)
	slog.Info("agent.pipeline.rag_index")
	ragTool, err := registry.Get("rag_index")
	if err != nil {
		slog.Warn("agent.pipeline.no_rag_tool", "err", err)
		return formattedJSON, nil
	}
	ragArgs, _ := tool.ToJSON(map[string]string{
		"text":       formattedJSON,
		"task_id":    taskID,
		"user_id":    userID,
		"session_id": sessionID,
	})
	ragJSON, err := ragTool.InvokableRun(ctx, ragArgs)
	if err != nil {
		slog.Warn("agent.pipeline.rag_index_failed", "err", err)
	} else {
		slog.Info("agent.pipeline.rag_index_done", "result", ragJSON)
	}

	return formattedJSON, nil
}

// ExecuteChatQuery runs the RAG Q&A pipeline.
func ExecuteChatQuery(ctx context.Context, registry *tool.Registry, query, userID, sessionID string) (string, error) {
	slog.Info("agent.pipeline.chat", "query", query)
	ragQueryTool, err := registry.Get("rag_query")
	if err != nil {
		return "", fmt.Errorf("get rag_query tool: %w", err)
	}
	ragArgs, _ := tool.ToJSON(map[string]string{
		"query":      query,
		"user_id":    userID,
		"session_id": sessionID,
	})
	result, err := ragQueryTool.InvokableRun(ctx, ragArgs)
	if err != nil {
		return "", fmt.Errorf("rag query: %w", err)
	}
	return result, nil
}

// BuildContextMessages builds the context messages for an LLM call based on retrieved chunks.
func BuildContextMessages(query string, contextChunks []string) []*schema.Message {
	msgs := []*schema.Message{
		schema.SystemMessage("You are a helpful assistant. Answer the user's question based ONLY on the provided context. If the context does not contain enough information, say so."),
	}

	contextText := ""
	for i, chunk := range contextChunks {
		if i > 0 {
			contextText += "\n---\n"
		}
		contextText += fmt.Sprintf("[Chunk %d]: %s", i+1, chunk)
	}

	if len(contextChunks) > 0 {
		msgs = append(msgs, &schema.Message{
			Role:    "user",
			Content: fmt.Sprintf("Context:\n%s\n\nQuestion: %s", contextText, query),
		})
	} else {
		msgs = append(msgs, schema.UserMessage(query))
	}

	return msgs
}
