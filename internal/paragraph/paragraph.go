package paragraph

import (
	"context"
	"log/slog"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	einomodel "github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/raiki02/video-extractor/internal/appconfig"
)

const maxParallelChunks = 3

func FormatText(ctx context.Context, rawText string, cfg appconfig.LLMConfig) (string, error) {
	return formatText(ctx, rawText, cfg, false)
}

func FormatTextWithFallback(ctx context.Context, rawText string, cfg appconfig.LLMConfig) (string, error) {
	return formatText(ctx, rawText, cfg, true)
}

func formatText(ctx context.Context, rawText string, cfg appconfig.LLMConfig, hasFallback bool) (string, error) {
	start := time.Now()
	rawText = strings.TrimSpace(rawText)
	if rawText == "" {
		return "", nil
	}
	if cfg.Enabled != nil && !*cfg.Enabled {
		return rawText, nil
	}
	if strings.TrimSpace(cfg.Model) == "" {
		return rawText, nil
	}
	fallback := hasFallback && (cfg.FallbackToRawOnError == nil || *cfg.FallbackToRawOnError)

	cm, err := NewChatModel(ctx, cfg)
	if err != nil {
		if fallback {
			slog.Warn("llm.format.unavailable_fallback", "err", err)
			return rawText, nil
		}
		return "", err
	}

	chunks := splitByRunes(rawText, cfg.ChunkRunes)
	slog.Info("llm.format.start", "chunks", len(chunks), "chunk_runes", cfg.ChunkRunes)

	perChunkTimeout, err := cfg.TimeoutDuration()
	if err != nil {
		perChunkTimeout = 3 * time.Minute
	}

	formatted := formatChunksParallel(ctx, cm, chunks, cfg, perChunkTimeout, fallback)
	slog.Info("llm.format.done", "elapsed", time.Since(start))
	return strings.Join(formatted, "\n\n"), nil
}

type chunkResult struct {
	index int
	text  string
}

func formatChunksParallel(
	ctx context.Context,
	cm einomodel.BaseChatModel,
	chunks []string,
	cfg appconfig.LLMConfig,
	perChunkTimeout time.Duration,
	fallback bool,
) []string {
	if len(chunks) == 1 {
		text := formatChunk(ctx, cm, cfg, 0, chunks[0], perChunkTimeout)
		if text == "" && !fallback {
			return nil
		}
		if text == "" {
			return []string{chunks[0]}
		}
		return []string{text}
	}

	sem := make(chan struct{}, maxParallelChunks)
	results := make([]chunkResult, len(chunks))
	var wg sync.WaitGroup
	var mu sync.Mutex
	var firstErr error

	for i, chunk := range chunks {
		wg.Add(1)
		go func(idx int, text string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			result := formatChunk(ctx, cm, cfg, idx, text, perChunkTimeout)

			mu.Lock()
			if result == "" && firstErr == nil {
				firstErr = errChunkFailed
			}
			if result != "" {
				results[idx] = chunkResult{index: idx, text: result}
			}
			mu.Unlock()
		}(i, chunk)
	}
	wg.Wait()

	if firstErr != nil {
		if fallback {
			slog.Warn("llm.format.chunk_failed_fallback", "err", firstErr)
			return nil
		}
		return nil
	}

	formatted := make([]string, 0, len(results))
	for _, r := range results {
		if r.text != "" {
			formatted = append(formatted, r.text)
		}
	}
	return formatted
}

var errChunkFailed = errChunkFailedType{}

type errChunkFailedType struct{}

func (e errChunkFailedType) Error() string { return "chunk formatting failed" }

func formatChunk(
	parentCtx context.Context,
	cm einomodel.BaseChatModel,
	cfg appconfig.LLMConfig,
	idx int,
	chunk string,
	timeout time.Duration,
) string {
	stage := time.Now()
	ctx, cancel := context.WithTimeout(parentCtx, timeout)
	defer cancel()

	resp, err := cm.Generate(ctx, []*schema.Message{
		schema.SystemMessage(cfg.Prompt.System),
		schema.UserMessage(renderUserPrompt(cfg.Prompt.UserTemplate, chunk)),
	}, einomodel.WithTemperature(cfg.Temperature), einomodel.WithMaxTokens(cfg.MaxTokens))
	if err != nil {
		slog.Warn("llm.format.chunk_failed", "index", idx, "elapsed", time.Since(stage), "err", err)
		return ""
	}
	slog.Info("llm.format.chunk_done", "index", idx, "elapsed", time.Since(stage))

	return strings.TrimSpace(resp.Content)
}

func renderUserPrompt(template, text string) string {
	if template == "" {
		return text
	}
	return strings.ReplaceAll(template, "{{text}}", text)
}

func splitByRunes(text string, limit int) []string {
	if limit <= 0 {
		limit = utf8.RuneCountInString(text)
	}
	if utf8.RuneCountInString(text) <= limit {
		return []string{text}
	}

	var chunks []string
	runes := []rune(text)
	for start := 0; start < len(runes); start += limit {
		end := start + limit
		if end > len(runes) {
			end = len(runes)
		}
		chunk := strings.TrimSpace(string(runes[start:end]))
		if chunk != "" {
			chunks = append(chunks, chunk)
		}
	}
	return chunks
}
