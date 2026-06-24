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
	"github.com/raiki02/vidwise/internal/appconfig"
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
	formatCtx := context.WithoutCancel(ctx)

	cm, err := NewChatModel(formatCtx, cfg)
	if err != nil {
		if fallback {
			slog.Warn("llm.format.unavailable_fallback", "err", err)
			return rawText, nil
		}
		return "", err
	}

	perChunkTimeout, err := cfg.TimeoutDuration()
	if err != nil {
		perChunkTimeout = 3 * time.Minute
	}

	if cfg.TwoStep {
		text := formatTwoStep(formatCtx, cm, rawText, cfg, perChunkTimeout, fallback)
		if text == "" && !fallback {
			return "", errChunkFailed
		}
		return text, nil
	}

	chunks := splitByRunes(rawText, cfg.ChunkRunes)
	slog.Info("llm.format.start", "chunks", len(chunks), "chunk_runes", cfg.ChunkRunes)

	formatted := formatChunksParallel(formatCtx, cm, chunks, rawText, cfg, perChunkTimeout, fallback)
	slog.Info("llm.format.done", "elapsed", time.Since(start))
	return strings.Join(formatted, "\n\n"), nil
}

func formatChunksParallel(
	ctx context.Context,
	cm einomodel.BaseChatModel,
	chunks []string,
	rawText string,
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
	results := make([]string, len(chunks))
	if fallback {
		copy(results, chunks)
	}
	var wg sync.WaitGroup
	var mu sync.Mutex
	var failedCount int

	for i, chunk := range chunks {
		wg.Add(1)
		go func(idx int, text string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			result := formatChunk(ctx, cm, cfg, idx, text, perChunkTimeout)

			mu.Lock()
			if result == "" {
				failedCount++
			}
			if result != "" {
				results[idx] = result
			}
			mu.Unlock()
		}(i, chunk)
	}
	wg.Wait()

	if failedCount > 0 && !fallback {
		return nil
	}
	if failedCount == len(chunks) && fallback {
		slog.Warn("llm.format.chunk_failed_fallback", "err", errChunkFailed)
		return []string{rawText}
	}
	if failedCount > 0 {
		slog.Warn("llm.format.chunk_partial_fallback", "failed_chunks", failedCount, "total_chunks", len(chunks))
	}

	formatted := make([]string, 0, len(results))
	for _, text := range results {
		if text != "" {
			formatted = append(formatted, text)
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
	ctx, cancel := context.WithTimeout(context.WithoutCancel(parentCtx), timeout)
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

func formatTwoStep(
	ctx context.Context,
	cm einomodel.BaseChatModel,
	rawText string,
	cfg appconfig.LLMConfig,
	perChunkTimeout time.Duration,
	fallback bool,
) string {
	start := time.Now()

	// Step 1: split into smaller chunks, fix typos + trad→simp per chunk
	chunks := splitByRunes(rawText, cfg.Step1ChunkRunes)
	slog.Info("llm.format.step1.start", "chunks", len(chunks), "chunk_runes", cfg.Step1ChunkRunes)

	step1Cfg := appconfig.LLMConfig{
		Prompt:      cfg.Step1Prompt,
		Temperature: cfg.Temperature,
		MaxTokens:   cfg.MaxTokens,
	}
	step1Results := formatChunksParallel(ctx, cm, chunks, rawText, step1Cfg, perChunkTimeout, fallback)
	if step1Results == nil {
		return ""
	}
	step1Text := strings.Join(step1Results, "\n\n")
	slog.Info("llm.format.step1.done", "elapsed", time.Since(start))

	// Step 2: semantic paragraph organization, chunked to avoid OOM on long texts
	step2Chunks := splitByRunes(step1Text, cfg.ChunkRunes)
	slog.Info("llm.format.step2.start", "chunks", len(step2Chunks), "chunk_runes", cfg.ChunkRunes)
	step2Results := formatChunksParallel(ctx, cm, step2Chunks, step1Text, cfg, perChunkTimeout, fallback)
	if step2Results == nil {
		return ""
	}
	result := strings.Join(step2Results, "\n\n")
	slog.Info("llm.format.done", "elapsed", time.Since(start))
	return result
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
