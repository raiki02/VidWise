package paragraph

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	einomodel "github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/raiki02/vidwise/internal/appconfig"
)

type fakeChatModel struct {
	mu        sync.Mutex
	respond   func(string) fakeChatResponse
	startHook func()
	doneHook  func()
	calls     int
	options   []*einomodel.Options
}

type fakeChatResponse struct {
	content string
	err     error
	delay   time.Duration
}

func (m *fakeChatModel) Generate(ctx context.Context, input []*schema.Message, opts ...einomodel.Option) (*schema.Message, error) {
	m.mu.Lock()
	m.calls++
	m.options = append(m.options, einomodel.GetCommonOptions(nil, opts...))
	m.mu.Unlock()
	if m.startHook != nil {
		m.startHook()
	}
	var resp fakeChatResponse
	if m.respond != nil {
		chunk := ""
		if len(input) > 0 {
			chunk = input[len(input)-1].Content
		}
		resp = m.respond(chunk)
	}

	if resp.delay > 0 {
		select {
		case <-time.After(resp.delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if resp.err != nil {
		if m.doneHook != nil {
			m.doneHook()
		}
		return nil, resp.err
	}
	if m.doneHook != nil {
		m.doneHook()
	}
	return schema.AssistantMessage(resp.content, nil), nil
}

func (m *fakeChatModel) lastOptions() *einomodel.Options {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.options) == 0 {
		return nil
	}
	return m.options[len(m.options)-1]
}

func (m *fakeChatModel) Stream(_ context.Context, _ []*schema.Message, _ ...einomodel.Option) (*schema.StreamReader[*schema.Message], error) {
	return nil, errors.New("stream not implemented")
}

func TestFormatChunksParallelIgnoresParentCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	model := &fakeChatModel{respond: func(chunk string) fakeChatResponse {
		return fakeChatResponse{content: strings.ToUpper(chunk), delay: 10 * time.Millisecond}
	}}
	cfg := appconfig.LLMConfig{
		Prompt: appconfig.PromptConfig{
			System:       "system",
			UserTemplate: "{{text}}",
		},
		Temperature: 0.2,
		MaxTokens:   16,
	}

	got := formatChunksParallel(ctx, model, []string{"a", "b"}, "a\n\nb", cfg, 100*time.Millisecond, false)
	if got == nil {
		t.Fatalf("expected formatted chunks, got nil")
	}
	if strings.Join(got, "\n\n") != "A\n\nB" {
		t.Fatalf("unexpected formatted output: %q", strings.Join(got, "\n\n"))
	}
}

func TestFormatChunksParallelLimitsConcurrency(t *testing.T) {
	var active int32
	var maxActive int32
	model := &fakeChatModel{
		respond: func(chunk string) fakeChatResponse {
			return fakeChatResponse{content: strings.ToUpper(chunk), delay: 25 * time.Millisecond}
		},
		startHook: func() {
			now := atomic.AddInt32(&active, 1)
			for {
				peak := atomic.LoadInt32(&maxActive)
				if now <= peak || atomic.CompareAndSwapInt32(&maxActive, peak, now) {
					break
				}
			}
		},
		doneHook: func() {
			atomic.AddInt32(&active, -1)
		},
	}
	cfg := appconfig.LLMConfig{
		Prompt: appconfig.PromptConfig{
			System:       "system",
			UserTemplate: "{{text}}",
		},
		Temperature: 0.2,
		MaxTokens:   16,
	}

	chunks := []string{"a", "b", "c", "d", "e", "f", "g", "h"}
	got := formatChunksParallel(context.Background(), model, chunks, strings.Join(chunks, "\n\n"), cfg, time.Second, false)
	if got == nil {
		t.Fatal("expected formatted chunks")
	}
	if atomic.LoadInt32(&maxActive) > maxParallelChunks {
		t.Fatalf("max concurrency = %d, want <= %d", maxActive, maxParallelChunks)
	}
	if strings.Join(got, "\n\n") != "A\n\nB\n\nC\n\nD\n\nE\n\nF\n\nG\n\nH" {
		t.Fatalf("unexpected ordering: %q", strings.Join(got, "\n\n"))
	}
}

func TestFormatChunksParallelFallsBackPerChunk(t *testing.T) {
	model := &fakeChatModel{respond: func(chunk string) fakeChatResponse {
		switch chunk {
		case "first":
			return fakeChatResponse{err: errors.New("first chunk failed")}
		case "second":
			return fakeChatResponse{content: "B"}
		default:
			return fakeChatResponse{err: errors.New("unexpected chunk")}
		}
	}}
	cfg := appconfig.LLMConfig{
		Prompt: appconfig.PromptConfig{
			System:       "system",
			UserTemplate: "{{text}}",
		},
		Temperature: 0.2,
		MaxTokens:   16,
	}

	got := formatChunksParallel(context.Background(), model, []string{"first", "second"}, "first\n\nsecond", cfg, time.Second, true)
	if got == nil {
		t.Fatalf("expected partial fallback result, got nil")
	}
	if strings.Join(got, "\n\n") != "first\n\nB" {
		t.Fatalf("unexpected fallback output: %q", strings.Join(got, "\n\n"))
	}
}

func TestFormatChunksParallelFallsBackToRawWhenAllChunksFail(t *testing.T) {
	model := &fakeChatModel{respond: func(string) fakeChatResponse {
		return fakeChatResponse{err: errors.New("chunk failed")}
	}}
	cfg := appconfig.LLMConfig{
		Prompt: appconfig.PromptConfig{
			System:       "system",
			UserTemplate: "{{text}}",
		},
		Temperature: 0.2,
		MaxTokens:   16,
	}

	got := formatChunksParallel(context.Background(), model, []string{"first", "second"}, "first\n\nsecond", cfg, time.Second, true)
	if len(got) != 1 || got[0] != "first\n\nsecond" {
		t.Fatalf("unexpected raw fallback output: %#v", got)
	}
}

func TestFormatChunkOmitsSamplingParamsWhenDisabled(t *testing.T) {
	disableSamplingParams := true
	model := &fakeChatModel{respond: func(string) fakeChatResponse {
		return fakeChatResponse{content: "formatted"}
	}}
	cfg := appconfig.LLMConfig{
		Prompt: appconfig.PromptConfig{
			System:       "system",
			UserTemplate: "{{text}}",
		},
		Temperature:           0.2,
		DisableSamplingParams: &disableSamplingParams,
		MaxTokens:             16,
	}

	got := formatChunk(context.Background(), model, cfg, 0, "raw", time.Second)
	if got != "formatted" {
		t.Fatalf("formatChunk() = %q, want formatted", got)
	}
	opts := model.lastOptions()
	if opts == nil {
		t.Fatal("expected recorded options")
	}
	if opts.Temperature != nil {
		t.Fatalf("expected no temperature option, got %v", *opts.Temperature)
	}
	if opts.MaxTokens == nil || *opts.MaxTokens != 16 {
		t.Fatalf("MaxTokens = %v, want 16", opts.MaxTokens)
	}
}

func TestFormatChunkPassesSamplingParamsByDefault(t *testing.T) {
	model := &fakeChatModel{respond: func(string) fakeChatResponse {
		return fakeChatResponse{content: "formatted"}
	}}
	cfg := appconfig.LLMConfig{
		Prompt: appconfig.PromptConfig{
			System:       "system",
			UserTemplate: "{{text}}",
		},
		Temperature: 0.2,
		MaxTokens:   16,
	}

	got := formatChunk(context.Background(), model, cfg, 0, "raw", time.Second)
	if got != "formatted" {
		t.Fatalf("formatChunk() = %q, want formatted", got)
	}
	opts := model.lastOptions()
	if opts == nil {
		t.Fatal("expected recorded options")
	}
	if opts.Temperature == nil || *opts.Temperature != 0.2 {
		t.Fatalf("Temperature = %v, want 0.2", opts.Temperature)
	}
}

func TestSplitMarkdownByRunesKeepsSectionsTogether(t *testing.T) {
	text := strings.Join([]string{
		"# Intro",
		"",
		"Short opening paragraph.",
		"",
		"## Details",
		"",
		"- first point",
		"- second point",
		"",
		"## Next",
		"",
		"Another paragraph.",
	}, "\n")

	got := splitForLLM(text, 80, TextFormatMarkdown)
	if len(got) != 2 {
		t.Fatalf("expected two markdown chunks, got %#v", got)
	}
	if !strings.Contains(got[0], "# Intro") || !strings.Contains(got[0], "## Details") {
		t.Fatalf("expected first chunk to keep adjacent sections, got %q", got[0])
	}
	if !strings.HasPrefix(got[1], "## Next") {
		t.Fatalf("expected second chunk to start at heading, got %q", got[1])
	}
}

func TestSplitMarkdownByRunesDoesNotSplitFenceAsBlock(t *testing.T) {
	text := strings.Join([]string{
		"# Notes",
		"",
		"```go",
		"func main() {",
		`	fmt.Println("hello")`,
		"}",
		"```",
		"",
		"After the code block.",
	}, "\n")

	got := splitForLLM(text, 70, TextFormatMarkdown)
	if len(got) != 2 {
		t.Fatalf("expected heading/code and paragraph chunks, got %#v", got)
	}
	if strings.Count(got[0], "```") != 2 {
		t.Fatalf("expected complete fenced code block in first chunk, got %q", got[0])
	}
	if strings.Contains(got[1], "```") {
		t.Fatalf("did not expect code fence to be split into second chunk, got %q", got[1])
	}
}

func BenchmarkFormatChunksParallel(b *testing.B) {
	runFormatChunksBenchmark(b, false)
}

func BenchmarkLegacyFormatChunksParallel(b *testing.B) {
	runFormatChunksBenchmark(b, true)
}

func runFormatChunksBenchmark(b *testing.B, legacy bool) {
	quietLogger := slog.New(slog.NewTextHandler(io.Discard, nil))
	prevLogger := slog.Default()
	slog.SetDefault(quietLogger)
	b.Cleanup(func() {
		slog.SetDefault(prevLogger)
	})

	cfg := appconfig.LLMConfig{
		Prompt: appconfig.PromptConfig{
			System:       "system",
			UserTemplate: "{{text}}",
		},
		Temperature: 0.2,
		MaxTokens:   16,
	}
	chunks := make([]string, 64)
	for i := range chunks {
		chunks[i] = strings.Repeat("chunk", 8)
	}
	text := strings.Join(chunks, "\n\n")
	model := benchChatModel{}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if legacy {
			_ = legacyFormatChunksParallel(context.Background(), model, chunks, text, cfg, time.Second, false)
			continue
		}
		_ = formatChunksParallel(context.Background(), model, chunks, text, cfg, time.Second, false)
	}
}

type benchChatModel struct{}

func (benchChatModel) Generate(_ context.Context, input []*schema.Message, _ ...einomodel.Option) (*schema.Message, error) {
	chunk := ""
	if len(input) > 0 {
		chunk = input[len(input)-1].Content
	}
	return schema.AssistantMessage(strings.ToUpper(chunk), nil), nil
}

func (benchChatModel) Stream(_ context.Context, _ []*schema.Message, _ ...einomodel.Option) (*schema.StreamReader[*schema.Message], error) {
	return nil, errors.New("stream not implemented")
}

func legacyFormatChunksParallel(
	ctx context.Context,
	cm einomodel.BaseChatModel,
	chunks []string,
	rawText string,
	cfg appconfig.LLMConfig,
	perChunkTimeout time.Duration,
	fallback bool,
) []string {
	if len(chunks) == 0 {
		return nil
	}
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
		return []string{rawText}
	}
	formatted := make([]string, 0, len(results))
	for _, text := range results {
		if text != "" {
			formatted = append(formatted, text)
		}
	}
	return formatted
}
