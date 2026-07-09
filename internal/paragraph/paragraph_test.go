package paragraph

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	einomodel "github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/raiki02/vidwise/internal/appconfig"
)

type fakeChatModel struct {
	mu      sync.Mutex
	respond func(string) fakeChatResponse
	calls   int
}

type fakeChatResponse struct {
	content string
	err     error
	delay   time.Duration
}

func (m *fakeChatModel) Generate(ctx context.Context, input []*schema.Message, _ ...einomodel.Option) (*schema.Message, error) {
	m.mu.Lock()
	m.calls++
	m.mu.Unlock()
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
		return nil, resp.err
	}
	return schema.AssistantMessage(resp.content, nil), nil
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
