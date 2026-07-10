package rag

import (
	"strings"
	"testing"
)

func TestFormatChunkSourceIncludesTraceableMetadata(t *testing.T) {
	got := FormatChunkSource(RelevantChunk{
		SourceName:  "guide.md",
		HeadingPath: "Guide > Install",
		SourceURL:   "https://example.com/video",
		TaskID:      "task-1",
		ChunkID:     "chunk-1",
	})

	want := "guide.md / Guide > Install / https://example.com/video / task:task-1 / chunk:chunk-1"
	if got != want {
		t.Fatalf("FormatChunkSource() = %q, want %q", got, want)
	}
}

func TestPackContextFormatsNumberedSourceLabelledSnippets(t *testing.T) {
	got := PackContext([]RelevantChunk{
		{
			Text:        "Install the CLI.",
			Score:       0.91,
			SourceName:  "guide.md",
			HeadingPath: "Guide > Install",
		},
		{
			Text:          "Run the command.",
			Score:         0.73,
			DocumentTitle: "Guide",
		},
	}, ContextConfig{MaxRunes: 1000})

	if !got.HasContext() {
		t.Fatal("expected packed context")
	}
	if got.UsedChunks != 2 {
		t.Fatalf("UsedChunks = %d, want 2", got.UsedChunks)
	}
	if len(got.Citations) != 2 {
		t.Fatalf("citations = %#v, want 2 entries", got.Citations)
	}
	if got.Citations[0].SnippetNumber != 1 || got.Citations[0].Chunk.SourceName != "guide.md" {
		t.Fatalf("unexpected first citation: %#v", got.Citations[0])
	}
	if got.Citations[1].SnippetNumber != 2 || got.Citations[1].Chunk.DocumentTitle != "Guide" {
		t.Fatalf("unexpected second citation: %#v", got.Citations[1])
	}
	if got.Truncated {
		t.Fatal("expected untruncated context")
	}
	for _, want := range []string{
		"[片段 1]",
		"来源: guide.md / Guide > Install",
		"相关度: 0.910",
		"Install the CLI.",
		"[片段 2]",
		"来源: Guide",
	} {
		if !strings.Contains(got.Text, want) {
			t.Fatalf("packed context missing %q:\n%s", want, got.Text)
		}
	}
}

func TestPackContextEnforcesRuneBudget(t *testing.T) {
	got := PackContext([]RelevantChunk{
		{
			Text:       strings.Repeat("界", 200),
			Score:      0.99,
			SourceName: "long.md",
		},
		{
			Text:  "second chunk",
			Score: 0.8,
		},
	}, ContextConfig{MaxRunes: 80})

	if got.UsedChunks != 1 {
		t.Fatalf("UsedChunks = %d, want 1", got.UsedChunks)
	}
	if len(got.Citations) != 1 {
		t.Fatalf("citations = %#v, want only the chunk that reached context", got.Citations)
	}
	if got.Citations[0].SnippetNumber != 1 || got.Citations[0].Chunk.SourceName != "long.md" {
		t.Fatalf("unexpected citation for truncated context: %#v", got.Citations[0])
	}
	if !got.Truncated {
		t.Fatal("expected truncated context")
	}
	if runeLen(got.Text) > 80 {
		t.Fatalf("packed context exceeds budget: %d", runeLen(got.Text))
	}
	if !strings.Contains(got.Text, "[片段已截断]") {
		t.Fatalf("missing truncation marker:\n%s", got.Text)
	}
	if strings.Contains(got.Text, "second chunk") {
		t.Fatalf("expected later chunks to be omitted after truncation:\n%s", got.Text)
	}
}

func TestPackContextSkipsEmptyChunks(t *testing.T) {
	got := PackContext([]RelevantChunk{
		{Text: "   "},
		{Text: "Useful.", Score: 0.6},
	}, ContextConfig{MaxRunes: 1000})

	if got.UsedChunks != 1 {
		t.Fatalf("UsedChunks = %d, want 1", got.UsedChunks)
	}
	if strings.Contains(got.Text, "[片段 1]\n相关度: 0.000\n\n") {
		t.Fatalf("empty chunk leaked into context:\n%s", got.Text)
	}
	if !strings.Contains(got.Text, "Useful.") {
		t.Fatalf("expected useful chunk:\n%s", got.Text)
	}
}
