package rag

import (
	"strings"
	"testing"
)

func TestCleanSourceTextPlainTextFiltersNoiseAndUnwrapsLines(t *testing.T) {
	input := strings.Join([]string{
		"\ufeffTitle",
		"====",
		"",
		"This line is wrapped without",
		"terminal punctuation and continues here.",
		"",
		"Page 1 of 9",
		"第 2 页",
		"",
		"中文内容没有标点",
		"继续保持同一句。",
		"",
		"- item one",
		"- item two",
		"",
		"!!!@@@###",
		"Useful final sentence.",
	}, "\r\n")

	got := CleanSourceText(input, ContentFormatPlain)

	for _, unwanted := range []string{"\ufeff", "====", "Page 1 of 9", "第 2 页", "!!!@@@###"} {
		if strings.Contains(got, unwanted) {
			t.Fatalf("expected %q to be filtered from %q", unwanted, got)
		}
	}
	if !strings.Contains(got, "This line is wrapped without terminal punctuation and continues here.") {
		t.Fatalf("expected hard-wrapped English prose to be joined, got %q", got)
	}
	if !strings.Contains(got, "中文内容没有标点继续保持同一句。") {
		t.Fatalf("expected hard-wrapped CJK prose to be joined without a space, got %q", got)
	}
	if !strings.Contains(got, "- item one\n- item two") {
		t.Fatalf("expected list lines to stay separate, got %q", got)
	}
}

func TestCleanSourceTextMarkdownFiltersMetadataWithoutTouchingCodeFences(t *testing.T) {
	input := strings.Join([]string{
		"---",
		"title: Guide",
		"draft: false",
		"---",
		"",
		"# Guide",
		"",
		"<!-- hidden comment -->",
		"Intro [linked][ref].",
		"",
		"<script>tracking()</script>",
		"[ref]: https://example.com",
		"[^1]: Footnote content remains useful.",
		"",
		"```txt",
		"<!-- keep inside code -->",
		"[ref]: keep inside code",
		"```",
	}, "\n")

	got := CleanSourceText(input, ContentFormatMarkdown)

	for _, unwanted := range []string{"title: Guide", "draft: false", "hidden comment", "tracking()", "[ref]: https://example.com"} {
		if strings.Contains(got, unwanted) {
			t.Fatalf("expected %q to be filtered from %q", unwanted, got)
		}
	}
	for _, wanted := range []string{
		"# Guide",
		"Intro [linked][ref].",
		"[^1]: Footnote content remains useful.",
		"```txt\n<!-- keep inside code -->\n[ref]: keep inside code\n```",
	} {
		if !strings.Contains(got, wanted) {
			t.Fatalf("expected %q to be preserved in %q", wanted, got)
		}
	}
}

func TestDocumentsFromSourceCleansBeforeParsingMarkdown(t *testing.T) {
	docs, format := DocumentsFromSource(Source{
		Text: strings.Join([]string{
			"---",
			"title: Guide",
			"---",
			"",
			"# Guide",
			"",
			"<!-- generated -->",
			"Useful body.",
		}, "\n"),
		Filename: "guide.md",
	})

	if format != ContentFormatMarkdown {
		t.Fatalf("format = %q, want markdown", format)
	}
	if len(docs) != 1 {
		t.Fatalf("expected one cleaned markdown document, got %#v", docs)
	}
	if strings.Contains(docs[0].PageContent, "title: Guide") || strings.Contains(docs[0].PageContent, "generated") {
		t.Fatalf("expected markdown metadata noise to be removed, got %q", docs[0].PageContent)
	}
	if !strings.Contains(docs[0].PageContent, "# Guide\n\nUseful body.") {
		t.Fatalf("expected cleaned markdown body, got %q", docs[0].PageContent)
	}
}
