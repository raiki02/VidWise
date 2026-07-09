package chunk

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestSplitTextMarkdownPreservesHeadingSections(t *testing.T) {
	text := strings.Join([]string{
		"# Alpha",
		"",
		"first paragraph",
		"",
		"# Beta",
		"",
		"second paragraph",
	}, "\n")

	got := SplitText(text, Config{MaxRunes: 28, Format: FormatMarkdown})
	if len(got) != 2 {
		t.Fatalf("expected two section chunks, got %#v", got)
	}
	if !strings.HasPrefix(got[0].Text, "# Alpha") || got[0].Source != SourceSection {
		t.Fatalf("expected first heading section, got %#v", got[0])
	}
	if !strings.HasPrefix(got[1].Text, "# Beta") || got[1].Source != SourceSection {
		t.Fatalf("expected second heading section, got %#v", got[1])
	}
}

func TestSplitTextRecursivelySplitsOversizedParagraphBySentence(t *testing.T) {
	text := "第一句内容很长很长。第二句内容很长很长。第三句内容很长很长。"

	got := SplitText(text, Config{MaxRunes: 12, Format: FormatPlain})
	if len(got) != 3 {
		t.Fatalf("expected three sentence chunks, got %#v", got)
	}
	for _, c := range got {
		if c.Source != SourceSentence {
			t.Fatalf("expected sentence source, got %#v", c)
		}
		if utf8.RuneCountInString(c.Text) > 12 {
			t.Fatalf("chunk exceeded limit: %#v", c)
		}
	}
}

func TestSplitTextFallsBackFromSentenceToClause(t *testing.T) {
	text := "甲乙丙丁戊己庚辛，甲乙丙丁戊己庚辛，甲乙丙丁戊己庚辛"

	got := SplitText(text, Config{MaxRunes: 10, Format: FormatPlain})
	if len(got) != 3 {
		t.Fatalf("expected three clause chunks, got %#v", got)
	}
	for _, c := range got {
		if c.Source != SourceClause {
			t.Fatalf("expected clause source, got %#v", c)
		}
		if utf8.RuneCountInString(c.Text) > 10 {
			t.Fatalf("chunk exceeded limit: %#v", c)
		}
	}
}

func TestSplitTextMarkdownDoesNotSplitFenceWhenBlocksFit(t *testing.T) {
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

	got := SplitText(text, Config{MaxRunes: 70, Format: FormatMarkdown})
	if len(got) != 2 {
		t.Fatalf("expected heading/code and paragraph chunks, got %#v", got)
	}
	if strings.Count(got[0].Text, "```") != 2 {
		t.Fatalf("expected complete fenced code block in first chunk, got %q", got[0].Text)
	}
	if strings.Contains(got[1].Text, "```") {
		t.Fatalf("did not expect code fence in second chunk, got %q", got[1].Text)
	}
}

func TestSplitTextQualityFilterNeverDropsAllNonEmptyChunks(t *testing.T) {
	got := SplitText("!!!\n\n???", Config{MaxRunes: 10, MinRunes: 5, Format: FormatPlain})
	if len(got) == 0 {
		t.Fatal("expected non-empty low-quality content to be retained when everything would be filtered")
	}
}
