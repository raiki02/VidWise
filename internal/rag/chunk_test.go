package rag

import (
	"strings"
	"testing"
)

func TestSplitEmpty(t *testing.T) {
	if got := Split("", 10, 0); got != nil {
		t.Errorf("Split(\"\") = %v, want nil", got)
	}
	if got := Split("   \n\t  ", 10, 0); got != nil {
		t.Errorf("Split(whitespace) = %v, want nil", got)
	}
}

func TestSplitShort(t *testing.T) {
	got := Split("hello world", 100, 10)
	if len(got) != 1 || got[0] != "hello world" {
		t.Fatalf("got %v", got)
	}
}

func TestSplitWindows(t *testing.T) {
	text := strings.Repeat("a", 25)
	got := Split(text, 10, 2)
	if len(got) != 3 {
		t.Fatalf("want 3 chunks, got %d: %v", len(got), got)
	}
	if got[0] != "aaaaaaaaaa" || got[1] != "aaaaaaaaaa" || got[2] != "aaaaaaaaa" {
		t.Errorf("chunks = %q", got)
	}
}

func TestSplitOverlaps(t *testing.T) {
	// "abcdefghij" size 6 overlap 2 -> windows [0:6] [4:10]
	got := Split("abcdefghij", 6, 2)
	if len(got) != 2 {
		t.Fatalf("got %v", got)
	}
	if got[0] != "abcdef" || got[1] != "efghij" {
		t.Errorf("chunks = %q", got)
	}
}

func TestSplitOverlapContained(t *testing.T) {
	text := strings.Repeat("x", 32)
	// overlap >= size must not hang and still yields chunks
	got := Split(text, 8, 8)
	if len(got) < 2 {
		t.Fatalf("got %v", got)
	}
	for _, c := range got {
		if len([]rune(c)) > 8 {
			t.Errorf("chunk larger than size: %q", c)
		}
	}
}

func TestSplitCollapsesWhitespace(t *testing.T) {
	got := Split("a   b\n\nc", 100, 0)
	if len(got) != 1 || got[0] != "a b c" {
		t.Fatalf("got %q", got)
	}
}
