package tool

import (
	"path/filepath"
	"testing"
)

func TestPathInWorkDirAcceptsInside(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sub", "a.mp3")
	if err := PathInWorkDir(dir, path); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestPathInWorkDirRejectsEscape(t *testing.T) {
	dir := t.TempDir()
	paths := []string{
		"../evil.mp3",
		filepath.Join(dir, "..", "evil.mp3"),
		"/etc/passwd",
	}
	for _, p := range paths {
		if err := PathInWorkDir(dir, p); err == nil {
			t.Errorf("PathInWorkDir(%q) accepted escape", p)
		}
	}
}

func TestPathInWorkDirRejectsRelative(t *testing.T) {
	dir := t.TempDir()
	if err := PathInWorkDir(dir, "relative.mp3"); err == nil {
		t.Error("accepted relative path outside work dir")
	}
}

func TestIsHTTPURL(t *testing.T) {
	cases := map[string]bool{
		"https://example.com/video": true,
		"http://a.b/c":              true,
		"ftp://example.com/x":       false,
		"bilibili.com/video":        false,
		"file:///etc/passwd":        false,
		"javascript:alert(1)":       false,
		"https://":                  false,
		"":                          false,
	}
	for in, want := range cases {
		if got := IsHTTPURL(in); got != want {
			t.Errorf("IsHTTPURL(%q) = %v, want %v", in, got, want)
		}
	}
}
