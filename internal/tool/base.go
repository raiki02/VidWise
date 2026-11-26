package tool

import (
	"fmt"
	"net/url"
	"path/filepath"
	"strings"
)

// PathInWorkDir reports whether path resolves inside workDir (both absolute).
// It is the security boundary for tool-provided file paths.
func PathInWorkDir(workDir, path string) error {
	workAbs, err := filepath.Abs(workDir)
	if err != nil {
		return fmt.Errorf("resolve work dir: %w", err)
	}
	pathAbs, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("resolve path: %w", err)
	}
	rel, err := filepath.Rel(workAbs, pathAbs)
	if err != nil {
		return fmt.Errorf("path outside work dir: %w", err)
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return fmt.Errorf("path %q escapes work dir %q", path, workAbs)
	}
	return nil
}

// IsHTTPURL reports whether s is an absolute http:// or https:// URL.
func IsHTTPURL(s string) bool {
	s = strings.TrimSpace(s)
	u, err := url.Parse(s)
	if err != nil {
		return false
	}
	return (u.Scheme == "http" || u.Scheme == "https") && u.Host != ""
}
