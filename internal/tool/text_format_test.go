package tool

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/raiki02/vidwise/internal/appconfig"
)

func TestTextFormatToolReportsDisabledLLMAsUnformatted(t *testing.T) {
	enabled := false
	inner, _, err := NewTextFormatTool(appconfig.LLMConfig{Enabled: &enabled})
	if err != nil {
		t.Fatalf("NewTextFormatTool returned error: %v", err)
	}
	args, err := ToJSON(TextFormatInput{RawText: "raw transcript"})
	if err != nil {
		t.Fatalf("encode args: %v", err)
	}

	resultJSON, err := inner.InvokableRun(context.Background(), args)
	if err != nil {
		t.Fatalf("InvokableRun returned error: %v", err)
	}

	var out TextFormatOutput
	if err := json.Unmarshal([]byte(resultJSON), &out); err != nil {
		t.Fatalf("decode output: %v", err)
	}
	if out.Formatted {
		t.Fatalf("Formatted = true, want false: %#v", out)
	}
	if out.FormattedText != "raw transcript" {
		t.Fatalf("FormattedText = %q, want raw transcript", out.FormattedText)
	}
	if out.Status != "skipped" || out.Reason != "llm disabled" {
		t.Fatalf("unexpected status: %#v", out)
	}
}
