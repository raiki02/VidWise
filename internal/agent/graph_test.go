package agent

import (
	"context"
	"errors"
	"testing"

	einotool "github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"
	"github.com/raiki02/vidwise/internal/tool"
)

type fakeGraphTool struct {
	name   string
	output string
	err    error
}

func (f fakeGraphTool) Info(context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{Name: f.name}, nil
}

func (f fakeGraphTool) InvokableRun(context.Context, string, ...einotool.Option) (string, error) {
	if f.err != nil {
		return "", f.err
	}
	return f.output, nil
}

type recordingVideoObserver struct {
	events []string
}

func (o *recordingVideoObserver) StepStarted(name string) {
	o.events = append(o.events, "start:"+name)
}

func (o *recordingVideoObserver) StepDone(name string) {
	o.events = append(o.events, "done:"+name)
}

func (o *recordingVideoObserver) StepFailed(name string, err error) {
	o.events = append(o.events, "failed:"+name+":"+err.Error())
}

func (o *recordingVideoObserver) StepSkipped(name, reason string) {
	o.events = append(o.events, "skipped:"+name+":"+reason)
}

func TestExecuteVideoProcessReportsStepProgress(t *testing.T) {
	registry := tool.NewRegistry()
	registerGraphTool(registry, "extract_audio", "")
	registerGraphTool(registry, "transcribe_audio", `{"text":"raw transcript","language":"zh","duration":1}`)
	registerGraphTool(registry, "format_transcript", `{"formatted_text":"formatted transcript"}`)
	registerGraphTool(registry, "rag_index", `{"status":"ok"}`)
	observer := &recordingVideoObserver{}

	got, err := ExecuteVideoProcessWithObserver(context.Background(), registry, "https://example.com/v", "/tmp", "demo", "u1", "s1", "task-1", "zh", observer)
	if err != nil {
		t.Fatalf("ExecuteVideoProcessWithObserver returned error: %v", err)
	}
	if got != "formatted transcript" {
		t.Fatalf("formatted text = %q, want formatted transcript", got)
	}

	want := []string{
		"start:download_audio",
		"done:download_audio",
		"start:transcribe_audio",
		"done:transcribe_audio",
		"start:format_transcript",
		"done:format_transcript",
		"start:index_knowledge_base",
		"done:index_knowledge_base",
	}
	if !equalStrings(observer.events, want) {
		t.Fatalf("events = %#v, want %#v", observer.events, want)
	}
}

func TestExecuteVideoProcessReportsOptionalStepSkips(t *testing.T) {
	registry := tool.NewRegistry()
	registerGraphTool(registry, "extract_audio", "")
	registerGraphTool(registry, "transcribe_audio", `{"text":"raw transcript","language":"zh","duration":1}`)
	observer := &recordingVideoObserver{}

	got, err := ExecuteVideoProcessWithObserver(context.Background(), registry, "https://example.com/v", "/tmp", "demo", "u1", "s1", "task-1", "zh", observer)
	if err != nil {
		t.Fatalf("ExecuteVideoProcessWithObserver returned error: %v", err)
	}
	if got != "raw transcript" {
		t.Fatalf("text = %q, want raw transcript", got)
	}

	want := []string{
		"start:download_audio",
		"done:download_audio",
		"start:transcribe_audio",
		"done:transcribe_audio",
		"skipped:format_transcript:format_transcript tool unavailable",
		"skipped:index_knowledge_base:rag_index tool unavailable",
	}
	if !equalStrings(observer.events, want) {
		t.Fatalf("events = %#v, want %#v", observer.events, want)
	}
}

func TestExecuteVideoProcessReportsFailedStep(t *testing.T) {
	registry := tool.NewRegistry()
	registerGraphTool(registry, "extract_audio", "")
	registry.Register("transcribe_audio", fakeGraphTool{name: "transcribe_audio", err: errors.New("asr down")}, nil)
	observer := &recordingVideoObserver{}

	_, err := ExecuteVideoProcessWithObserver(context.Background(), registry, "https://example.com/v", "/tmp", "demo", "u1", "s1", "task-1", "zh", observer)
	if err == nil {
		t.Fatal("expected pipeline error")
	}

	want := []string{
		"start:download_audio",
		"done:download_audio",
		"start:transcribe_audio",
		"failed:transcribe_audio:asr down",
	}
	if !equalStrings(observer.events, want) {
		t.Fatalf("events = %#v, want %#v", observer.events, want)
	}
}

func registerGraphTool(registry *tool.Registry, name, output string) {
	registry.Register(name, fakeGraphTool{name: name, output: output}, nil)
}

func equalStrings(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}
