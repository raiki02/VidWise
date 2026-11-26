package agent

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	einomodel "github.com/cloudwego/eino/components/model"
	einotool "github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"
	"github.com/raiki02/vidwise/internal/asr"
	apptool "github.com/raiki02/vidwise/internal/tool"
)

// fakeToolCallingModel streams a fixed script: first a tool call message, then
// a final assistant answer. It lets the react agent run end-to-end without a
// real LLM.
type fakeToolCallingModel struct {
	first  *schema.Message
	second *schema.Message
	calls  int
}

func (m *fakeToolCallingModel) Generate(_ context.Context, msgs []*schema.Message, _ ...einomodel.Option) (*schema.Message, error) {
	m.calls++
	if m.calls == 1 {
		return m.first, nil
	}
	return m.second, nil
}

func (m *fakeToolCallingModel) Stream(_ context.Context, _ []*schema.Message, _ ...einomodel.Option) (*schema.StreamReader[*schema.Message], error) {
	m.calls++
	if m.calls == 1 {
		return schema.StreamReaderFromArray([]*schema.Message{m.first}), nil
	}
	return schema.StreamReaderFromArray([]*schema.Message{m.second}), nil
}

func (m *fakeToolCallingModel) WithTools(_ []*schema.ToolInfo) (einomodel.ToolCallingChatModel, error) {
	return m, nil
}

func toolCallMessage(name, args string) *schema.Message {
	return &schema.Message{
		Role: schema.Assistant,
		ToolCalls: []schema.ToolCall{{
			ID:   "call-1",
			Type: "function",
			Function: schema.FunctionCall{
				Name:      name,
				Arguments: args,
			},
		}},
	}
}

type fakeTranscriber struct {
	resp asr.TranscribeResponse
}

func (f *fakeTranscriber) Transcribe(_ context.Context, _, _ string) (asr.TranscribeResponse, error) {
	return f.resp, nil
}

func TestNewValidates(t *testing.T) {
	if _, err := New(context.Background(), nil, nil, 10); err == nil {
		t.Fatal("expected error for nil model")
	}
	fake := &fakeToolCallingModel{}
	if _, err := New(context.Background(), fake, nil, 10); err == nil {
		t.Fatal("expected error for no tools")
	}
}

func TestAgentRunsToolThenAnswers(t *testing.T) {
	dir := t.TempDir()
	audioPath := dir + "/a.mp3"
	if err := os.WriteFile(audioPath, []byte("fake"), 0o644); err != nil {
		t.Fatal(err)
	}

	asrTool, err := apptool.NewTranscribeTool(dir, &fakeTranscriber{
		resp: asr.TranscribeResponse{Text: "这是转写内容", Duration: 3},
	})
	if err != nil {
		t.Fatal(err)
	}
	args, _ := json.Marshal(map[string]string{"audio_path": audioPath})
	fake := &fakeToolCallingModel{
		first:  toolCallMessage("transcribe_audio", string(args)),
		second: &schema.Message{Role: schema.Assistant, Content: "转写完成：这是转写内容"},
	}

	agent, err := New(context.Background(), fake, []einotool.BaseTool{asrTool}, 10)
	if err != nil {
		t.Fatal(err)
	}
	out, err := agent.Generate(context.Background(), []*schema.Message{schema.UserMessage("请转写这个音频")})
	if err != nil {
		t.Fatal(err)
	}
	if out.Content != "转写完成：这是转写内容" {
		t.Errorf("content = %q", out.Content)
	}
	if fake.calls < 2 {
		t.Errorf("expected at least 2 model calls, got %d", fake.calls)
	}
}