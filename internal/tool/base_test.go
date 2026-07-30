package tool

import (
	"context"
	"errors"
	"testing"
	"time"

	einotool "github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"
)

func TestWrapperHonorsMaxRetry(t *testing.T) {
	inner := &failingInvokableTool{err: errors.New("boom")}
	wrapper := NewWrapper(inner, WrapperConfig{
		Name:     "failing_tool",
		MaxRetry: 1,
		Timeout:  time.Second,
	})

	if _, err := wrapper.Run(context.Background(), `{"x":1}`); err == nil {
		t.Fatal("expected wrapped tool to fail")
	}

	if inner.attempts != 2 {
		t.Fatalf("attempts = %d, want 2", inner.attempts)
	}
}

func TestWrapperDelegatesInfo(t *testing.T) {
	inner := &failingInvokableTool{err: errors.New("boom")}
	wrapper := NewWrapper(inner, WrapperConfig{Name: "failing_tool"})

	info, err := wrapper.Info(context.Background())
	if err != nil {
		t.Fatalf("Info returned error: %v", err)
	}
	if info.Name != "failing_tool" {
		t.Fatalf("Info().Name = %q, want failing_tool", info.Name)
	}
	if inner.infoCalls != 1 {
		t.Fatalf("Info calls = %d, want 1", inner.infoCalls)
	}
}

func TestRegistryLoadsInfoFromRegisteredTool(t *testing.T) {
	inner := &failingInvokableTool{err: errors.New("boom")}
	registry := NewRegistry()

	registry.Register("failing_tool", inner, nil)

	info, err := registry.GetInfo("failing_tool")
	if err != nil {
		t.Fatalf("GetInfo returned error: %v", err)
	}
	if info == nil {
		t.Fatal("GetInfo returned nil info")
	}
	if info.Name != "failing_tool" {
		t.Fatalf("Info.Name = %q, want failing_tool", info.Name)
	}
	if inner.infoCalls != 1 {
		t.Fatalf("Info calls = %d, want 1", inner.infoCalls)
	}
}

func TestRegistryKeepsExplicitInfo(t *testing.T) {
	inner := &failingInvokableTool{err: errors.New("boom")}
	registry := NewRegistry()
	explicit := &schema.ToolInfo{Name: "explicit_tool", Desc: "explicit metadata"}

	registry.Register("failing_tool", inner, explicit)

	info, err := registry.GetInfo("failing_tool")
	if err != nil {
		t.Fatalf("GetInfo returned error: %v", err)
	}
	if info != explicit {
		t.Fatal("GetInfo did not return the explicit info")
	}
	if inner.infoCalls != 0 {
		t.Fatalf("Info calls = %d, want 0", inner.infoCalls)
	}
}

type failingInvokableTool struct {
	attempts  int
	infoCalls int
	err       error
}

func (f *failingInvokableTool) Info(context.Context) (*schema.ToolInfo, error) {
	f.infoCalls++
	return &schema.ToolInfo{Name: "failing_tool"}, nil
}

func (f *failingInvokableTool) InvokableRun(context.Context, string, ...einotool.Option) (string, error) {
	f.attempts++
	return "", f.err
}
