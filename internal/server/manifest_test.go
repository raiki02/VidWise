package server

import (
	"reflect"
	"testing"

	"github.com/raiki02/vidwise/internal/capability"
	"github.com/raiki02/vidwise/internal/tool"
)

func TestBuildFrontendManifestReflectsCapabilities(t *testing.T) {
	registry := tool.NewRegistry()
	registry.Register("zeta_tool", nil, nil)
	registry.Register("alpha_tool", nil, nil)

	caps := capability.FromRuntime(capability.RuntimeDeps{
		ChatSessionStore: true,
		MemoryStore:      false,
		VectorStore:      true,
		VectorCollection: true,
		Embedding:        true,
		Rerank:           false,
		ASR:              false,
		VideoSummary:     true,
		LLMConfig:        testRouterConfig().LLM,
	})

	got := BuildFrontendManifest(registry, caps)
	if got.Version == "" {
		t.Fatal("Version is empty")
	}
	if !reflect.DeepEqual(got.Tools, []string{"alpha_tool", "zeta_tool"}) {
		t.Fatalf("Tools = %#v, want sorted names", got.Tools)
	}
	if len(got.VideoProcessSteps) == 0 {
		t.Fatal("VideoProcessSteps is empty")
	}

	chat := frontendFeatureByID(t, got, "chat")
	if !chat.Available || chat.Status != capability.Degraded {
		t.Fatalf("chat feature = %#v, want available degraded because memory/RAG are optional", chat)
	}

	video := frontendFeatureByID(t, got, "video")
	if video.Available || video.Status != capability.Unavailable {
		t.Fatalf("video feature = %#v, want unavailable without ASR", video)
	}

	extract := frontendFeatureByID(t, got, "extract")
	if !extract.Available || extract.Status != capability.Degraded {
		t.Fatalf("extract feature = %#v, want available degraded when optional extract backends are missing", extract)
	}

	memory := frontendFeatureByID(t, got, "memory")
	if memory.Available || memory.Status != capability.Unavailable {
		t.Fatalf("memory feature = %#v, want unavailable without memory store", memory)
	}

	text := extractTypeByValue(t, got, "text")
	if text.Available || text.Status != capability.Unavailable {
		t.Fatalf("text extract type = %#v, want unavailable without ASR", text)
	}

	transcript := extractTypeByValue(t, got, "transcript")
	if transcript.AliasOf != "text" || transcript.Available || transcript.Status != capability.Unavailable {
		t.Fatalf("transcript extract type = %#v, want alias of unavailable text", transcript)
	}

	summary := extractTypeByValue(t, got, "summary")
	if !summary.Available || summary.Status != capability.Available {
		t.Fatalf("summary extract type = %#v, want available with video summary capability", summary)
	}
}

func frontendFeatureByID(t *testing.T, manifest FrontendManifest, id string) FrontendFeature {
	t.Helper()
	for _, feature := range manifest.Features {
		if feature.ID == id {
			return feature
		}
	}
	t.Fatalf("feature %q not found in %#v", id, manifest.Features)
	return FrontendFeature{}
}

func extractTypeByValue(t *testing.T, manifest FrontendManifest, value string) ExtractTypeSpec {
	t.Helper()
	for _, typ := range manifest.ExtractTypes {
		if typ.Value == value {
			return typ
		}
	}
	t.Fatalf("extract type %q not found in %#v", value, manifest.ExtractTypes)
	return ExtractTypeSpec{}
}
