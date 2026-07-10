package capability

import (
	"testing"

	"github.com/raiki02/vidwise/internal/appconfig"
)

func TestFromRuntimeMarksRAGDegradedWithoutRerank(t *testing.T) {
	snapshot := FromRuntime(RuntimeDeps{
		VectorStore:      true,
		VectorCollection: true,
		Embedding:        true,
		Rerank:           false,
		LLMConfig:        llmConfig(true, "qwen"),
	})

	rag := snapshot.Get(RAG)
	if rag.Status != Degraded {
		t.Fatalf("expected RAG degraded without rerank, got %#v", rag)
	}
	if !snapshot.Usable(RAG) {
		t.Fatalf("expected degraded RAG to be usable")
	}
	if snapshot.LegacyStatus(RAG) != Available {
		t.Fatalf("expected legacy RAG status to remain available")
	}
}

func TestFromRuntimeMarksRAGUnavailableWithoutEmbedding(t *testing.T) {
	snapshot := FromRuntime(RuntimeDeps{
		VectorStore:      true,
		VectorCollection: true,
		Embedding:        false,
		Rerank:           true,
		LLMConfig:        llmConfig(true, "qwen"),
	})

	rag := snapshot.Get(RAG)
	if rag.Status != Unavailable {
		t.Fatalf("expected RAG unavailable without embedding, got %#v", rag)
	}
	if snapshot.Usable(RAG) {
		t.Fatalf("expected unavailable RAG not to be usable")
	}
}

func TestFromRuntimeMarksRAGUnavailableWithoutVectorCollection(t *testing.T) {
	snapshot := FromRuntime(RuntimeDeps{
		VectorStore:      true,
		VectorCollection: false,
		Embedding:        true,
		Rerank:           true,
		LLMConfig:        llmConfig(true, "qwen"),
	})

	rag := snapshot.Get(RAG)
	if rag.Status != Unavailable {
		t.Fatalf("expected RAG unavailable without vector collection, got %#v", rag)
	}
	if rag.Reason != "qdrant collection unavailable" {
		t.Fatalf("unexpected reason: %#v", rag)
	}
}

func TestFromRuntimeMarksLLMStatesFromConfig(t *testing.T) {
	disabled := FromRuntime(RuntimeDeps{LLMConfig: llmConfig(false, "qwen")}).Get(LLM)
	if disabled.Status != Unavailable {
		t.Fatalf("expected disabled LLM unavailable, got %#v", disabled)
	}

	missingModel := FromRuntime(RuntimeDeps{LLMConfig: llmConfig(true, "")}).Get(LLM)
	if missingModel.Status != Degraded {
		t.Fatalf("expected enabled LLM without model degraded, got %#v", missingModel)
	}

	available := FromRuntime(RuntimeDeps{LLMConfig: llmConfig(true, "qwen")}).Get(LLM)
	if available.Status != Available {
		t.Fatalf("expected configured LLM available, got %#v", available)
	}
}

func TestSnapshotMapUsesStableStringKeys(t *testing.T) {
	snapshot := FromRuntime(RuntimeDeps{LLMConfig: llmConfig(true, "qwen")})
	got := snapshot.Map()

	if got[string(ChatSessionStore)].Name != ChatSessionStore {
		t.Fatalf("expected chat capability keyed by string name, got %#v", got)
	}
	if _, ok := got[string(RAG)]; !ok {
		t.Fatalf("expected rag capability in map, got %#v", got)
	}
	if _, ok := got[string(ASR)]; !ok {
		t.Fatalf("expected asr capability in map, got %#v", got)
	}
	if _, ok := got[string(VideoSummary)]; !ok {
		t.Fatalf("expected video summary capability in map, got %#v", got)
	}
}

func TestFromRuntimeMarksExternalModelServices(t *testing.T) {
	snapshot := FromRuntime(RuntimeDeps{
		ASR:          true,
		VideoSummary: false,
		LLMConfig:    llmConfig(true, "qwen"),
	})

	if got := snapshot.Get(ASR); got.Status != Available {
		t.Fatalf("expected ASR available, got %#v", got)
	}
	video := snapshot.Get(VideoSummary)
	if video.Status != Unavailable {
		t.Fatalf("expected video summary unavailable, got %#v", video)
	}
	if video.Reason != "video summary service unavailable" {
		t.Fatalf("unexpected video summary reason: %#v", video)
	}
}

func llmConfig(enabled bool, model string) appconfig.LLMConfig {
	return appconfig.LLMConfig{
		Enabled: &enabled,
		Model:   model,
	}
}
