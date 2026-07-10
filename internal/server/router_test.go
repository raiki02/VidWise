package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/capability"
	"github.com/raiki02/vidwise/internal/model"
	"github.com/raiki02/vidwise/internal/rag"
	"github.com/raiki02/vidwise/internal/ragruntime"
	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
	"github.com/raiki02/vidwise/internal/tool"
)

func TestHealthExposesCanonicalCapabilities(t *testing.T) {
	engine := Router(
		testRouterConfig(),
		tool.NewRegistry(),
		testRouterRAGRuntime(),
		nil,
		nil,
		testRouterCapabilities(),
	)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	resp := httptest.NewRecorder()
	engine.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var out map[string]any
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	caps := out["capabilities"].(map[string]any)
	rag := caps[string(capability.RAG)].(map[string]any)
	if rag["status"] != string(capability.Degraded) {
		t.Fatalf("expected canonical RAG degraded without rerank, got %#v", rag)
	}
}

func TestReadyReturnsOKWhenRequiredCapabilitiesAreUsable(t *testing.T) {
	engine := Router(
		testRouterConfig(),
		tool.NewRegistry(),
		testRouterRAGRuntime(),
		nil,
		nil,
		testRouterCapabilities(),
	)

	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	resp := httptest.NewRecorder()
	engine.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var out map[string]any
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out["ready"] != true {
		t.Fatalf("expected ready=true, got %#v", out)
	}
}

func TestReadyReturnsUnavailableWhenRAGIsBlocking(t *testing.T) {
	engine := Router(
		testRouterConfig(),
		tool.NewRegistry(),
		ragruntime.Runtime{},
		nil,
		nil,
		capability.FromRuntime(capability.RuntimeDeps{
			LLMConfig: testRouterConfig().LLM,
		}),
	)

	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	resp := httptest.NewRecorder()
	engine.ServeHTTP(resp, req)

	if resp.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status 503, got %d: %s", resp.Code, resp.Body.String())
	}

	var out map[string]any
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out["ready"] != false {
		t.Fatalf("expected ready=false, got %#v", out)
	}
	blocking, ok := out["blocking"].([]any)
	if !ok || len(blocking) == 0 {
		t.Fatalf("expected blocking capabilities, got %#v", out)
	}
}

func TestRAGHealthKeepsLegacyRAGAvailableWhenCanonicalRAGIsDegraded(t *testing.T) {
	engine := Router(
		testRouterConfig(),
		tool.NewRegistry(),
		testRouterRAGRuntime(),
		nil,
		nil,
		testRouterCapabilities(),
	)

	req := httptest.NewRequest(http.MethodGet, "/rag/health", nil)
	resp := httptest.NewRecorder()
	engine.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var out map[string]any
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out["rag"] != string(capability.Available) {
		t.Fatalf("expected legacy rag status available, got %#v", out)
	}
	caps := out["capabilities"].(map[string]any)
	rag := caps[string(capability.RAG)].(map[string]any)
	if rag["status"] != string(capability.Degraded) {
		t.Fatalf("expected canonical RAG degraded without rerank, got %#v", rag)
	}
}

func TestRouterMountsRAGSourceDeleteRoute(t *testing.T) {
	engine := Router(
		testRouterConfig(),
		tool.NewRegistry(),
		ragruntime.Runtime{},
		nil,
		nil,
		capability.FromRuntime(capability.RuntimeDeps{}),
	)

	req := httptest.NewRequest(http.MethodDelete, "/rag/source/source-1", nil)
	resp := httptest.NewRecorder()
	engine.ServeHTTP(resp, req)

	if resp.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected mounted route to return 503 from handler, got %d: %s", resp.Code, resp.Body.String())
	}
}

func TestRouterMountsRAGSourceListRoute(t *testing.T) {
	engine := Router(
		testRouterConfig(),
		tool.NewRegistry(),
		ragruntime.Runtime{},
		nil,
		nil,
		capability.FromRuntime(capability.RuntimeDeps{}),
	)

	req := httptest.NewRequest(http.MethodGet, "/rag/sources?user_id=u1", nil)
	resp := httptest.NewRecorder()
	engine.ServeHTTP(resp, req)

	if resp.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected mounted route to return 503 from handler, got %d: %s", resp.Code, resp.Body.String())
	}
}

func TestRouterMountsTaskListRoute(t *testing.T) {
	engine := Router(
		testRouterConfig(),
		tool.NewRegistry(),
		ragruntime.Runtime{},
		nil,
		nil,
		capability.FromRuntime(capability.RuntimeDeps{}),
	)

	req := httptest.NewRequest(http.MethodGet, "/tasks", nil)
	resp := httptest.NewRecorder()
	engine.ServeHTTP(resp, req)

	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected mounted route to return 400 from handler, got %d: %s", resp.Code, resp.Body.String())
	}
}

func TestTaskTrackerOptionsFromConfig(t *testing.T) {
	opts := taskTrackerOptionsFromConfig(appconfig.Config{
		Task: appconfig.TaskConfig{
			MaxTracked: 7,
			RetainFor:  "2h",
		},
	})

	if opts.MaxTasks != 7 {
		t.Fatalf("MaxTasks = %d, want 7", opts.MaxTasks)
	}
	if opts.RetainFor != 2*time.Hour {
		t.Fatalf("RetainFor = %s, want 2h", opts.RetainFor)
	}
}

func testRouterConfig() appconfig.Config {
	enabled := true
	return appconfig.Config{
		LLM: appconfig.LLMConfig{
			Enabled: &enabled,
			Model:   "qwen",
		},
		Qdrant: appconfig.QdrantConfig{
			Collection: "test_chunks",
		},
	}
}

func testRouterCapabilities() capability.Snapshot {
	return capability.FromRuntime(capability.RuntimeDeps{
		VectorStore:      true,
		VectorCollection: true,
		Embedding:        true,
		Rerank:           false,
		LLMConfig:        testRouterConfig().LLM,
	})
}

func testRouterRAGRuntime() ragruntime.Runtime {
	return ragruntime.Build(context.Background(), testRouterConfig(), ragruntime.Deps{
		Qdrant:           &qdrantclient.Client{},
		Embed:            &model.EmbedClient{},
		EnsureCollection: func(context.Context, *rag.Indexer) error { return nil },
	}).Runtime
}
