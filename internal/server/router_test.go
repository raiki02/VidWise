package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/capability"
	"github.com/raiki02/vidwise/internal/knowledgeagent"
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

func TestCapabilitiesEndpointExposesFrontendManifest(t *testing.T) {
	engine := Router(
		testRouterConfig(),
		tool.NewRegistry(),
		testRouterRAGRuntime(),
		nil,
		nil,
		testRouterCapabilities(),
	)

	req := httptest.NewRequest(http.MethodGet, "/api/capabilities", nil)
	resp := httptest.NewRecorder()
	engine.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var out FrontendManifest
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out.Version == "" {
		t.Fatal("manifest version is empty")
	}
	if len(out.Features) == 0 || len(out.ExtractTypes) == 0 || len(out.VideoProcessSteps) == 0 {
		t.Fatalf("manifest missing core sections: %#v", out)
	}
	if frontendFeatureByID(t, out, "sources").Status != capability.Degraded {
		t.Fatalf("sources feature should reflect degraded RAG: %#v", frontendFeatureByID(t, out, "sources"))
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

func TestRouterMountsTaskTranscriptIndexRoute(t *testing.T) {
	engine := Router(
		testRouterConfig(),
		tool.NewRegistry(),
		ragruntime.Runtime{},
		nil,
		nil,
		capability.FromRuntime(capability.RuntimeDeps{}),
	)

	req := httptest.NewRequest(http.MethodPost, "/task/missing/index", nil)
	resp := httptest.NewRecorder()
	engine.ServeHTTP(resp, req)

	if resp.Code != http.StatusNotFound {
		t.Fatalf("expected mounted route to return 404 from handler, got %d: %s", resp.Code, resp.Body.String())
	}
}

func TestRouterMountsAgentTurnRoute(t *testing.T) {
	engine := Router(
		testRouterConfig(),
		tool.NewRegistry(),
		ragruntime.Runtime{},
		nil,
		nil,
		capability.FromRuntime(capability.RuntimeDeps{}),
	)

	body := bytes.NewBufferString(`{"user_id":"u1","message":"【演示视频】 https://example.com/watch?v=1"}`)
	req := httptest.NewRequest(http.MethodPost, "/agent/turn", body)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	engine.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}
	var out knowledgeagent.TurnResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(out.PendingActions) != 1 || out.PendingActions[0].Type != knowledgeagent.ActionProcessVideo {
		t.Fatalf("pending actions = %#v, want process_video", out.PendingActions)
	}
}

func TestRouterMountsAgentConfirmRoute(t *testing.T) {
	engine := Router(
		testRouterConfig(),
		tool.NewRegistry(),
		ragruntime.Runtime{},
		nil,
		nil,
		capability.FromRuntime(capability.RuntimeDeps{}),
	)

	body := bytes.NewBufferString(`{"user_id":"u1"}`)
	req := httptest.NewRequest(http.MethodPost, "/agent/actions/missing/confirm", body)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	engine.ServeHTTP(resp, req)

	if resp.Code != http.StatusNotFound {
		t.Fatalf("expected mounted route to return 404 from handler, got %d: %s", resp.Code, resp.Body.String())
	}
}

func TestStaticIndexExposesCurrentRAGAgentWorkflows(t *testing.T) {
	raw, err := webFS.ReadFile("web/index.html")
	if err != nil {
		t.Fatalf("read embedded index: %v", err)
	}
	body := string(raw)
	for _, want := range []string{
		"/agent/turn",
		"/agent/actions/",
		"/confirm",
		"/video/process",
		"/task/",
		"/index",
		"/tasks?",
		"/rag/sources?",
		"/rag/source/",
		"source_ids",
		"id=\"source-filter\"",
		"id=\"chat-source-picker\"",
		"document_ids",
		"task_ids",
		"heading_paths",
		"/api/capabilities",
		"manifest-pill",
		"feature-status-list",
		"data-nav=\"extract\"",
		"id=\"extract-form\"",
		"function startExtract",
		"contentDispositionFilename",
		"data-nav=\"memory\"",
		"id=\"memory-table\"",
		"function loadMemoryFacts",
		"data-doc-filter",
		"rag_answer_status",
		"rag_context_chunks",
		"data-nav=\"sources\"",
		"id=\"task-list\"",
		"data-task-download-text",
		"下载格式化文本",
		"data-task-index",
		"class=\"task-knowledge\"",
		"可存入知识库",
		"任务完成后可判断",
		"function localUserId()",
		"local-user-",
		"本地用户 ID",
		"个人知识库",
		"高级设置",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("static index missing %q", want)
		}
	}

	taskRenderStart := strings.Index(body, "function renderTask(task")
	knowledgeRenderStart := strings.Index(body, "function renderTaskKnowledgeAction")
	transcriptRenderStart := strings.Index(body, "function renderTaskTranscript")
	if taskRenderStart < 0 || knowledgeRenderStart < 0 || transcriptRenderStart < 0 {
		t.Fatal("static index missing task render helpers")
	}
	taskRender := body[taskRenderStart:knowledgeRenderStart]
	if !strings.Contains(taskRender, "${knowledge}") {
		t.Fatal("task card should render knowledge status outside transcript details")
	}
	if strings.Index(taskRender, "${knowledge}") > strings.Index(taskRender, `<div class="steps">${steps}</div>`) {
		t.Fatal("task card should show knowledge status before task steps")
	}

	knowledgeRender := body[knowledgeRenderStart:transcriptRenderStart]
	if !strings.Contains(knowledgeRender, "data-task-index-status") || !strings.Contains(knowledgeRender, "data-task-index") {
		t.Fatal("knowledge status/action should live in the visible knowledge area")
	}

	transcriptRenderEnd := strings.Index(body[transcriptRenderStart:], "function bindTaskActionButtons")
	if transcriptRenderEnd < 0 {
		t.Fatal("static index missing task action binding helper")
	}
	transcriptRender := body[transcriptRenderStart : transcriptRenderStart+transcriptRenderEnd]
	if strings.Contains(transcriptRender, "data-task-index") {
		t.Fatal("transcript details should not hide the knowledge action")
	}
}

func TestTaskTrackerOptionsFromConfig(t *testing.T) {
	opts := taskTrackerOptionsFromConfig(appconfig.Config{
		Task: appconfig.TaskConfig{
			MaxTracked:  7,
			RetainFor:   "2h",
			StoragePath: " .vidwise/test-tasks.json ",
		},
	})

	if opts.MaxTasks != 7 {
		t.Fatalf("MaxTasks = %d, want 7", opts.MaxTasks)
	}
	if opts.RetainFor != 2*time.Hour {
		t.Fatalf("RetainFor = %s, want 2h", opts.RetainFor)
	}
	if opts.StoragePath != ".vidwise/test-tasks.json" {
		t.Fatalf("StoragePath = %q, want .vidwise/test-tasks.json", opts.StoragePath)
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
