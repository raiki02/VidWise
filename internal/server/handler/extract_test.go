package handler

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/capability"
	"github.com/raiki02/vidwise/internal/rag"
)

func TestUploadTextReturnsCapabilityWhenRAGUnavailable(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h := NewExtractHandler(appconfig.Config{}, nil, nil, capability.FromRuntime(capability.RuntimeDeps{}))

	router := gin.New()
	router.Use(testTraceIDMiddleware("trace-upload-1"))
	router.POST("/upload", h.UploadText)

	req := httptest.NewRequest(http.MethodPost, "/upload", nil)
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status 503, got %d: %s", resp.Code, resp.Body.String())
	}

	var out map[string]any
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	capabilityPayload, ok := out["capability"].(map[string]any)
	if !ok {
		t.Fatalf("expected capability details, got %#v", out)
	}
	if capabilityPayload["name"] != string(capability.RAG) {
		t.Fatalf("expected RAG capability, got %#v", capabilityPayload)
	}
	if capabilityPayload["status"] != string(capability.Unavailable) {
		t.Fatalf("expected unavailable RAG capability, got %#v", capabilityPayload)
	}
	if out["trace_id"] != "trace-upload-1" {
		t.Fatalf("expected trace_id in capability error response, got %#v", out)
	}
}

func TestUploadTextReportsMissingIndexerEvenWhenCapabilityLooksUsable(t *testing.T) {
	gin.SetMode(gin.TestMode)
	caps := capability.FromRuntime(capability.RuntimeDeps{
		VectorStore:      true,
		VectorCollection: true,
		Embedding:        true,
		Rerank:           false,
	})
	h := NewExtractHandler(appconfig.Config{}, nil, nil, caps)

	router := gin.New()
	router.POST("/upload", h.UploadText)

	req := httptest.NewRequest(http.MethodPost, "/upload", bytes.NewBuffer(nil))
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status 503, got %d: %s", resp.Code, resp.Body.String())
	}

	var out map[string]any
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	capabilityPayload, ok := out["capability"].(map[string]any)
	if !ok {
		t.Fatalf("expected capability details, got %#v", out)
	}
	if capabilityPayload["reason"] != "rag indexer unavailable" {
		t.Fatalf("expected missing indexer reason, got %#v", capabilityPayload)
	}
}

func TestUploadTextRequiresScopeBeforeFileParsing(t *testing.T) {
	gin.SetMode(gin.TestMode)
	caps := capability.FromRuntime(capability.RuntimeDeps{
		VectorStore:      true,
		VectorCollection: true,
		Embedding:        true,
		Rerank:           false,
	})
	h := NewExtractHandler(appconfig.Config{}, nil, rag.NewIndexer(nil, nil, "test"), caps)

	router := gin.New()
	router.POST("/upload", h.UploadText)

	req := httptest.NewRequest(http.MethodPost, "/upload", bytes.NewBuffer(nil))
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d: %s", resp.Code, resp.Body.String())
	}
}

func TestDeleteRAGSourceReturnsCapabilityWhenRAGUnavailable(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h := NewExtractHandler(appconfig.Config{}, nil, nil, capability.FromRuntime(capability.RuntimeDeps{}))

	router := gin.New()
	router.DELETE("/rag/source/:source_id", h.DeleteRAGSource)

	req := httptest.NewRequest(http.MethodDelete, "/rag/source/source-1", nil)
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status 503, got %d: %s", resp.Code, resp.Body.String())
	}

	var out map[string]any
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	capabilityPayload, ok := out["capability"].(map[string]any)
	if !ok {
		t.Fatalf("expected capability details, got %#v", out)
	}
	if capabilityPayload["name"] != string(capability.RAG) {
		t.Fatalf("expected RAG capability, got %#v", capabilityPayload)
	}
}

func TestDeleteRAGSourceReportsMissingIndexerEvenWhenCapabilityLooksUsable(t *testing.T) {
	gin.SetMode(gin.TestMode)
	caps := capability.FromRuntime(capability.RuntimeDeps{
		VectorStore:      true,
		VectorCollection: true,
		Embedding:        true,
		Rerank:           false,
	})
	h := NewExtractHandler(appconfig.Config{}, nil, nil, caps)

	router := gin.New()
	router.DELETE("/rag/source/:source_id", h.DeleteRAGSource)

	req := httptest.NewRequest(http.MethodDelete, "/rag/source/source-1", nil)
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status 503, got %d: %s", resp.Code, resp.Body.String())
	}

	var out map[string]any
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	capabilityPayload, ok := out["capability"].(map[string]any)
	if !ok {
		t.Fatalf("expected capability details, got %#v", out)
	}
	if capabilityPayload["reason"] != "rag indexer unavailable" {
		t.Fatalf("expected missing indexer reason, got %#v", capabilityPayload)
	}
}

func TestDeleteRAGSourceRequiresScopeBeforeDeleting(t *testing.T) {
	gin.SetMode(gin.TestMode)
	caps := capability.FromRuntime(capability.RuntimeDeps{
		VectorStore:      true,
		VectorCollection: true,
		Embedding:        true,
		Rerank:           false,
	})
	h := NewExtractHandler(appconfig.Config{}, nil, rag.NewIndexer(nil, nil, "test"), caps)

	router := gin.New()
	router.DELETE("/rag/source/:source_id", h.DeleteRAGSource)

	req := httptest.NewRequest(http.MethodDelete, "/rag/source/source-1", nil)
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d: %s", resp.Code, resp.Body.String())
	}
}

func TestListRAGSourcesReturnsCapabilityWhenRAGUnavailable(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h := NewExtractHandler(appconfig.Config{}, nil, nil, capability.FromRuntime(capability.RuntimeDeps{}))

	router := gin.New()
	router.GET("/rag/sources", h.ListRAGSources)

	req := httptest.NewRequest(http.MethodGet, "/rag/sources?user_id=u1", nil)
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status 503, got %d: %s", resp.Code, resp.Body.String())
	}

	var out map[string]any
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	capabilityPayload, ok := out["capability"].(map[string]any)
	if !ok {
		t.Fatalf("expected capability details, got %#v", out)
	}
	if capabilityPayload["name"] != string(capability.RAG) {
		t.Fatalf("expected RAG capability, got %#v", capabilityPayload)
	}
}

func TestListRAGSourcesReportsMissingCatalogEvenWhenCapabilityLooksUsable(t *testing.T) {
	gin.SetMode(gin.TestMode)
	caps := capability.FromRuntime(capability.RuntimeDeps{
		VectorStore:      true,
		VectorCollection: true,
		Embedding:        true,
		Rerank:           false,
	})
	h := NewExtractHandler(appconfig.Config{}, nil, rag.NewIndexer(nil, nil, "test"), caps)

	router := gin.New()
	router.GET("/rag/sources", h.ListRAGSources)

	req := httptest.NewRequest(http.MethodGet, "/rag/sources?user_id=u1", nil)
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status 503, got %d: %s", resp.Code, resp.Body.String())
	}

	var out map[string]any
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	capabilityPayload, ok := out["capability"].(map[string]any)
	if !ok {
		t.Fatalf("expected capability details, got %#v", out)
	}
	if capabilityPayload["reason"] != "rag source catalog unavailable" {
		t.Fatalf("expected missing catalog reason, got %#v", capabilityPayload)
	}
}

func TestListRAGSourcesRequiresScopeBeforeListing(t *testing.T) {
	gin.SetMode(gin.TestMode)
	caps := capability.FromRuntime(capability.RuntimeDeps{
		VectorStore:      true,
		VectorCollection: true,
		Embedding:        true,
		Rerank:           false,
	})
	h := NewExtractHandlerWithCatalogAndBackground(
		appconfig.Config{},
		nil,
		rag.NewIndexer(nil, nil, "test"),
		rag.NewSourceCatalog(nil, "test"),
		caps,
		nil,
	)

	router := gin.New()
	router.GET("/rag/sources", h.ListRAGSources)

	req := httptest.NewRequest(http.MethodGet, "/rag/sources", nil)
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d: %s", resp.Code, resp.Body.String())
	}
}

func TestDeleteFilterFromRequestPrefersPersonalUserScope(t *testing.T) {
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodDelete, "/rag/source/source-1?user_id=u1&session_id=s1", nil)

	got, err := deleteFilterFromRequest(c)
	if err != nil {
		t.Fatalf("deleteFilterFromRequest: %v", err)
	}
	if got == nil || got.UserID != "u1" || got.SessionID != "" {
		t.Fatalf("unexpected filter: %#v", got)
	}
}

func TestStrictRAGScopeFromRequestPrefersHeaders(t *testing.T) {
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodGet, "/rag/sources?user_id=query-user&session_id=query-session", nil)
	c.Request.Header.Set("X-User-ID", " header-user ")
	c.Request.Header.Set("X-Session-ID", " header-session ")

	got, err := strictRAGScopeFromRequest(c)
	if err != nil {
		t.Fatalf("strictRAGScopeFromRequest: %v", err)
	}
	if got.UserID != "header-user" || got.SessionID != "" {
		t.Fatalf("expected header scope, got %#v", got)
	}
}

func TestStrictRAGScopeFromRequestReadsFormScope(t *testing.T) {
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/upload", strings.NewReader("user_id=form-user&session_id=form-session"))
	c.Request.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	got, err := strictRAGScopeFromRequest(c)
	if err != nil {
		t.Fatalf("strictRAGScopeFromRequest: %v", err)
	}
	if got.UserID != "form-user" || got.SessionID != "" {
		t.Fatalf("expected form scope, got %#v", got)
	}
}

func TestDeleteFilterFromRequestRequiresScope(t *testing.T) {
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodDelete, "/rag/source/source-1", nil)

	got, err := deleteFilterFromRequest(c)
	if err == nil {
		t.Fatal("expected scope error")
	}
	if got != nil {
		t.Fatalf("expected nil filter on error, got %#v", got)
	}
}

func TestSourceListLimitFromRequestParsesPositiveLimit(t *testing.T) {
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodGet, "/rag/sources?limit=25", nil)

	got, err := sourceListLimitFromRequest(c)
	if err != nil {
		t.Fatalf("sourceListLimitFromRequest: %v", err)
	}
	if got != 25 {
		t.Fatalf("limit = %d, want 25", got)
	}
}

func TestSourceListLimitFromRequestRejectsInvalidLimit(t *testing.T) {
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodGet, "/rag/sources?limit=0", nil)

	if _, err := sourceListLimitFromRequest(c); err == nil {
		t.Fatal("expected limit error")
	}
}

func TestRAGIndexingAvailableWhenRAGIsDegradedByMissingRerank(t *testing.T) {
	caps := capability.FromRuntime(capability.RuntimeDeps{
		VectorStore:      true,
		VectorCollection: true,
		Embedding:        true,
		Rerank:           false,
	})
	h := NewExtractHandler(appconfig.Config{}, nil, rag.NewIndexer(nil, nil, "test"), caps)

	if !h.ragIndexingAvailable() {
		t.Fatalf("expected degraded RAG to allow indexing")
	}
}

func TestExtractReturnsCapabilityWhenRequiredModelServiceUnavailable(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name       string
		extractTyp string
		deps       capability.RuntimeDeps
		wantName   capability.Name
		wantError  string
	}{
		{
			name:       "text requires ASR",
			extractTyp: "text",
			deps:       capability.RuntimeDeps{ASR: false, VideoSummary: true},
			wantName:   capability.ASR,
			wantError:  "ASR service is not available",
		},
		{
			name:       "transcript requires ASR",
			extractTyp: "transcript",
			deps:       capability.RuntimeDeps{ASR: false, VideoSummary: true},
			wantName:   capability.ASR,
			wantError:  "ASR service is not available",
		},
		{
			name:       "summary requires video summary",
			extractTyp: "summary",
			deps:       capability.RuntimeDeps{ASR: true, VideoSummary: false},
			wantName:   capability.VideoSummary,
			wantError:  "video summary service is not available",
		},
		{
			name:       "video_summary requires video summary",
			extractTyp: "video_summary",
			deps:       capability.RuntimeDeps{ASR: true, VideoSummary: false},
			wantName:   capability.VideoSummary,
			wantError:  "video summary service is not available",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewExtractHandler(appconfig.Config{}, nil, nil, capability.FromRuntime(tt.deps))

			router := gin.New()
			router.POST("/extract", h.Extract)

			body := bytes.NewBufferString(`{"url":"https://example.com/video","name":"demo","type":"` + tt.extractTyp + `"}`)
			req := httptest.NewRequest(http.MethodPost, "/extract", body)
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()

			router.ServeHTTP(resp, req)

			if resp.Code != http.StatusServiceUnavailable {
				t.Fatalf("expected status 503, got %d: %s", resp.Code, resp.Body.String())
			}

			var out map[string]any
			if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if out["error"] != tt.wantError {
				t.Fatalf("expected error %q, got %#v", tt.wantError, out["error"])
			}
			capabilityPayload, ok := out["capability"].(map[string]any)
			if !ok {
				t.Fatalf("expected capability details, got %#v", out)
			}
			if capabilityPayload["name"] != string(tt.wantName) {
				t.Fatalf("expected capability %q, got %#v", tt.wantName, capabilityPayload)
			}
			if capabilityPayload["status"] != string(capability.Unavailable) {
				t.Fatalf("expected unavailable capability, got %#v", capabilityPayload)
			}
		})
	}
}

func TestRequiredExtractCapabilityOnlyBlocksModelBackedTypes(t *testing.T) {
	tests := []struct {
		extractTyp string
		wantName   capability.Name
		want       bool
	}{
		{extractTyp: "text", wantName: capability.ASR, want: true},
		{extractTyp: "transcript", wantName: capability.ASR, want: true},
		{extractTyp: "summary", wantName: capability.VideoSummary, want: true},
		{extractTyp: "video_summary", wantName: capability.VideoSummary, want: true},
		{extractTyp: "audio", want: false},
		{extractTyp: "video", want: false},
		{extractTyp: "unknown", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.extractTyp, func(t *testing.T) {
			gotName, got := requiredExtractCapability(tt.extractTyp)
			if got != tt.want {
				t.Fatalf("requiredExtractCapability() required = %v, want %v", got, tt.want)
			}
			if gotName != tt.wantName {
				t.Fatalf("requiredExtractCapability() name = %q, want %q", gotName, tt.wantName)
			}
		})
	}
}

func TestUnavailableExtractCapabilityDoesNotBlockAudioVideoOrUnknownTypes(t *testing.T) {
	h := NewExtractHandler(appconfig.Config{}, nil, nil, capability.FromRuntime(capability.RuntimeDeps{}))

	for _, extractTyp := range []string{"audio", "video", "unknown"} {
		t.Run(extractTyp, func(t *testing.T) {
			if cap, blocked := h.unavailableExtractCapability(extractTyp); blocked {
				t.Fatalf("did not expect %q to be blocked by model capability, got %#v", extractTyp, cap)
			}
		})
	}
}
