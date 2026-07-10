package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/capability"
	"github.com/raiki02/vidwise/internal/chatagent"
	"github.com/raiki02/vidwise/internal/rag"
)

type handlerFakeRetriever struct {
	chunks []rag.RelevantChunk
	req    rag.RetrieveRequest
}

func (r *handlerFakeRetriever) RetrieveWithOptions(_ context.Context, req rag.RetrieveRequest) ([]rag.RelevantChunk, error) {
	r.req = req
	return r.chunks, nil
}

func TestChatQueryWithoutSessionStoreFallsBackToStatelessAnswer(t *testing.T) {
	gin.SetMode(gin.TestMode)
	llmCfg := testDisabledLLMConfig()
	h := NewChatHandler(nil, nil, nil, llmCfg, testCapabilities(llmCfg))

	router := gin.New()
	router.POST("/chat/query", h.ChatQuery)

	body := bytes.NewBufferString(`{"query":"你好"}`)
	req := httptest.NewRequest(http.MethodPost, "/chat/query", body)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var out ChatQueryResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out.Question != "你好" {
		t.Fatalf("question not echoed: %#v", out)
	}
	if out.Answer == "" {
		t.Fatalf("expected fallback answer, got %#v", out)
	}
	if out.SessionID != "" {
		t.Fatalf("expected no persisted session id without session store, got %#v", out)
	}
}

func TestChatQueryReportsRetrievalStatusWhenRetrieverUnavailable(t *testing.T) {
	gin.SetMode(gin.TestMode)
	llmCfg := testDisabledLLMConfig()
	h := NewChatHandler(nil, nil, nil, llmCfg, testCapabilities(llmCfg))

	router := gin.New()
	router.Use(testTraceIDMiddleware("trace-chat-1"))
	router.POST("/chat/query", h.ChatQuery)

	body := bytes.NewBufferString(`{"query":"视频里讲了什么？"}`)
	req := httptest.NewRequest(http.MethodPost, "/chat/query", body)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var out ChatQueryResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if !out.RAGTriggered {
		t.Fatalf("expected RAG to be required, got %#v", out)
	}
	if out.TraceID != "trace-chat-1" {
		t.Fatalf("trace id = %q, want trace-chat-1", out.TraceID)
	}
	if out.RAGStatus != "unavailable" {
		t.Fatalf("rag status = %q, want unavailable", out.RAGStatus)
	}
	if out.RAGQuery != "视频里讲了什么？" {
		t.Fatalf("rag query = %q, want original query when retriever is unavailable", out.RAGQuery)
	}
	if len(out.RAGQueries) != 1 || out.RAGQueries[0] != "视频里讲了什么？" {
		t.Fatalf("rag queries = %#v, want original query", out.RAGQueries)
	}
	if out.RAGChunkCount != 0 {
		t.Fatalf("rag chunk count = %d, want 0", out.RAGChunkCount)
	}
	if !strings.Contains(out.Answer, "没有在当前知识库范围内检索到足够相关的内容") {
		t.Fatalf("expected insufficient context answer, got %q", out.Answer)
	}
}

func TestChatQueryReportsPackedRAGContextOutcome(t *testing.T) {
	gin.SetMode(gin.TestMode)
	llmCfg := testDisabledLLMConfig()
	retriever := &handlerFakeRetriever{
		chunks: []rag.RelevantChunk{
			{Text: strings.Repeat("界", 200), Score: 0.93, SourceName: "long.md"},
			{Text: "second chunk should not reach prompt", Score: 0.8, SourceName: "later.md"},
		},
	}
	h := NewChatHandler(nil, nil, nil, llmCfg, testCapabilities(llmCfg))
	h.answerAgent = chatagent.NewWithRetriever(llmCfg, rag.ContextConfig{MaxRunes: 90}, retriever)

	router := gin.New()
	router.POST("/chat/query", h.ChatQuery)

	body := bytes.NewBufferString(`{"query":"查一下知识库里的内容","user_id":"u1"}`)
	req := httptest.NewRequest(http.MethodPost, "/chat/query", body)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var out ChatQueryResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out.RAGChunkCount != 2 {
		t.Fatalf("rag chunk count = %d, want 2", out.RAGChunkCount)
	}
	if out.RAGQuery != "查一下知识库里的内容" {
		t.Fatalf("rag query = %q, want request query", out.RAGQuery)
	}
	if len(out.RAGQueries) != 1 || out.RAGQueries[0] != "查一下知识库里的内容" {
		t.Fatalf("rag queries = %#v, want request query", out.RAGQueries)
	}
	if len(out.Chunks) != 2 {
		t.Fatalf("response chunks = %#v, want all retrieved chunks", out.Chunks)
	}
	if out.RAGContextUsedChunks != 1 {
		t.Fatalf("rag context used chunks = %d, want 1", out.RAGContextUsedChunks)
	}
	if !out.RAGContextTruncated {
		t.Fatalf("expected truncated RAG context, got %#v", out)
	}
	if len(out.RAGContextChunks) != 1 {
		t.Fatalf("rag context chunks = %#v, want only chunks that reached prompt", out.RAGContextChunks)
	}
	if out.RAGContextChunks[0].SnippetNumber != 1 || out.RAGContextChunks[0].SourceName != "long.md" {
		t.Fatalf("unexpected prompt citation chunk: %#v", out.RAGContextChunks[0])
	}
	if strings.Contains(out.Answer, "second chunk should not reach prompt") {
		t.Fatalf("expected omitted chunk not to appear in answer: %q", out.Answer)
	}
}

func TestSessionEndpointsReturnUnavailableWithoutSessionStore(t *testing.T) {
	gin.SetMode(gin.TestMode)
	llmCfg := testDisabledLLMConfig()
	h := NewChatHandler(nil, nil, nil, llmCfg, testCapabilities(llmCfg))

	router := gin.New()
	router.POST("/chat/new", h.NewSession)
	router.GET("/chat/sessions", h.ListSessions)
	router.GET("/chat/session/:id", h.GetSession)

	tests := []struct {
		name   string
		method string
		path   string
		body   string
	}{
		{name: "new", method: http.MethodPost, path: "/chat/new", body: `{}`},
		{name: "list", method: http.MethodGet, path: "/chat/sessions"},
		{name: "get", method: http.MethodGet, path: "/chat/session/demo"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, tt.path, bytes.NewBufferString(tt.body))
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()

			router.ServeHTTP(resp, req)

			if resp.Code != http.StatusServiceUnavailable {
				t.Fatalf("expected status 503, got %d: %s", resp.Code, resp.Body.String())
			}
		})
	}
}

func TestRAGHealthIncludesChatSessionStoreStatus(t *testing.T) {
	gin.SetMode(gin.TestMode)
	llmCfg := testDisabledLLMConfig()
	h := NewChatHandler(nil, nil, nil, llmCfg, testCapabilities(llmCfg))

	router := gin.New()
	router.GET("/rag/health", h.RAGHealth)

	req := httptest.NewRequest(http.MethodGet, "/rag/health", nil)
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out["chat"] != "unavailable" {
		t.Fatalf("expected chat status unavailable, got %#v", out)
	}
	if _, ok := out["capabilities"].(map[string]any); !ok {
		t.Fatalf("expected capability map in health response, got %#v", out)
	}
}

func TestUserFactsReturnsUnavailableWithoutMemoryStore(t *testing.T) {
	gin.SetMode(gin.TestMode)
	llmCfg := testDisabledLLMConfig()
	h := NewChatHandler(nil, nil, nil, llmCfg, testCapabilities(llmCfg))

	router := gin.New()
	router.GET("/user/facts", h.GetUserFacts)

	req := httptest.NewRequest(http.MethodGet, "/user/facts?user_id=u1", nil)
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status 503, got %d: %s", resp.Code, resp.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if _, ok := out["capability"].(map[string]any); !ok {
		t.Fatalf("expected capability details, got %#v", out)
	}
}

func TestChatQueryRejectsWhitespaceQuery(t *testing.T) {
	gin.SetMode(gin.TestMode)
	llmCfg := testDisabledLLMConfig()
	h := NewChatHandler(nil, nil, nil, llmCfg, testCapabilities(llmCfg))

	router := gin.New()
	router.Use(testTraceIDMiddleware("trace-chat-1"))
	router.POST("/chat/query", h.ChatQuery)

	body := bytes.NewBufferString(`{"query":"   "}`)
	req := httptest.NewRequest(http.MethodPost, "/chat/query", body)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d: %s", resp.Code, resp.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out["trace_id"] != "trace-chat-1" {
		t.Fatalf("expected trace_id in error response, got %#v", out)
	}
}

func testTraceIDMiddleware(traceID string) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Set("trace_id", traceID)
		c.Next()
	}
}

func TestChatChunkFromRelevantPreservesTraceIdentity(t *testing.T) {
	got := chatChunkFromRelevant(rag.RelevantChunk{
		Text:          "chunk text",
		Score:         0.91,
		SourceID:      "source-1",
		DocumentID:    "doc-1",
		ChunkID:       "chunk-1",
		ContentHash:   "hash-1",
		TaskID:        "task-1",
		SessionID:     "session-1",
		ChunkIdx:      7,
		SourceName:    "guide.md",
		SourceURL:     "https://example.com/guide",
		ContentType:   "markdown",
		DocumentTitle: "Guide",
		HeadingPath:   "Guide > Install",
	})

	if got.SourceID != "source-1" || got.DocumentID != "doc-1" || got.ChunkID != "chunk-1" || got.ContentHash != "hash-1" {
		t.Fatalf("identity metadata missing: %#v", got)
	}
	if got.TaskID != "task-1" || got.SessionID != "session-1" || got.ChunkIdx != 7 {
		t.Fatalf("scope metadata missing: %#v", got)
	}
	if got.SourceName != "guide.md" || got.SourceURL != "https://example.com/guide" || got.HeadingPath != "Guide > Install" {
		t.Fatalf("citation metadata missing: %#v", got)
	}
}

func TestChatChunkFromContextCitationAddsSnippetNumber(t *testing.T) {
	got := chatChunkFromContextCitation(rag.ContextCitation{
		SnippetNumber: 3,
		Chunk: rag.RelevantChunk{
			Text:       "chunk text",
			Score:      0.91,
			SourceName: "guide.md",
		},
	})

	if got.SnippetNumber != 3 {
		t.Fatalf("snippet number = %d, want 3", got.SnippetNumber)
	}
	if got.Text != "chunk text" || got.SourceName != "guide.md" {
		t.Fatalf("chunk metadata missing: %#v", got)
	}
}

func testDisabledLLMConfig() appconfig.LLMConfig {
	enabled := false
	return appconfig.LLMConfig{Enabled: &enabled}
}

func testCapabilities(llmCfg appconfig.LLMConfig) capability.Snapshot {
	return capability.FromRuntime(capability.RuntimeDeps{
		ChatSessionStore: false,
		MemoryStore:      false,
		VectorStore:      false,
		Embedding:        false,
		Rerank:           false,
		LLMConfig:        llmCfg,
	})
}
