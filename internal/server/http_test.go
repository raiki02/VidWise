package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/cloudwego/eino/schema"
	"github.com/raiki02/vidwise/internal/chat"
)

func TestHealth(t *testing.T) {
	r := Router(newTestService(t))
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d", w.Code)
	}
}

func TestAgentChatRejectsBadBody(t *testing.T) {
	r := Router(newTestService(t))
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/agent/chat", bytes.NewBufferString("{bad json"))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
}

func TestAgentChatRejectsEmptyMessage(t *testing.T) {
	r := Router(newTestService(t))
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/agent/chat", bytes.NewBufferString(`{"message":"  "}`))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
}

func TestAgentChatUnknownSession(t *testing.T) {
	r := Router(newTestService(t))
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/agent/chat", bytes.NewBufferString(`{"session_id":"nope","message":"hi"}`))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	if w.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", w.Code)
	}
}

func TestAgentChatOK(t *testing.T) {
	r := Router(newTestService(t))
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/agent/chat", bytes.NewBufferString(`{"message":"这个视频讲了什么？"}`))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body=%s", w.Code, w.Body.String())
	}
	var resp agentChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.SessionID == "" || resp.Reply == "" {
		t.Fatalf("unexpected response: %+v", resp)
	}
}

func TestIndexServed(t *testing.T) {
	r := Router(newTestService(t))
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK || !bytes.Contains(w.Body.Bytes(), []byte("VidWise")) {
		t.Fatalf("index not served: %d %s", w.Code, w.Body.String()[:80])
	}
}

func newTestService(t *testing.T) *chat.Service {
	t.Helper()
	return chat.NewService(chat.NewMemoryStore(), &staticGenerator{reply: "模拟回答"}, 20)
}

type staticGenerator struct {
	reply string
}

func (g *staticGenerator) Generate(_ context.Context, _ []*schema.Message) (*schema.Message, error) {
	return &schema.Message{Role: schema.Assistant, Content: g.reply}, nil
}
