package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/raiki02/vidwise/internal/chat"
	"github.com/raiki02/vidwise/internal/knowledgeagent"
)

func TestAgentTurnCreatesSessionAndReturnsProcessVideoAction(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := newAgentFakeSessionStore()
	service := knowledgeagent.NewService(knowledgeagent.ServiceConfig{
		Sessions: store,
		Actions:  knowledgeagent.NewActionStore(knowledgeagent.ActionStoreOptions{}),
	})
	h := NewKnowledgeAgentHandler(service)

	router := gin.New()
	router.Use(testTraceIDMiddleware("trace-agent-1"))
	router.POST("/agent/turn", h.Turn)

	body := bytes.NewBufferString(`{"user_id":"u1","message":"【演示视频】 https://www.bilibili.com/video/BV1xx411c7mD/"}`)
	req := httptest.NewRequest(http.MethodPost, "/agent/turn", body)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}
	var out knowledgeagent.TurnResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out.TraceID != "trace-agent-1" {
		t.Fatalf("trace id = %q", out.TraceID)
	}
	if out.SessionID != "session-1" {
		t.Fatalf("session id = %q, want session-1", out.SessionID)
	}
	if len(out.PendingActions) != 1 || out.PendingActions[0].Type != knowledgeagent.ActionProcessVideo {
		t.Fatalf("pending actions = %#v, want process_video", out.PendingActions)
	}
	if len(store.messages["session-1"]) != 2 {
		t.Fatalf("stored messages = %#v, want user and assistant", store.messages["session-1"])
	}
}

func TestAgentTurnReusesExistingSession(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := newAgentFakeSessionStore()
	store.messages["session-1"] = []chat.Message{{SessionID: "session-1", Role: "user", Content: "之前的问题"}}
	service := knowledgeagent.NewService(knowledgeagent.ServiceConfig{
		Sessions: store,
		Actions:  knowledgeagent.NewActionStore(knowledgeagent.ActionStoreOptions{}),
	})
	h := NewKnowledgeAgentHandler(service)

	router := gin.New()
	router.POST("/agent/turn", h.Turn)

	body := bytes.NewBufferString(`{"user_id":"u1","session_id":"session-1","message":"这个呢？"}`)
	req := httptest.NewRequest(http.MethodPost, "/agent/turn", body)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}
	var out knowledgeagent.TurnResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out.SessionID != "session-1" {
		t.Fatalf("session id = %q, want session-1", out.SessionID)
	}
	if store.createCalls != 0 {
		t.Fatalf("CreateSessionForUser calls = %d, want 0", store.createCalls)
	}
	if len(store.messages["session-1"]) != 3 {
		t.Fatalf("stored messages = %#v, want prior, user, assistant", store.messages["session-1"])
	}
}

func TestAgentConfirmStartsVideoAction(t *testing.T) {
	gin.SetMode(gin.TestMode)
	videos := &agentFakeVideoProcessor{result: knowledgeagent.VideoProcessResult{TaskID: "task-1", Status: "pending", SessionID: "s1"}}
	service := knowledgeagent.NewService(knowledgeagent.ServiceConfig{
		Videos:  videos,
		Actions: knowledgeagent.NewActionStore(knowledgeagent.ActionStoreOptions{}),
	})
	turn, err := service.Turn(context.Background(), knowledgeagent.TurnRequest{UserID: "u1", SessionID: "s1", Message: "https://example.com/video"})
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}
	h := NewKnowledgeAgentHandler(service)

	router := gin.New()
	router.Use(testTraceIDMiddleware("trace-confirm-1"))
	router.POST("/agent/actions/:id/confirm", h.ConfirmAction)

	body := bytes.NewBufferString(`{"user_id":"u1"}`)
	req := httptest.NewRequest(http.MethodPost, "/agent/actions/"+turn.PendingActions[0].ID+"/confirm", body)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}
	var out knowledgeagent.TurnResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(out.TaskIDs) != 1 || out.TaskIDs[0] != "task-1" {
		t.Fatalf("task ids = %#v, want task-1", out.TaskIDs)
	}
	if videos.calls != 1 {
		t.Fatalf("video calls = %d, want 1", videos.calls)
	}
	if videos.traceID != "trace-confirm-1" {
		t.Fatalf("trace id passed to video = %q", videos.traceID)
	}
}

func TestAgentConfirmIndexesTaskTranscript(t *testing.T) {
	gin.SetMode(gin.TestMode)
	indexer := &agentFakeTranscriptIndexer{result: knowledgeagent.TranscriptIndexResult{
		Status:      "indexed",
		TaskID:      "task-1",
		ChunkCount:  2,
		ContentType: "text/plain",
		SourceIDs:   []string{"source-1"},
	}}
	service := knowledgeagent.NewService(knowledgeagent.ServiceConfig{
		Indexer: indexer,
		Actions: knowledgeagent.NewActionStore(knowledgeagent.ActionStoreOptions{}),
	})
	turn, err := service.Turn(context.Background(), knowledgeagent.TurnRequest{UserID: "u1", SessionID: "s1", Message: "把 task-1 存入知识库"})
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}
	h := NewKnowledgeAgentHandler(service)

	router := gin.New()
	router.POST("/agent/actions/:id/confirm", h.ConfirmAction)

	body := bytes.NewBufferString(`{"user_id":"u1"}`)
	req := httptest.NewRequest(http.MethodPost, "/agent/actions/"+turn.PendingActions[0].ID+"/confirm", body)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}
	var out knowledgeagent.TurnResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(out.ExecutedActions) != 1 || out.ExecutedActions[0].Type != knowledgeagent.ActionIndexTaskTranscript {
		t.Fatalf("executed actions = %#v, want index_task_transcript", out.ExecutedActions)
	}
	if indexer.taskID != "task-1" {
		t.Fatalf("indexed task id = %q", indexer.taskID)
	}
}

type agentFakeSessionStore struct {
	messages    map[string][]chat.Message
	createCalls int
}

func newAgentFakeSessionStore() *agentFakeSessionStore {
	return &agentFakeSessionStore{messages: map[string][]chat.Message{}}
}

func (s *agentFakeSessionStore) CreateSessionForUser(_ context.Context, userID, title string) (*chat.Session, error) {
	s.createCalls++
	return &chat.Session{ID: "session-1", UserID: userID, Title: title}, nil
}

func (s *agentFakeSessionStore) AddMessage(_ context.Context, sessionID, role, content string) (*chat.Message, error) {
	msg := chat.Message{SessionID: sessionID, Role: role, Content: content}
	s.messages[sessionID] = append(s.messages[sessionID], msg)
	return &msg, nil
}

func (s *agentFakeSessionStore) GetRecentMessages(_ context.Context, sessionID string, limit int) ([]chat.Message, error) {
	return s.messages[sessionID], nil
}

type agentFakeVideoProcessor struct {
	req     knowledgeagent.VideoProcessRequest
	result  knowledgeagent.VideoProcessResult
	traceID string
	calls   int
}

func (p *agentFakeVideoProcessor) StartVideoProcess(_ context.Context, req knowledgeagent.VideoProcessRequest, traceID string) (knowledgeagent.VideoProcessResult, error) {
	p.calls++
	p.req = req
	p.traceID = traceID
	if p.result.TraceID == "" {
		p.result.TraceID = traceID
	}
	return p.result, nil
}

type agentFakeTranscriptIndexer struct {
	taskID string
	result knowledgeagent.TranscriptIndexResult
}

func (i *agentFakeTranscriptIndexer) IndexTranscriptTask(_ context.Context, taskID string) (knowledgeagent.TranscriptIndexResult, error) {
	i.taskID = taskID
	return i.result, nil
}
