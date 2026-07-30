package handler

import (
	"context"
	"errors"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/knowledgeagent"
	"github.com/raiki02/vidwise/internal/paragraph"
	taskpkg "github.com/raiki02/vidwise/internal/task"
)

type KnowledgeAgentHandler struct {
	agent *knowledgeagent.Service
}

func NewKnowledgeAgentHandler(agent *knowledgeagent.Service) *KnowledgeAgentHandler {
	return &KnowledgeAgentHandler{agent: agent}
}

func (h *KnowledgeAgentHandler) Turn(c *gin.Context) {
	if h.agent == nil {
		errorJSON(c, http.StatusServiceUnavailable, "agent is not available")
		return
	}
	var req knowledgeagent.TurnRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		errorJSON(c, http.StatusBadRequest, "user_id and message are required")
		return
	}
	req.TraceID = requestTraceID(c)
	resp, err := h.agent.Turn(c.Request.Context(), req)
	if err != nil {
		errorJSON(c, statusForKnowledgeAgentError(err), err.Error())
		return
	}
	c.JSON(http.StatusOK, resp)
}

func (h *KnowledgeAgentHandler) ConfirmAction(c *gin.Context) {
	if h.agent == nil {
		errorJSON(c, http.StatusServiceUnavailable, "agent is not available")
		return
	}
	var req knowledgeagent.ConfirmRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		errorJSON(c, http.StatusBadRequest, "user_id is required")
		return
	}
	req.TraceID = requestTraceID(c)
	resp, err := h.agent.Confirm(c.Request.Context(), c.Param("id"), req)
	if err != nil {
		errorJSON(c, statusForKnowledgeAgentError(err), err.Error())
		return
	}
	c.JSON(http.StatusOK, resp)
}

func statusForKnowledgeAgentError(err error) int {
	switch {
	case errors.Is(err, knowledgeagent.ErrActionNotFound):
		return http.StatusNotFound
	case errors.Is(err, knowledgeagent.ErrActionAlreadyExecuted):
		return http.StatusConflict
	case errors.Is(err, knowledgeagent.ErrActionUserMismatch):
		return http.StatusForbidden
	default:
		return statusFromError(err, http.StatusBadRequest)
	}
}

func KnowledgeVideoAdapter(video *VideoHandler) knowledgeagent.VideoProcessor {
	return knowledgeVideoAdapter{video: video}
}

type knowledgeVideoAdapter struct {
	video *VideoHandler
}

func (a knowledgeVideoAdapter) StartVideoProcess(ctx context.Context, req knowledgeagent.VideoProcessRequest, traceID string) (knowledgeagent.VideoProcessResult, error) {
	if a.video == nil {
		return knowledgeagent.VideoProcessResult{}, errors.New("video processing is not available")
	}
	got, err := a.video.StartVideoProcess(ctx, VideoProcessRequest{
		URL:       req.URL,
		Name:      req.Name,
		UserID:    req.UserID,
		SessionID: req.SessionID,
		Language:  req.Language,
	}, traceID)
	if err != nil {
		return knowledgeagent.VideoProcessResult{}, err
	}
	return knowledgeagent.VideoProcessResult{
		TaskID:    got.TaskID,
		TraceID:   got.TraceID,
		Status:    got.Status,
		SessionID: got.SessionID,
	}, nil
}

func KnowledgeTranscriptIndexer(video *VideoHandler) knowledgeagent.TranscriptIndexer {
	return knowledgeTranscriptIndexer{video: video}
}

type knowledgeTranscriptIndexer struct {
	video *VideoHandler
}

func (a knowledgeTranscriptIndexer) IndexTranscriptTask(ctx context.Context, taskID string) (knowledgeagent.TranscriptIndexResult, error) {
	if a.video == nil {
		return knowledgeagent.TranscriptIndexResult{}, errors.New("transcript indexing is not available")
	}
	got, err := a.video.IndexTranscriptTask(ctx, taskID)
	if err != nil {
		return knowledgeagent.TranscriptIndexResult{}, err
	}
	return knowledgeagent.TranscriptIndexResult{
		Status:      got.Status,
		TaskID:      got.TaskID,
		ChunkCount:  got.ChunkCount,
		ContentType: got.ContentType,
		SourceIDs:   got.SourceIDs,
	}, nil
}

func KnowledgeTaskReader(tasks *taskpkg.Tracker) knowledgeagent.TaskReader {
	return knowledgeTaskReader{tasks: tasks}
}

type knowledgeTaskReader struct {
	tasks *taskpkg.Tracker
}

func (r knowledgeTaskReader) Get(id string) (knowledgeagent.TaskSnapshot, bool) {
	if r.tasks == nil {
		return knowledgeagent.TaskSnapshot{}, false
	}
	task, ok := r.tasks.Get(id)
	if !ok {
		return knowledgeagent.TaskSnapshot{}, false
	}
	return knowledgeagent.TaskSnapshot{
		ID:        task.ID,
		Type:      task.Type,
		Status:    string(task.Status),
		UserID:    task.UserID,
		SessionID: task.SessionID,
		Output:    task.Output,
		UpdatedAt: task.UpdatedAt,
	}, true
}

func KnowledgeTextFormatter(cfg appconfig.LLMConfig) knowledgeagent.TextFormatter {
	return knowledgeTextFormatter{cfg: cfg}
}

type knowledgeTextFormatter struct {
	cfg appconfig.LLMConfig
}

func (f knowledgeTextFormatter) FormatText(ctx context.Context, text string) (knowledgeagent.TextFormatResult, error) {
	formatted, err := paragraph.FormatText(ctx, text, f.cfg)
	if err != nil {
		return knowledgeagent.TextFormatResult{}, err
	}
	return knowledgeagent.TextFormatResult{Text: formatted}, nil
}
