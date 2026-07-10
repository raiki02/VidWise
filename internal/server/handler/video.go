package handler

import (
	"context"
	"log/slog"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/raiki02/vidwise/internal/agent"
	"github.com/raiki02/vidwise/internal/background"
	"github.com/raiki02/vidwise/internal/tool"
)

const videoProcessTimeout = 2 * time.Hour

type videoProcessor func(ctx context.Context, registry *tool.Registry, url, workDir, name, userID, sessionID, taskID, language string) (string, error)

type VideoHandler struct {
	registry *tool.Registry
	runner   *background.Runner
	process  videoProcessor
}

func NewVideoHandler(registry *tool.Registry) *VideoHandler {
	return NewVideoHandlerWithBackground(registry, nil)
}

func NewVideoHandlerWithBackground(registry *tool.Registry, runner *background.Runner) *VideoHandler {
	if runner == nil {
		runner = background.NewRunner(videoProcessTimeout)
	}
	return &VideoHandler{registry: registry, runner: runner, process: agent.ExecuteVideoProcess}
}

type VideoProcessRequest struct {
	URL       string `json:"url" binding:"required"`
	Name      string `json:"name" binding:"required"`
	UserID    string `json:"user_id" binding:"required"`
	SessionID string `json:"session_id"`
	WorkDir   string `json:"work_dir"`
	Language  string `json:"language"`
}

type VideoProcessResponse struct {
	TaskID    string `json:"task_id"`
	TraceID   string `json:"trace_id"`
	Status    string `json:"status"`
	SessionID string `json:"session_id"`
}

// VideoProcess handles POST /video/process — async video processing.
func (h *VideoHandler) VideoProcess(c *gin.Context) {
	var req VideoProcessRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		errorJSON(c, http.StatusBadRequest, "url, name, and user_id are required")
		return
	}

	taskID := uuid.New().String()
	traceID := requestTraceID(c)

	if req.SessionID == "" {
		req.SessionID = uuid.New().String()
	}
	if req.WorkDir == "" {
		req.WorkDir = "/tmp/video-extractor-" + taskID
	}
	if req.Language == "" {
		req.Language = "zh"
	}

	process := h.process
	if process == nil {
		process = agent.ExecuteVideoProcess
	}
	runner := h.runner
	if runner == nil {
		runner = background.NewRunner(videoProcessTimeout)
	}
	runner.Go("video.process", func(ctx context.Context) {
		_, err := process(
			ctx,
			h.registry,
			req.URL,
			req.WorkDir,
			req.Name,
			req.UserID,
			req.SessionID,
			taskID,
			req.Language,
		)
		if err != nil {
			slog.Error("video.process_failed", "trace_id", traceID, "task_id", taskID, "session_id", req.SessionID, "err", err)
		}
	})

	c.JSON(http.StatusAccepted, VideoProcessResponse{
		TaskID:    taskID,
		TraceID:   traceID,
		Status:    "pending",
		SessionID: req.SessionID,
	})
}
