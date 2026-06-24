package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/raiki02/video-extractor/internal/agent"
	"github.com/raiki02/video-extractor/internal/tool"
)

type VideoHandler struct {
	registry *tool.Registry
}

func NewVideoHandler(registry *tool.Registry) *VideoHandler {
	return &VideoHandler{registry: registry}
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
		c.JSON(http.StatusBadRequest, gin.H{"error": "url, name, and user_id are required"})
		return
	}

	taskID := uuid.New().String()
	traceID := uuid.New().String()

	if req.SessionID == "" {
		req.SessionID = uuid.New().String()
	}
	if req.WorkDir == "" {
		req.WorkDir = "/tmp/video-extractor-" + taskID
	}
	if req.Language == "" {
		req.Language = "zh"
	}

	// Execute the pipeline synchronously for now (can be made async with worker pool)
	go func() {
		_, err := agent.ExecuteVideoProcess(
			c.Request.Context(),
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
			_ = err // In production: update task status to failed
		}
	}()

	c.JSON(http.StatusAccepted, VideoProcessResponse{
		TaskID:    taskID,
		TraceID:   traceID,
		Status:    "pending",
		SessionID: req.SessionID,
	})
}
