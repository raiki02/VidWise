package handler

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/raiki02/vidwise/internal/agent"
	"github.com/raiki02/vidwise/internal/background"
	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
	taskpkg "github.com/raiki02/vidwise/internal/task"
	"github.com/raiki02/vidwise/internal/tool"
)

const videoProcessTimeout = 2 * time.Hour

type videoProcessor func(ctx context.Context, registry *tool.Registry, url, workDir, name, userID, sessionID, taskID, language string) (string, error)
type videoProcessorWithObserver func(ctx context.Context, registry *tool.Registry, url, workDir, name, userID, sessionID, taskID, language string, observer agent.VideoProcessObserver) (string, error)

type VideoHandler struct {
	registry            *tool.Registry
	runner              *background.Runner
	process             videoProcessor
	processWithObserver videoProcessorWithObserver
	tasks               *taskpkg.Tracker
}

func NewVideoHandler(registry *tool.Registry) *VideoHandler {
	return NewVideoHandlerWithBackground(registry, nil)
}

func NewVideoHandlerWithBackground(registry *tool.Registry, runner *background.Runner) *VideoHandler {
	return NewVideoHandlerWithBackgroundAndTasks(registry, runner, taskpkg.NewTracker())
}

func NewVideoHandlerWithBackgroundAndTasks(registry *tool.Registry, runner *background.Runner, tasks *taskpkg.Tracker) *VideoHandler {
	if runner == nil {
		runner = background.NewRunner(videoProcessTimeout)
	}
	if tasks == nil {
		tasks = taskpkg.NewTracker()
	}
	return &VideoHandler{
		registry:            registry,
		runner:              runner,
		process:             agent.ExecuteVideoProcess,
		processWithObserver: agent.ExecuteVideoProcessWithObserver,
		tasks:               tasks,
	}
}

type VideoProcessRequest struct {
	URL       string `json:"url" binding:"required"`
	Name      string `json:"name"`
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

type VideoTaskIndexResponse struct {
	Status      string   `json:"status"`
	TaskID      string   `json:"task_id"`
	ChunkCount  int      `json:"chunk_count"`
	ContentType string   `json:"content_type"`
	SourceIDs   []string `json:"source_ids,omitempty"`
}

// VideoProcess handles POST /video/process — async video processing.
func (h *VideoHandler) VideoProcess(c *gin.Context) {
	var req VideoProcessRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		errorJSON(c, http.StatusBadRequest, "url and user_id are required")
		return
	}

	out, err := h.StartVideoProcess(c.Request.Context(), req, requestTraceID(c))
	if err != nil {
		errorJSON(c, statusFromError(err, http.StatusInternalServerError), err.Error())
		return
	}
	c.JSON(http.StatusAccepted, out)
}

func (h *VideoHandler) StartVideoProcess(_ context.Context, req VideoProcessRequest, traceID string) (VideoProcessResponse, error) {
	normalized := normalizeVideoShareInput(req.URL, req.Name)
	req.URL = normalized.URL
	req.Name = normalized.Name
	if req.URL == "" {
		return VideoProcessResponse{}, newResponseError(http.StatusBadRequest, "url is required")
	}
	if req.Name == "" {
		return VideoProcessResponse{}, newResponseError(http.StatusBadRequest, "name is required when the URL field does not include a share title")
	}

	taskID := uuid.New().String()

	if req.SessionID == "" {
		req.SessionID = uuid.New().String()
	}
	if req.WorkDir == "" {
		req.WorkDir = "/tmp/video-extractor-" + taskID
	}
	if req.Language == "" {
		req.Language = "zh"
	}
	tasks := h.tasks
	if tasks == nil {
		tasks = taskpkg.NewTracker()
		h.tasks = tasks
	}
	tasks.Create(taskpkg.TrackCreateRequest{
		ID:        taskID,
		Type:      "video_process",
		UserID:    req.UserID,
		SessionID: req.SessionID,
		TraceID:   traceID,
		Steps:     agent.VideoProcessStepNames(),
	})

	processWithObserver := h.processWithObserver
	process := h.process
	runner := h.runner
	if runner == nil {
		runner = background.NewRunner(videoProcessTimeout)
	}
	runner.Go("video.process", func(ctx context.Context) {
		tasks.Start(taskID)
		observer := taskStepObserver{tasks: tasks, taskID: taskID}
		var result string
		var err error
		if processWithObserver != nil {
			result, err = processWithObserver(
				ctx,
				h.registry,
				req.URL,
				req.WorkDir,
				req.Name,
				req.UserID,
				req.SessionID,
				taskID,
				req.Language,
				observer,
			)
		} else {
			if process == nil {
				process = agent.ExecuteVideoProcess
			}
			result, err = process(
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
		}
		if err != nil {
			tasks.Fail(taskID, err.Error())
			slog.Error("video.process_failed", "trace_id", traceID, "task_id", taskID, "session_id", req.SessionID, "err", err)
			return
		}
		tasks.Complete(taskID, completeVideoProcessOutput(tasks, taskID, result, req.Name, req.URL))
	})

	return VideoProcessResponse{
		TaskID:    taskID,
		TraceID:   traceID,
		Status:    "pending",
		SessionID: req.SessionID,
	}, nil
}

// IndexTaskTranscript handles POST /task/:id/index — user-triggered indexing
// of the transcript produced by a completed async video task.
func (h *VideoHandler) IndexTaskTranscript(c *gin.Context) {
	taskID := strings.TrimSpace(c.Param("id"))
	out, err := h.IndexTranscriptTask(c.Request.Context(), taskID)
	if err != nil {
		errorJSON(c, statusFromError(err, http.StatusInternalServerError), err.Error())
		return
	}
	c.JSON(http.StatusOK, out)
}

func (h *VideoHandler) IndexTranscriptTask(ctx context.Context, taskID string) (VideoTaskIndexResponse, error) {
	taskID = strings.TrimSpace(taskID)
	if taskID == "" {
		return VideoTaskIndexResponse{}, newResponseError(http.StatusBadRequest, "task id is required")
	}

	tasks := h.tasks
	if tasks == nil {
		return VideoTaskIndexResponse{}, newResponseError(http.StatusNotFound, "task not found")
	}
	tracked, ok := tasks.Get(taskID)
	if !ok {
		return VideoTaskIndexResponse{}, newResponseError(http.StatusNotFound, "task not found")
	}
	if tracked.Type != "video_process" {
		return VideoTaskIndexResponse{}, newResponseError(http.StatusBadRequest, "task must be a video_process task")
	}
	if tracked.Status != taskpkg.StatusDone {
		return VideoTaskIndexResponse{}, newResponseError(http.StatusConflict, "task must be done before indexing")
	}

	if taskOutputBool(tracked.Output, "knowledge_indexed") {
		return VideoTaskIndexResponse{
			Status:      "already_indexed",
			TaskID:      tracked.ID,
			ChunkCount:  taskOutputInt(tracked.Output, "knowledge_chunk_count"),
			ContentType: taskOutputString(tracked.Output, "knowledge_content_type"),
			SourceIDs:   taskOutputStrings(tracked.Output, "knowledge_source_ids"),
		}, nil
	}
	text := taskOutputString(tracked.Output, "formatted_text")
	if text == "" || !taskOutputBool(tracked.Output, "text_formatted") || !taskOutputBool(tracked.Output, "can_index_knowledge") {
		return VideoTaskIndexResponse{}, newResponseError(http.StatusConflict, "task transcript must be formatted before indexing")
	}

	if h.registry == nil {
		return VideoTaskIndexResponse{}, newResponseError(http.StatusServiceUnavailable, "RAG indexing is not available")
	}
	ragTool, err := h.registry.Get("rag_index")
	if err != nil {
		return VideoTaskIndexResponse{}, newResponseError(http.StatusServiceUnavailable, "RAG indexing is not available")
	}

	filename := taskOutputString(tracked.Output, "filename")
	if filename == "" {
		filename = tracked.ID + ".txt"
	}
	sourceURL := taskOutputString(tracked.Output, "source_url")
	metadata := map[string]string{
		qdrantclient.FieldTaskID: tracked.ID,
	}
	if sourceURL != "" {
		metadata[qdrantclient.FieldSourceURL] = sourceURL
	}

	args, _ := tool.ToJSON(tool.RAGIndexInput{
		Text:        text,
		Filename:    filename,
		ContentType: "text/plain",
		Format:      "plain",
		UserID:      tracked.UserID,
		SessionID:   tracked.SessionID,
		Metadata:    metadata,
	})
	resultJSON, err := ragTool.InvokableRun(ctx, args)
	if err != nil {
		slog.Warn("video.task_index_failed", "task_id", tracked.ID, "err", err)
		return VideoTaskIndexResponse{}, newResponseError(http.StatusBadGateway, "indexing failed: "+err.Error())
	}

	var indexed tool.RAGIndexOutput
	if err := json.Unmarshal([]byte(resultJSON), &indexed); err != nil {
		slog.Warn("video.task_index_decode_failed", "task_id", tracked.ID, "err", err)
		return VideoTaskIndexResponse{}, newResponseError(http.StatusBadGateway, "indexing returned invalid response")
	}
	tasks.PatchOutput(tracked.ID, map[string]any{
		"knowledge_indexed":      true,
		"knowledge_indexed_at":   time.Now().UTC().Format(time.RFC3339Nano),
		"knowledge_chunk_count":  indexed.ChunkCount,
		"knowledge_content_type": indexed.ContentType,
		"knowledge_source_ids":   indexed.SourceIDs,
	})

	return VideoTaskIndexResponse{
		Status:      "indexed",
		TaskID:      tracked.ID,
		ChunkCount:  indexed.ChunkCount,
		ContentType: indexed.ContentType,
		SourceIDs:   indexed.SourceIDs,
	}, nil
}

type taskStepObserver struct {
	tasks  *taskpkg.Tracker
	taskID string
}

func (o taskStepObserver) StepStarted(name string) {
	o.tasks.StartStep(o.taskID, name)
}

func (o taskStepObserver) StepDone(name string) {
	o.tasks.CompleteStep(o.taskID, name)
}

func (o taskStepObserver) StepFailed(name string, err error) {
	message := ""
	if err != nil {
		message = err.Error()
	}
	o.tasks.FailStep(o.taskID, name, message)
}

func (o taskStepObserver) StepSkipped(name, reason string) {
	o.tasks.SkipStep(o.taskID, name, reason)
}

func (o taskStepObserver) OutputUpdated(output map[string]any) {
	o.tasks.PatchOutput(o.taskID, output)
}

func completeVideoProcessOutput(tasks *taskpkg.Tracker, taskID, result, name, sourceURL string) map[string]any {
	rawText := ""
	formattedText := ""
	textFormatted := false
	if tasks != nil {
		if tracked, ok := tasks.Get(taskID); ok {
			rawText = taskOutputString(tracked.Output, "raw_text")
			formattedText = taskOutputString(tracked.Output, "formatted_text")
			textFormatted = taskOutputBool(tracked.Output, "text_formatted")
		}
	}
	if formattedText != "" && textFormatted {
		return agent.VideoProcessReadyTextOutput(formattedText, rawText, name, sourceURL, true)
	}
	if rawText == "" {
		rawText = result
	}
	return agent.VideoProcessReadyTextOutput(result, rawText, name, sourceURL, false)
}

func taskOutputString(output map[string]any, key string) string {
	value, ok := output[key]
	if !ok {
		return ""
	}
	text, ok := value.(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(text)
}

func taskOutputBool(output map[string]any, key string) bool {
	value, ok := output[key]
	if !ok {
		return false
	}
	got, ok := value.(bool)
	return ok && got
}

func taskOutputInt(output map[string]any, key string) int {
	value, ok := output[key]
	if !ok {
		return 0
	}
	switch got := value.(type) {
	case int:
		return got
	case int64:
		return int(got)
	case float64:
		return int(got)
	default:
		return 0
	}
}

func taskOutputStrings(output map[string]any, key string) []string {
	value, ok := output[key]
	if !ok {
		return nil
	}
	switch got := value.(type) {
	case []string:
		return got
	case []any:
		out := make([]string, 0, len(got))
		for _, item := range got {
			if text, ok := item.(string); ok && strings.TrimSpace(text) != "" {
				out = append(out, strings.TrimSpace(text))
			}
		}
		return out
	default:
		return nil
	}
}
