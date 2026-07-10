package handler

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	taskpkg "github.com/raiki02/vidwise/internal/task"
)

type TaskHandler struct {
	tasks *taskpkg.Tracker
}

type TaskListResponse struct {
	Tasks []taskpkg.TrackedTask `json:"tasks"`
	Count int                   `json:"count"`
}

func NewTaskHandler() *TaskHandler {
	return NewTaskHandlerWithTracker(taskpkg.NewTracker())
}

func NewTaskHandlerWithTracker(tasks *taskpkg.Tracker) *TaskHandler {
	if tasks == nil {
		tasks = taskpkg.NewTracker()
	}
	return &TaskHandler{tasks: tasks}
}

// GetTask handles GET /task/:id — returns task status and steps.
func (h *TaskHandler) GetTask(c *gin.Context) {
	taskID := c.Param("id")
	if taskID == "" {
		errorJSON(c, http.StatusBadRequest, "task id is required")
		return
	}

	task, ok := h.tracker().Get(taskID)
	if !ok {
		errorJSON(c, http.StatusNotFound, "task not found")
		return
	}

	c.JSON(http.StatusOK, task)
}

func (h *TaskHandler) ListTasks(c *gin.Context) {
	tasks := h.tracker()
	userID := taskScopeValueFromRequest(c, "user_id", "X-User-ID")
	sessionID := taskScopeValueFromRequest(c, "session_id", "X-Session-ID")
	if userID == "" && sessionID == "" {
		errorJSON(c, http.StatusBadRequest, "user_id or session_id is required")
		return
	}

	status, err := taskStatusFromRequest(c.Query("status"))
	if err != nil {
		errorJSON(c, http.StatusBadRequest, err.Error())
		return
	}
	limit, err := taskListLimitFromRequest(c)
	if err != nil {
		errorJSON(c, http.StatusBadRequest, err.Error())
		return
	}

	list := tasks.List(taskpkg.TrackListRequest{
		UserID:    userID,
		SessionID: sessionID,
		Status:    status,
		Limit:     limit,
	})
	c.JSON(http.StatusOK, TaskListResponse{
		Tasks: list,
		Count: len(list),
	})
}

func (h *TaskHandler) tracker() *taskpkg.Tracker {
	if h.tasks == nil {
		h.tasks = taskpkg.NewTracker()
	}
	return h.tasks
}

func taskScopeValueFromRequest(c *gin.Context, field, header string) string {
	for _, value := range []string{
		c.GetHeader(header),
		c.Query(field),
	} {
		if value = strings.TrimSpace(value); value != "" {
			return value
		}
	}
	return ""
}

func taskStatusFromRequest(raw string) (taskpkg.Status, error) {
	status := taskpkg.Status(strings.TrimSpace(raw))
	if status == "" {
		return "", nil
	}
	switch status {
	case taskpkg.StatusPending, taskpkg.StatusRunning, taskpkg.StatusDone, taskpkg.StatusFailed:
		return status, nil
	default:
		return "", fmt.Errorf("status must be one of pending, running, done or failed")
	}
}

func taskListLimitFromRequest(c *gin.Context) (int, error) {
	raw := strings.TrimSpace(c.Query("limit"))
	if raw == "" {
		return 0, nil
	}
	limit, err := strconv.Atoi(raw)
	if err != nil || limit <= 0 {
		return 0, fmt.Errorf("limit must be a positive integer")
	}
	return limit, nil
}
