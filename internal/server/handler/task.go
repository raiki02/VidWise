package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"
	taskpkg "github.com/raiki02/vidwise/internal/task"
)

type TaskHandler struct {
	tasks *taskpkg.Tracker
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

	tasks := h.tasks
	if tasks == nil {
		tasks = taskpkg.NewTracker()
		h.tasks = tasks
	}
	task, ok := tasks.Get(taskID)
	if !ok {
		errorJSON(c, http.StatusNotFound, "task not found")
		return
	}

	c.JSON(http.StatusOK, task)
}
