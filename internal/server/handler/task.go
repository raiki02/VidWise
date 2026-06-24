package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

type TaskHandler struct{}

func NewTaskHandler() *TaskHandler {
	return &TaskHandler{}
}

// GetTask handles GET /task/:id — returns task status and steps.
func (h *TaskHandler) GetTask(c *gin.Context) {
	taskID := c.Param("id")
	if taskID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "task id is required"})
		return
	}

	// In production: query MySQL task table
	// For now return a stub indicating the task system is active
	c.JSON(http.StatusOK, gin.H{
		"task_id": taskID,
		"status":  "pending",
		"message": "Task system is active. Connect MySQL for full task tracking.",
	})
}
