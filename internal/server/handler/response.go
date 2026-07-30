package handler

import (
	"errors"

	"github.com/gin-gonic/gin"
)

type responseError struct {
	status  int
	message string
}

func (e responseError) Error() string {
	return e.message
}

func newResponseError(status int, message string) responseError {
	return responseError{status: status, message: message}
}

func statusFromError(err error, fallback int) int {
	if err == nil {
		return fallback
	}
	var got responseError
	if errors.As(err, &got) {
		return got.status
	}
	return fallback
}

func errorJSON(c *gin.Context, status int, message string) {
	errorJSONWithFields(c, status, message, nil)
}

func errorJSONWithFields(c *gin.Context, status int, message string, fields gin.H) {
	payload := gin.H{"error": message}
	for key, value := range fields {
		payload[key] = value
	}
	if traceID := requestTraceID(c); traceID != "" {
		payload["trace_id"] = traceID
	}
	c.JSON(status, payload)
}

func requestTraceID(c *gin.Context) string {
	return c.GetString("trace_id")
}
