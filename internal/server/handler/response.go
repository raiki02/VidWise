package handler

import "github.com/gin-gonic/gin"

func errorJSON(c *gin.Context, status int, message string) {
	errorJSONWithFields(c, status, message, nil)
}

func errorJSONWithFields(c *gin.Context, status int, message string, fields gin.H) {
	payload := gin.H{"error": message}
	for key, value := range fields {
		payload[key] = value
	}
	if traceID := c.GetString("trace_id"); traceID != "" {
		payload["trace_id"] = traceID
	}
	c.JSON(status, payload)
}
