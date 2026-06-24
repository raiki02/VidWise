package server

import (
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

const traceIDHeader = "X-Trace-Id"

// TraceID middleware injects or propagates a trace_id into every request.
func TraceID() gin.HandlerFunc {
	return func(c *gin.Context) {
		traceID := c.GetHeader(traceIDHeader)
		if traceID == "" {
			traceID = uuid.New().String()
		}
		c.Header(traceIDHeader, traceID)
		c.Set("trace_id", traceID)
		c.Next()
	}
}

// RequestLogger logs each request with method, path, status, and latency.
func RequestLogger() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		c.Next()
		latency := time.Since(start)
		c.Header("X-Response-Time", latency.String())
	}
}
