package server

import (
	"log/slog"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/raiki02/vidwise/internal/trace"
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
		c.Request = c.Request.WithContext(trace.WithID(c.Request.Context(), traceID))
		c.Next()
	}
}

// RequestLogger logs each request with method, path, status, and latency.
func RequestLogger() gin.HandlerFunc {
	return RequestLoggerWithLogger(slog.Default())
}

// RequestLoggerWithLogger logs each request with an injectable logger.
func RequestLoggerWithLogger(logger *slog.Logger) gin.HandlerFunc {
	if logger == nil {
		logger = slog.Default()
	}
	return func(c *gin.Context) {
		start := time.Now()
		c.Next()
		latency := time.Since(start)
		c.Header("X-Response-Time", latency.String())
		logger.Info("http.request",
			"trace_id", c.GetString("trace_id"),
			"method", c.Request.Method,
			"path", c.FullPath(),
			"raw_path", c.Request.URL.Path,
			"status", c.Writer.Status(),
			"latency_ms", float64(latency.Microseconds())/1000,
			"client_ip", c.ClientIP(),
			"errors", c.Errors.String(),
		)
	}
}
