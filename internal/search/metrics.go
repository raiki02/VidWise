package search

import (
	"context"
	"log/slog"
	"time"

	"github.com/raiki02/vidwise/internal/trace"
)

type LoggingMetrics struct {
	logger *slog.Logger
}

func NewLoggingMetrics(logger *slog.Logger) *LoggingMetrics {
	if logger == nil {
		logger = slog.Default()
	}
	return &LoggingMetrics{logger: logger}
}

func (m *LoggingMetrics) ObserveSearch(ctx context.Context, status string, elapsed time.Duration) {
	m.logger.InfoContext(ctx, "search.metric.search", logAttrsWithTrace(ctx, "status", status, "elapsed_ms", elapsed.Milliseconds())...)
}

func (m *LoggingMetrics) ObserveProvider(ctx context.Context, provider string, status string, elapsed time.Duration) {
	m.logger.InfoContext(ctx, "search.metric.provider", logAttrsWithTrace(ctx, "provider", provider, "status", status, "elapsed_ms", elapsed.Milliseconds())...)
}

func (m *LoggingMetrics) ObserveCache(ctx context.Context, hit bool) {
	m.logger.InfoContext(ctx, "search.metric.cache", logAttrsWithTrace(ctx, "hit", hit)...)
}

func logAttrsWithTrace(ctx context.Context, attrs ...any) []any {
	traceID := trace.ID(ctx)
	if traceID == "" {
		return attrs
	}
	out := make([]any, 0, len(attrs)+2)
	out = append(out, "trace_id", traceID)
	out = append(out, attrs...)
	return out
}
