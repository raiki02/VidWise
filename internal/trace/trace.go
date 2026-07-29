package trace

import "context"

type contextKey struct{}

func WithID(ctx context.Context, traceID string) context.Context {
	if traceID == "" {
		return ctx
	}
	return context.WithValue(ctx, contextKey{}, traceID)
}

func ID(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	traceID, _ := ctx.Value(contextKey{}).(string)
	return traceID
}
