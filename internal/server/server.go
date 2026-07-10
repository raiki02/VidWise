package server

import (
	"context"

	"github.com/gin-gonic/gin"
	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/capability"
	"github.com/raiki02/vidwise/internal/ragruntime"
)

// New creates a Gin engine with all routes registered.
// This keeps backward compatibility — optional services are nil.
func New(cfg appconfig.Config) *gin.Engine {
	caps := capability.FromRuntime(capability.RuntimeDeps{LLMConfig: cfg.LLM})
	return Router(cfg, nil, ragruntime.Build(context.Background(), cfg, ragruntime.Deps{}).Runtime, nil, nil, caps)
}
