package server

import (
	"github.com/gin-gonic/gin"
	"github.com/raiki02/video-extractor/internal/appconfig"
)

// New creates a Gin engine with all routes registered.
func New(cfg appconfig.Config) *gin.Engine {
	return Router(cfg, nil)
}

// NewWithRegistry creates a Gin engine with a tool registry for new endpoints.
func NewWithRegistry(cfg appconfig.Config, registry interface{}) *gin.Engine {
	// For backward compatibility, accept nil registry
	_ = registry
	return Router(cfg, nil)
}
