package server

import (
	"github.com/gin-gonic/gin"
	"github.com/raiki02/vidwise/internal/appconfig"
)

// New creates a Gin engine with all routes registered.
// This keeps backward compatibility — optional services are nil.
func New(cfg appconfig.Config) *gin.Engine {
	return Router(cfg, nil, nil, nil, nil, nil)
}
