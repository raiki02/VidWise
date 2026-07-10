package server

import (
	"fmt"
	"net/http"

	"github.com/raiki02/vidwise/internal/appconfig"
)

// NewHTTPServer builds the production HTTP server wrapper around the Gin engine.
func NewHTTPServer(cfg appconfig.ServerConfig, handler http.Handler) (*http.Server, error) {
	if handler == nil {
		return nil, fmt.Errorf("http handler is required")
	}

	readHeaderTimeout, err := cfg.ReadHeaderTimeoutDuration()
	if err != nil {
		return nil, fmt.Errorf("read_header_timeout: %w", err)
	}
	readTimeout, err := cfg.ReadTimeoutDuration()
	if err != nil {
		return nil, fmt.Errorf("read_timeout: %w", err)
	}
	writeTimeout, err := cfg.WriteTimeoutDuration()
	if err != nil {
		return nil, fmt.Errorf("write_timeout: %w", err)
	}
	idleTimeout, err := cfg.IdleTimeoutDuration()
	if err != nil {
		return nil, fmt.Errorf("idle_timeout: %w", err)
	}

	addr := cfg.Addr
	if addr == "" {
		addr = ":8080"
	}
	return &http.Server{
		Addr:              addr,
		Handler:           handler,
		ReadHeaderTimeout: readHeaderTimeout,
		ReadTimeout:       readTimeout,
		WriteTimeout:      writeTimeout,
		IdleTimeout:       idleTimeout,
	}, nil
}
