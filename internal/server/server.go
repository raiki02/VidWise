package server

import (
	"fmt"
	"net/http"
	"time"

	"github.com/raiki02/vidwise/internal/appconfig"
)

// NewHTTPServer wraps the gin engine with the configured timeouts. WriteTimeout
// defaults to 0 so long-running download/transcribe turns are not cut off.
func NewHTTPServer(cfg appconfig.ServerConfig, handler http.Handler) (*http.Server, error) {
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
	return &http.Server{
		Addr:              cfg.Addr,
		Handler:           handler,
		ReadHeaderTimeout: readHeaderTimeout,
		ReadTimeout:       readTimeout,
		WriteTimeout:      writeTimeout,
		IdleTimeout:       idleTimeout,
	}, nil
}

// ShutdownTimeout is the graceful shutdown budget.
func ShutdownTimeout(cfg appconfig.ServerConfig) time.Duration {
	d, err := cfg.ShutdownTimeoutDuration()
	if err != nil {
		return 10 * time.Second
	}
	return d
}
