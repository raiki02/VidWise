package server

import (
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/raiki02/vidwise/internal/appconfig"
)

func TestNewHTTPServerUsesConfiguredTimeouts(t *testing.T) {
	handler := http.NewServeMux()
	srv, err := NewHTTPServer(appconfig.ServerConfig{
		Addr:              ":9090",
		ReadHeaderTimeout: "2s",
		ReadTimeout:       "30s",
		WriteTimeout:      "45s",
		IdleTimeout:       "90s",
	}, handler)
	if err != nil {
		t.Fatalf("NewHTTPServer returned error: %v", err)
	}

	if srv.Addr != ":9090" {
		t.Fatalf("Addr = %q, want :9090", srv.Addr)
	}
	if srv.Handler != handler {
		t.Fatal("expected configured handler")
	}
	if srv.ReadHeaderTimeout != 2*time.Second {
		t.Fatalf("ReadHeaderTimeout = %s, want 2s", srv.ReadHeaderTimeout)
	}
	if srv.ReadTimeout != 30*time.Second {
		t.Fatalf("ReadTimeout = %s, want 30s", srv.ReadTimeout)
	}
	if srv.WriteTimeout != 45*time.Second {
		t.Fatalf("WriteTimeout = %s, want 45s", srv.WriteTimeout)
	}
	if srv.IdleTimeout != 90*time.Second {
		t.Fatalf("IdleTimeout = %s, want 90s", srv.IdleTimeout)
	}
}

func TestNewHTTPServerUsesDefaults(t *testing.T) {
	srv, err := NewHTTPServer(appconfig.ServerConfig{}, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	if err != nil {
		t.Fatalf("NewHTTPServer returned error: %v", err)
	}

	if srv.Addr != ":8080" {
		t.Fatalf("Addr = %q, want :8080", srv.Addr)
	}
	if srv.ReadHeaderTimeout != 5*time.Second {
		t.Fatalf("ReadHeaderTimeout = %s, want 5s", srv.ReadHeaderTimeout)
	}
	if srv.ReadTimeout != 30*time.Second {
		t.Fatalf("ReadTimeout = %s, want 30s", srv.ReadTimeout)
	}
	if srv.WriteTimeout != 0 {
		t.Fatalf("WriteTimeout = %s, want 0s", srv.WriteTimeout)
	}
	if srv.IdleTimeout != 120*time.Second {
		t.Fatalf("IdleTimeout = %s, want 120s", srv.IdleTimeout)
	}
}

func TestNewHTTPServerRejectsInvalidConfig(t *testing.T) {
	_, err := NewHTTPServer(appconfig.ServerConfig{ReadHeaderTimeout: "0s"}, http.NewServeMux())
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "read_header_timeout") {
		t.Fatalf("expected read_header_timeout error, got %v", err)
	}
}

func TestNewHTTPServerRejectsNilHandler(t *testing.T) {
	_, err := NewHTTPServer(appconfig.ServerConfig{}, nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "handler") {
		t.Fatalf("expected handler error, got %v", err)
	}
}
