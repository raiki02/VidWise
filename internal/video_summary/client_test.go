package video_summary

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestClientHealthChecksEndpoint(t *testing.T) {
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.Method != http.MethodGet {
			t.Fatalf("method = %q, want GET", r.Method)
		}
		if r.URL.Path != "/health" {
			t.Fatalf("path = %q, want /health", r.URL.Path)
		}
		return jsonResponse(http.StatusOK, `{"status":"ok"}`), nil
	})}

	client := &Client{
		baseURL: "http://video-summary.local",
		http:    httpClient,
	}

	if err := client.Health(context.Background()); err != nil {
		t.Fatalf("Health: %v", err)
	}
}

func TestClientHealthRejectsUnhealthyEndpoint(t *testing.T) {
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return jsonResponse(http.StatusServiceUnavailable, "not ready"), nil
	})}

	client := &Client{
		baseURL: "http://video-summary.local",
		http:    httpClient,
	}

	err := client.Health(context.Background())
	if err == nil {
		t.Fatal("expected health error")
	}
	if !strings.Contains(err.Error(), "503") {
		t.Fatalf("error = %v, want status code", err)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func jsonResponse(statusCode int, body string) *http.Response {
	return &http.Response{
		StatusCode: statusCode,
		Status:     fmt.Sprintf("%d %s", statusCode, http.StatusText(statusCode)),
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     make(http.Header),
	}
}
