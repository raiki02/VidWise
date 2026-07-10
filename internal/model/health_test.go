package model

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestEmbedClientHealthChecksEndpoint(t *testing.T) {
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.Path != "/health" {
			t.Fatalf("path = %q, want /health", r.URL.Path)
		}
		return jsonResponse(http.StatusOK, `{"status":"ok"}`), nil
	})}

	client := &EmbedClient{
		baseURL: "http://embedding.local",
		http:    httpClient,
	}

	if err := client.Health(context.Background()); err != nil {
		t.Fatalf("Health: %v", err)
	}
}

func TestRerankClientHealthChecksEndpoint(t *testing.T) {
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.Path != "/health" {
			t.Fatalf("path = %q, want /health", r.URL.Path)
		}
		return jsonResponse(http.StatusOK, `{"status":"ok"}`), nil
	})}

	client := &RerankClient{
		baseURL: "http://rerank.local",
		http:    httpClient,
	}

	if err := client.Health(context.Background()); err != nil {
		t.Fatalf("Health: %v", err)
	}
}

func TestHealthRejectsHTTPFailure(t *testing.T) {
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return jsonResponse(http.StatusServiceUnavailable, "not ready"), nil
	})}

	client := &EmbedClient{
		baseURL: "http://embedding.local",
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

func TestHealthRejectsNonOKStatusPayload(t *testing.T) {
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return jsonResponse(http.StatusOK, `{"status":"loading"}`), nil
	})}

	client := &EmbedClient{
		baseURL: "http://embedding.local",
		http:    httpClient,
	}

	err := client.Health(context.Background())
	if err == nil {
		t.Fatal("expected health error")
	}
	if !strings.Contains(err.Error(), "loading") {
		t.Fatalf("error = %v, want loading status", err)
	}
}

func TestHealthRejectsEmptyBaseURL(t *testing.T) {
	client := &EmbedClient{
		http: &http.Client{Timeout: time.Second},
	}

	err := client.Health(context.Background())
	if err == nil {
		t.Fatal("expected health error")
	}
	if !strings.Contains(err.Error(), "base_url") {
		t.Fatalf("error = %v, want base_url message", err)
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
