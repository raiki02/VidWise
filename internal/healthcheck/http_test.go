package healthcheck

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestCheckHTTPAcceptsOKHealth(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.Method != http.MethodGet {
			t.Fatalf("method = %q, want GET", r.Method)
		}
		if r.URL.String() != "http://svc.local/health" {
			t.Fatalf("url = %q, want http://svc.local/health", r.URL.String())
		}
		return jsonResponse(http.StatusOK, `{"status":"ok"}`), nil
	})}

	if err := CheckHTTP(context.Background(), client, "http://svc.local/", "svc"); err != nil {
		t.Fatalf("CheckHTTP: %v", err)
	}
}

func TestCheckHTTPRejectsHTTPFailure(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return jsonResponse(http.StatusServiceUnavailable, "not ready"), nil
	})}

	err := CheckHTTP(context.Background(), client, "http://svc.local", "svc")
	if err == nil {
		t.Fatal("expected health error")
	}
	if !strings.Contains(err.Error(), "503") {
		t.Fatalf("error = %v, want status code", err)
	}
}

func TestCheckHTTPRejectsNonOKStatusPayload(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return jsonResponse(http.StatusOK, `{"status":"loading"}`), nil
	})}

	err := CheckHTTP(context.Background(), client, "http://svc.local", "svc")
	if err == nil {
		t.Fatal("expected health error")
	}
	if !strings.Contains(err.Error(), "loading") {
		t.Fatalf("error = %v, want loading status", err)
	}
}

func TestCheckHTTPRejectsEmptyBaseURL(t *testing.T) {
	err := CheckHTTP(context.Background(), nil, "  ", "svc")
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
