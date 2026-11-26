package embedding
package embedding

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestClientHealth(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Errorf("path = %q", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"status":"ok"}`)
	}))
	defer srv.Close()

	client, err := NewClient(srv.URL, "qwen", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if err := client.Health(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestClientEmbed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embed" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("method = %q", r.Method)
		}
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"model":"qwen"`) {
			t.Errorf("request body missing model: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"embeddings":[[0.1,0.2],[0.3,0.4]]}`)
	}))
	defer srv.Close()

	client, err := NewClient(srv.URL, "qwen", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	vectors, err := client.Embed(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vectors) != 2 || len(vectors[0]) != 2 || vectors[1][1] != 0.4 {
		t.Fatalf("vectors = %v", vectors)
	}
}

func TestClientEmbedRejectsCountMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"embeddings":[[0.1]]}`)
	}))
	defer srv.Close()

	client, err := NewClient(srv.URL, "m", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Embed(context.Background(), []string{"a", "b"}); err == nil {
		t.Fatal("expected count mismatch error")
	}
}

func TestClientEmbedEmpty(t *testing.T) {
	client, err := NewClient("http://localhost:1", "m", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	vectors, err := client.Embed(context.Background(), nil)
	if err != nil || vectors != nil {
		t.Fatalf("empty embed = %v, %v", vectors, err)
	}
}

func TestNewClientRequiresBaseURL(t *testing.T) {
	if _, err := NewClient("", "m", time.Second); err == nil {
		t.Fatal("expected error for empty base_url")
	}
}