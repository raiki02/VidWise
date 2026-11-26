package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/raiki02/vidwise/internal/healthcheck"
)

type Client struct {
	baseURL string
	model   string
	http    *http.Client
}

type EmbedRequest struct {
	Texts []string `json:"texts"`
	Model string   `json:"model,omitempty"`
}

type EmbedResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
}

// NewClient creates an embedding service client. model is sent on every
// request so the Python service can choose the right backend.
func NewClient(baseURL, model string, timeout time.Duration) (*Client, error) {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if baseURL == "" {
		return nil, fmt.Errorf("embedding base_url is required")
	}
	if _, err := url.ParseRequestURI(baseURL); err != nil {
		return nil, fmt.Errorf("invalid embedding base_url: %w", err)
	}
	if timeout <= 0 {
		timeout = 2 * time.Minute
	}
	return &Client{
		baseURL: baseURL,
		model:   model,
		http:    &http.Client{Timeout: timeout},
	}, nil
}

func (c *Client) Health(ctx context.Context) error {
	return healthcheck.CheckHTTP(ctx, c.http, c.baseURL, "embedding service")
}

// Embed embeds a batch of texts. A per-request batch is bounded by the caller;
// this client sends the full list in one request.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	body, err := json.Marshal(EmbedRequest{Texts: texts, Model: c.model})
	if err != nil {
		return nil, fmt.Errorf("marshal embed request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/embed", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create embed request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("call embedding service: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read embed response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embedding service returned %s: %s", resp.Status, strings.TrimSpace(string(respBody)))
	}

	var output EmbedResponse
	if err := json.Unmarshal(respBody, &output); err != nil {
		return nil, fmt.Errorf("decode embed response: %w", err)
	}
	if len(output.Embeddings) != len(texts) {
		return nil, fmt.Errorf("embedding service returned %d embeddings for %d texts", len(output.Embeddings), len(texts))
	}
	return output.Embeddings, nil
}
