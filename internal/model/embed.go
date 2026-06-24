package model

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/raiki02/vidwise/internal/appconfig"
)

type EmbedClient struct {
	baseURL string
	model   string
	http    *http.Client
}

type EmbedRequest struct {
	Texts []string `json:"texts"`
	Model string   `json:"model"`
}

type EmbedResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
}

func NewEmbedClient(cfg appconfig.EmbeddingConfig) (*EmbedClient, error) {
	timeout, err := cfg.TimeoutDuration()
	if err != nil {
		timeout = 2 * time.Minute
	}
	return &EmbedClient{
		baseURL: cfg.BaseURL,
		model:   cfg.Model,
		http:    &http.Client{Timeout: timeout},
	}, nil
}

// Embed sends texts to the Python embedding service and returns vectors.
func (c *EmbedClient) Embed(ctx context.Context, texts []string) ([][]float64, error) {
	req := EmbedRequest{Texts: texts, Model: c.model}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal embed request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/embed", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create embed request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("call embedding service: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read embed response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embedding service returned %s: %s", resp.Status, string(respBody))
	}

	var output EmbedResponse
	if err := json.Unmarshal(respBody, &output); err != nil {
		return nil, fmt.Errorf("decode embed response: %w", err)
	}
	return output.Embeddings, nil
}

// EmbedSingle is a convenience method to embed a single text.
func (c *EmbedClient) EmbedSingle(ctx context.Context, text string) ([]float64, error) {
	embeddings, err := c.Embed(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}
	return embeddings[0], nil
}
