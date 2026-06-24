package model

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/raiki02/video-extractor/internal/appconfig"
)

type RerankClient struct {
	baseURL string
	model   string
	topK    int
	http    *http.Client
}

type RerankRequest struct {
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	Model     string   `json:"model"`
	TopK      int      `json:"top_k"`
}

type RerankResponse struct {
	Results []RerankResult `json:"results"`
}

type RerankResult struct {
	Index int     `json:"index"`
	Score float64 `json:"score"`
	Text  string  `json:"text,omitempty"`
}

func NewRerankClient(cfg appconfig.RerankConfig) (*RerankClient, error) {
	timeout, err := cfg.TimeoutDuration()
	if err != nil {
		timeout = 1 * time.Minute
	}
	return &RerankClient{
		baseURL: cfg.BaseURL,
		model:   cfg.Model,
		topK:    cfg.TopK,
		http:    &http.Client{Timeout: timeout},
	}, nil
}

// Rerank sends query and documents to the Python reranking service.
func (c *RerankClient) Rerank(ctx context.Context, query string, documents []string) ([]RerankResult, error) {
	topK := c.topK
	if topK > len(documents) {
		topK = len(documents)
	}

	req := RerankRequest{
		Query:     query,
		Documents: documents,
		Model:     c.model,
		TopK:      topK,
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal rerank request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/rerank", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create rerank request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("call rerank service: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read rerank response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("rerank service returned %s: %s", resp.Status, string(respBody))
	}

	var output RerankResponse
	if err := json.Unmarshal(respBody, &output); err != nil {
		return nil, fmt.Errorf("decode rerank response: %w", err)
	}
	return output.Results, nil
}
