package search

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

type HTTPProviderConfig struct {
	BaseURL     string
	APIKey      string
	APIKeyEnv   string
	MaxResults  int
	SearchDepth string
	Client      *http.Client
	Timeout     time.Duration
	UserAgent   string
}

type BingProvider struct {
	cfg HTTPProviderConfig
}

type TavilyProvider struct {
	cfg HTTPProviderConfig
}

type DuckDuckGoProvider struct {
	cfg HTTPProviderConfig
}

func NewBingProvider(cfg HTTPProviderConfig) (*BingProvider, error) {
	cfg = normalizeHTTPProviderConfig(cfg, "https://api.bing.microsoft.com/v7.0/search", "BING_SEARCH_API_KEY")
	if resolveAPIKey(cfg) == "" {
		return nil, errors.New("bing search api key is required")
	}
	return &BingProvider{cfg: cfg}, nil
}

func NewTavilyProvider(cfg HTTPProviderConfig) (*TavilyProvider, error) {
	cfg = normalizeHTTPProviderConfig(cfg, "https://api.tavily.com/search", "TAVILY_API_KEY")
	if cfg.SearchDepth == "" {
		cfg.SearchDepth = "basic"
	}
	if resolveAPIKey(cfg) == "" {
		return nil, errors.New("tavily api key is required")
	}
	return &TavilyProvider{cfg: cfg}, nil
}

func NewDuckDuckGoProvider(cfg HTTPProviderConfig) (*DuckDuckGoProvider, error) {
	cfg = normalizeHTTPProviderConfig(cfg, "https://duckduckgo.com/html/", "")
	return &DuckDuckGoProvider{cfg: cfg}, nil
}

func (p *BingProvider) Search(ctx context.Context, query string) ([]SearchItem, error) {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, errors.New("bing search query is required")
	}
	endpoint, err := url.Parse(p.cfg.BaseURL)
	if err != nil {
		return nil, fmt.Errorf("parse bing base url: %w", err)
	}
	params := endpoint.Query()
	params.Set("q", query)
	params.Set("count", strconv.Itoa(p.cfg.MaxResults))
	params.Set("responseFilter", "Webpages")
	params.Set("safeSearch", "Moderate")
	params.Set("textDecorations", "false")
	params.Set("textFormat", "Raw")
	endpoint.RawQuery = params.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("create bing search request: %w", err)
	}
	req.Header.Set("Ocp-Apim-Subscription-Key", resolveAPIKey(p.cfg))
	req.Header.Set("Accept", "application/json")
	setUserAgent(req, p.cfg.UserAgent)

	var output bingSearchResponse
	if err := doJSON(p.cfg.Client, req, nil, &output); err != nil {
		return nil, fmt.Errorf("call bing search: %w", err)
	}
	items := make([]SearchItem, 0, len(output.WebPages.Value))
	for _, value := range output.WebPages.Value {
		publishedAt := parseProviderTime(value.DateLastCrawled)
		items = append(items, SearchItem{
			Title:       value.Name,
			URL:         value.URL,
			Snippet:     value.Snippet,
			Provider:    ProviderBing,
			PublishedAt: publishedAt,
		})
	}
	return items, nil
}

func (p *TavilyProvider) Search(ctx context.Context, query string) ([]SearchItem, error) {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, errors.New("tavily search query is required")
	}
	body := tavilySearchRequest{
		APIKey:      resolveAPIKey(p.cfg),
		Query:       query,
		SearchDepth: p.cfg.SearchDepth,
		MaxResults:  p.cfg.MaxResults,
	}
	reqBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal tavily search request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.cfg.BaseURL, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("create tavily search request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	setUserAgent(req, p.cfg.UserAgent)

	var output tavilySearchResponse
	if err := doJSON(p.cfg.Client, req, reqBody, &output); err != nil {
		return nil, fmt.Errorf("call tavily search: %w", err)
	}
	items := make([]SearchItem, 0, len(output.Results))
	for _, result := range output.Results {
		content := firstNonEmpty(result.Content, result.RawContent)
		items = append(items, SearchItem{
			Title:       result.Title,
			URL:         result.URL,
			Snippet:     content,
			Provider:    ProviderTavily,
			PublishedAt: parseProviderTime(result.PublishedDate),
		})
	}
	return items, nil
}

func (p *DuckDuckGoProvider) Search(ctx context.Context, query string) ([]SearchItem, error) {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, errors.New("duckduckgo search query is required")
	}
	endpoint, err := url.Parse(p.cfg.BaseURL)
	if err != nil {
		return nil, fmt.Errorf("parse duckduckgo base url: %w", err)
	}
	params := endpoint.Query()
	params.Set("q", query)
	endpoint.RawQuery = params.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("create duckduckgo search request: %w", err)
	}
	req.Header.Set("Accept", "text/html,application/xhtml+xml")
	setUserAgent(req, p.cfg.UserAgent)

	resp, err := p.cfg.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("call duckduckgo search: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("duckduckgo search returned %s: %s", resp.Status, string(body))
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 2*1024*1024))
	if err != nil {
		return nil, fmt.Errorf("read duckduckgo search response: %w", err)
	}
	return parseDuckDuckGoHTML(string(body), p.cfg.MaxResults), nil
}

type bingSearchResponse struct {
	WebPages struct {
		Value []struct {
			Name            string `json:"name"`
			URL             string `json:"url"`
			Snippet         string `json:"snippet"`
			DateLastCrawled string `json:"dateLastCrawled"`
		} `json:"value"`
	} `json:"webPages"`
}

type tavilySearchRequest struct {
	APIKey      string `json:"api_key"`
	Query       string `json:"query"`
	SearchDepth string `json:"search_depth,omitempty"`
	MaxResults  int    `json:"max_results,omitempty"`
}

type tavilySearchResponse struct {
	Results []struct {
		Title         string `json:"title"`
		URL           string `json:"url"`
		Content       string `json:"content"`
		RawContent    string `json:"raw_content"`
		PublishedDate string `json:"published_date"`
	} `json:"results"`
}

func normalizeHTTPProviderConfig(cfg HTTPProviderConfig, defaultBaseURL, defaultAPIKeyEnv string) HTTPProviderConfig {
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.APIKeyEnv == "" {
		cfg.APIKeyEnv = defaultAPIKeyEnv
	}
	if cfg.MaxResults <= 0 {
		cfg.MaxResults = 10
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 10 * time.Second
	}
	if cfg.Client == nil {
		cfg.Client = &http.Client{Timeout: cfg.Timeout}
	}
	return cfg
}

func resolveAPIKey(cfg HTTPProviderConfig) string {
	if strings.TrimSpace(cfg.APIKey) != "" {
		return strings.TrimSpace(cfg.APIKey)
	}
	if strings.TrimSpace(cfg.APIKeyEnv) == "" {
		return ""
	}
	return strings.TrimSpace(os.Getenv(cfg.APIKeyEnv))
}

func doJSON(client *http.Client, req *http.Request, _ []byte, out any) error {
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 4*1024*1024))
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return fmt.Errorf("unexpected status %s: %s", resp.Status, string(body))
	}
	if err := json.Unmarshal(body, out); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}
	return nil
}

func setUserAgent(req *http.Request, userAgent string) {
	if strings.TrimSpace(userAgent) != "" {
		req.Header.Set("User-Agent", userAgent)
	}
}

func parseProviderTime(raw string) time.Time {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return time.Time{}
	}
	layouts := []string{time.RFC3339, "2006-01-02", "2006-01-02 15:04:05"}
	for _, layout := range layouts {
		if t, err := time.Parse(layout, raw); err == nil {
			return t
		}
	}
	return time.Time{}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			return value
		}
	}
	return ""
}
