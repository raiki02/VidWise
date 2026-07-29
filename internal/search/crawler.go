package search

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

type BasicCrawlerConfig struct {
	Client           *http.Client
	Timeout          time.Duration
	MaxResponseBytes int64
	MaxConcurrency   int
	UserAgent        string
	AllowLocalhost   bool
	RobotsPolicy     string
	Logger           *slog.Logger
}

type BasicCrawler struct {
	client           *http.Client
	maxResponseBytes int64
	maxConcurrency   int
	userAgent        string
	allowLocalhost   bool
	robotsPolicy     string
	logger           *slog.Logger
}

func NewBasicCrawler(cfg BasicCrawlerConfig) *BasicCrawler {
	if cfg.Timeout <= 0 {
		cfg.Timeout = 10 * time.Second
	}
	if cfg.MaxResponseBytes <= 0 {
		cfg.MaxResponseBytes = 2 * 1024 * 1024
	}
	if cfg.MaxConcurrency <= 0 {
		cfg.MaxConcurrency = 4
	}
	if cfg.UserAgent == "" {
		cfg.UserAgent = "VidwiseSearchBot/0.1 (+https://example.com)"
	}
	if cfg.Client == nil {
		cfg.Client = &http.Client{Timeout: cfg.Timeout}
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	return &BasicCrawler{
		client:           cfg.Client,
		maxResponseBytes: cfg.MaxResponseBytes,
		maxConcurrency:   cfg.MaxConcurrency,
		userAgent:        cfg.UserAgent,
		allowLocalhost:   cfg.AllowLocalhost,
		robotsPolicy:     cfg.RobotsPolicy,
		logger:           cfg.Logger,
	}
}

func (c *BasicCrawler) Fetch(ctx context.Context, urls []string) ([]Document, error) {
	if c == nil {
		return nil, errors.New("crawler is nil")
	}
	if len(urls) == 0 {
		return nil, nil
	}
	if c.robotsPolicy != "" {
		c.logger.Debug("search.crawler.robots_policy_reserved", "policy", c.robotsPolicy)
	}

	sem := make(chan struct{}, c.maxConcurrency)
	results := make([]Document, len(urls))
	ok := make([]bool, len(urls))
	errs := make([]error, 0)
	var mu sync.Mutex
	var wg sync.WaitGroup

	for i, rawURL := range urls {
		i, rawURL := i, rawURL
		wg.Add(1)
		go func() {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-ctx.Done():
				mu.Lock()
				errs = append(errs, fmt.Errorf("fetch %q: %w", rawURL, ctx.Err()))
				mu.Unlock()
				return
			}

			doc, err := c.fetchOne(ctx, rawURL)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				errs = append(errs, err)
				c.logger.Warn("search.crawler.fetch_failed", "url", rawURL, "err", err)
				return
			}
			results[i] = doc
			ok[i] = true
		}()
	}
	wg.Wait()

	docs := make([]Document, 0, len(urls))
	for i, doc := range results {
		if ok[i] {
			docs = append(docs, doc)
		}
	}
	if len(docs) == 0 && len(errs) > 0 {
		return nil, fmt.Errorf("fetch urls: %w", errors.Join(errs...))
	}
	if len(errs) > 0 {
		c.logger.Warn("search.crawler.partial_failure", "failed", len(errs), "total", len(urls))
	}
	return docs, nil
}

func (c *BasicCrawler) fetchOne(ctx context.Context, rawURL string) (Document, error) {
	parsed, err := parseCrawlURL(rawURL, c.allowLocalhost)
	if err != nil {
		return Document{}, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, parsed.String(), nil)
	if err != nil {
		return Document{}, fmt.Errorf("build request for %q: %w", rawURL, err)
	}
	req.Header.Set("User-Agent", c.userAgent)
	req.Header.Set("Accept", "text/html,application/xhtml+xml,text/plain;q=0.8,*/*;q=0.5")

	resp, err := c.client.Do(req)
	if err != nil {
		return Document{}, fmt.Errorf("fetch %q: %w", rawURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return Document{}, fmt.Errorf("fetch %q: unexpected status %d", rawURL, resp.StatusCode)
	}

	limited := io.LimitReader(resp.Body, c.maxResponseBytes+1)
	body, err := io.ReadAll(limited)
	if err != nil {
		return Document{}, fmt.Errorf("read %q: %w", rawURL, err)
	}
	if int64(len(body)) > c.maxResponseBytes {
		return Document{}, fmt.Errorf("read %q: response exceeds %d bytes", rawURL, c.maxResponseBytes)
	}

	return Document{
		URL:        parsed.String(),
		StatusCode: resp.StatusCode,
		FetchedAt:  time.Now(),
		html:       string(body),
	}, nil
}

func parseCrawlURL(rawURL string, allowLocalhost bool) (*url.URL, error) {
	parsed, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil {
		return nil, fmt.Errorf("parse url %q: %w", rawURL, err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, fmt.Errorf("url %q has unsupported scheme", rawURL)
	}
	if parsed.Hostname() == "" {
		return nil, fmt.Errorf("url %q is missing host", rawURL)
	}
	if !allowLocalhost && isLocalHost(parsed.Hostname()) {
		return nil, fmt.Errorf("url %q points to a local/private host", rawURL)
	}
	return parsed, nil
}

func isLocalHost(host string) bool {
	host = strings.ToLower(strings.Trim(host, "[]"))
	if host == "localhost" || strings.HasSuffix(host, ".localhost") || strings.HasSuffix(host, ".local") {
		return true
	}
	ip := net.ParseIP(host)
	if ip == nil {
		return false
	}
	return ip.IsLoopback() || ip.IsPrivate() || ip.IsUnspecified() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast()
}
