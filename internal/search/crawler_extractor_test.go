package search

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestBasicCrawlerFetchesHTMLWithLimits(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if got := r.Header.Get("User-Agent"); got != "test-agent" {
			t.Fatalf("User-Agent = %q, want test-agent", got)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader("<html><body>Hello crawler</body></html>")),
			Header:     make(http.Header),
			Request:    r,
		}, nil
	})}

	crawler := NewBasicCrawler(BasicCrawlerConfig{
		Client:           client,
		UserAgent:        "test-agent",
		MaxResponseBytes: 1024,
	})
	docs, err := crawler.Fetch(context.Background(), []string{"https://example.com/page"})
	if err != nil {
		t.Fatalf("Fetch() error = %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("docs len = %d, want 1", len(docs))
	}
	if !strings.Contains(docs[0].html, "Hello crawler") {
		t.Fatalf("raw html not captured internally: %q", docs[0].html)
	}
}

func TestBasicCrawlerRejectsOversizedResponses(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader("0123456789")),
			Header:     make(http.Header),
			Request:    r,
		}, nil
	})}

	crawler := NewBasicCrawler(BasicCrawlerConfig{
		Client:           client,
		MaxResponseBytes: 4,
	})
	if _, err := crawler.Fetch(context.Background(), []string{"https://example.com/large"}); err == nil {
		t.Fatal("expected oversized response error")
	}
}

func TestBasicCrawlerLimitsConcurrency(t *testing.T) {
	var active int32
	var maxActive int32
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		now := atomic.AddInt32(&active, 1)
		for {
			peak := atomic.LoadInt32(&maxActive)
			if now <= peak || atomic.CompareAndSwapInt32(&maxActive, peak, now) {
				break
			}
		}
		defer atomic.AddInt32(&active, -1)

		time.Sleep(25 * time.Millisecond)
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader("<html><body>ok</body></html>")),
			Header:     make(http.Header),
			Request:    r,
		}, nil
	})}

	crawler := NewBasicCrawler(BasicCrawlerConfig{
		Client:         client,
		MaxConcurrency: 3,
	})
	urls := []string{
		"https://example.com/1",
		"https://example.com/2",
		"https://example.com/3",
		"https://example.com/4",
		"https://example.com/5",
		"https://example.com/6",
	}
	docs, err := crawler.Fetch(context.Background(), urls)
	if err != nil {
		t.Fatalf("Fetch() error = %v", err)
	}
	if len(docs) != len(urls) {
		t.Fatalf("docs len = %d, want %d", len(docs), len(urls))
	}
	if got := atomic.LoadInt32(&maxActive); got > 3 {
		t.Fatalf("max concurrency = %d, want <= 3", got)
	}
}

func TestBasicExtractorRemovesNoiseAndTitleFromContent(t *testing.T) {
	extractor := NewBasicExtractor()
	docs, err := extractor.Extract(context.Background(), []Document{{
		URL: "https://example.com/post",
		html: `<html>
<head><title>Post title</title><script>bad()</script><style>.x{}</style></head>
<body><header>top</header><nav>menu</nav><div class="ad-card">ad copy</div><main>Useful article text.</main><footer>bottom</footer></body>
</html>`,
	}})
	if err != nil {
		t.Fatalf("Extract() error = %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("docs len = %d, want 1", len(docs))
	}
	if docs[0].Title != "Post title" {
		t.Fatalf("title = %q", docs[0].Title)
	}
	if docs[0].Content != "Useful article text." {
		t.Fatalf("content = %q", docs[0].Content)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func BenchmarkBasicCrawlerFetch(b *testing.B) {
	runCrawlerBenchmark(b, false)
}

func BenchmarkLegacyBasicCrawlerFetch(b *testing.B) {
	runCrawlerBenchmark(b, true)
}

func runCrawlerBenchmark(b *testing.B, legacy bool) {
	transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader("<html><body>ok</body></html>")),
			Header:     make(http.Header),
			Request:    r,
		}, nil
	})
	crawler := NewBasicCrawler(BasicCrawlerConfig{
		Client:         &http.Client{Transport: transport},
		MaxConcurrency: 4,
		Logger:         slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	urls := make([]string, 64)
	for i := range urls {
		urls[i] = "https://example.com/" + strings.Repeat("a", 8)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if legacy {
			_ = legacyBasicCrawlerFetch(context.Background(), crawler, urls)
			continue
		}
		_, _ = crawler.Fetch(context.Background(), urls)
	}
}

func legacyBasicCrawlerFetch(ctx context.Context, c *BasicCrawler, urls []string) []Document {
	if c == nil || len(urls) == 0 {
		return nil
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
				errs = append(errs, ctx.Err())
				mu.Unlock()
				return
			}

			doc, err := c.fetchOne(ctx, rawURL)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				errs = append(errs, err)
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
		return nil
	}
	return docs
}
