package search

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
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
