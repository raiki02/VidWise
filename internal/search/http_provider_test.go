package search

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestBingProviderSearchMapsResponse(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if got := r.Header.Get("Ocp-Apim-Subscription-Key"); got != "secret" {
			t.Fatalf("api key header = %q", got)
		}
		if got := r.URL.Query().Get("q"); got != "OpenAI latest" {
			t.Fatalf("query = %q", got)
		}
		return jsonResponse(r, `{"webPages":{"value":[{"name":"OpenAI","url":"https://example.com/openai","snippet":"latest news","dateLastCrawled":"2026-07-28T00:00:00Z"}]}}`)
	})}
	provider, err := NewBingProvider(HTTPProviderConfig{
		APIKey:     "secret",
		Client:     client,
		MaxResults: 2,
	})
	if err != nil {
		t.Fatalf("NewBingProvider() error = %v", err)
	}

	items, err := provider.Search(context.Background(), "OpenAI latest")
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(items) != 1 || items[0].Provider != ProviderBing || items[0].Title != "OpenAI" {
		t.Fatalf("unexpected items: %#v", items)
	}
}

func TestTavilyProviderSearchMapsResponse(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"api_key":"secret"`) {
			t.Fatalf("request body missing api key: %s", string(body))
		}
		return jsonResponse(r, `{"results":[{"title":"Doc","url":"https://example.com/doc","content":"useful content","published_date":"2026-07-28"}]}`)
	})}
	provider, err := NewTavilyProvider(HTTPProviderConfig{
		APIKey:     "secret",
		Client:     client,
		MaxResults: 1,
	})
	if err != nil {
		t.Fatalf("NewTavilyProvider() error = %v", err)
	}

	items, err := provider.Search(context.Background(), "docs")
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(items) != 1 || items[0].Provider != ProviderTavily || items[0].Snippet != "useful content" {
		t.Fatalf("unexpected items: %#v", items)
	}
}

func TestDuckDuckGoProviderSearchParsesHTML(t *testing.T) {
	html := `<html><body>
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpost">Example Post</a>
<a class="result__snippet">Snippet text</a>
</body></html>`
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(html)),
			Header:     make(http.Header),
			Request:    r,
		}, nil
	})}
	provider, err := NewDuckDuckGoProvider(HTTPProviderConfig{
		Client:     client,
		MaxResults: 1,
	})
	if err != nil {
		t.Fatalf("NewDuckDuckGoProvider() error = %v", err)
	}

	items, err := provider.Search(context.Background(), "example")
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("items len = %d, want 1", len(items))
	}
	if items[0].URL != "https://example.com/post" || items[0].Snippet != "Snippet text" {
		t.Fatalf("unexpected item: %#v", items[0])
	}
}

func jsonResponse(r *http.Request, body string) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     make(http.Header),
		Request:    r,
	}, nil
}
