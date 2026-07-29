package search

import "time"

type ProviderName string

const (
	ProviderMock          ProviderName = "mock"
	ProviderBing          ProviderName = "bing"
	ProviderTavily        ProviderName = "tavily"
	ProviderDuckDuckGo    ProviderName = "duckduckgo"
	ProviderInternal      ProviderName = "internal"
	ProviderGitHub        ProviderName = "github"
	ProviderDocumentation ProviderName = "documentation"
)

type ProviderRegistration struct {
	Name     ProviderName
	Provider SearchProvider
}

type SearchItem struct {
	Title       string
	URL         string
	Snippet     string
	Provider    ProviderName
	PublishedAt time.Time
}

type Document struct {
	Title       string    `json:"title"`
	URL         string    `json:"url"`
	Content     string    `json:"content"`
	Provider    string    `json:"-"`
	Score       float64   `json:"-"`
	StatusCode  int       `json:"-"`
	FetchedAt   time.Time `json:"-"`
	PublishedAt time.Time `json:"-"`

	html string
}

type Source struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Content string `json:"content"`
}

type SearchResult struct {
	Summary string   `json:"summary"`
	Sources []Source `json:"sources"`

	Query            string        `json:"-"`
	RewrittenQueries []string      `json:"-"`
	Cached           bool          `json:"-"`
	Quality          SearchQuality `json:"-"`
}

type SearchQuality struct {
	Score       float64 `json:"score"`
	SourceCount int     `json:"source_count"`
	Reason      string  `json:"reason"`
}
