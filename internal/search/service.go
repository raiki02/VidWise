package search

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"strings"
	"time"
)

type ServiceConfig struct {
	MaxProviderResults int
}

type ServiceDependencies struct {
	Cache      SearchCache
	Rewriter   QueryRewriter
	Router     ProviderRouter
	Crawler    Crawler
	Extractor  Extractor
	Reranker   Reranker
	Compressor Compressor
	Quality    QualityEvaluator
	Metrics    Metrics
	Logger     *slog.Logger
	Config     ServiceConfig
}

type Service struct {
	cache      SearchCache
	rewriter   QueryRewriter
	router     ProviderRouter
	crawler    Crawler
	extractor  Extractor
	reranker   Reranker
	compressor Compressor
	quality    QualityEvaluator
	metrics    Metrics
	logger     *slog.Logger
	cfg        ServiceConfig
}

func NewService(deps ServiceDependencies) (*Service, error) {
	if deps.Router == nil {
		return nil, errors.New("search router is required")
	}
	if deps.Crawler == nil {
		return nil, errors.New("search crawler is required")
	}
	if deps.Extractor == nil {
		return nil, errors.New("search extractor is required")
	}
	if deps.Reranker == nil {
		return nil, errors.New("search reranker is required")
	}
	if deps.Compressor == nil {
		return nil, errors.New("search compressor is required")
	}
	if deps.Quality == nil {
		deps.Quality = NewBasicQualityEvaluator()
	}
	if deps.Rewriter == nil {
		deps.Rewriter = MockQueryRewriter{}
	}
	if deps.Cache == nil {
		deps.Cache = NewMemoryCache(MemoryCacheConfig{})
	}
	if deps.Metrics == nil {
		deps.Metrics = noopMetrics{}
	}
	if deps.Logger == nil {
		deps.Logger = slog.Default()
	}
	if deps.Config.MaxProviderResults <= 0 {
		deps.Config.MaxProviderResults = 10
	}
	return &Service{
		cache:      deps.Cache,
		rewriter:   deps.Rewriter,
		router:     deps.Router,
		crawler:    deps.Crawler,
		extractor:  deps.Extractor,
		reranker:   deps.Reranker,
		compressor: deps.Compressor,
		quality:    deps.Quality,
		metrics:    deps.Metrics,
		logger:     deps.Logger,
		cfg:        deps.Config,
	}, nil
}

func NewMockService() (*Service, error) {
	router := NewProviderRouter(ProviderRegistration{
		Name:     ProviderMock,
		Provider: NewMockProvider(nil),
	})
	return NewService(ServiceDependencies{
		Router:     router,
		Crawler:    NewNoopCrawler(),
		Extractor:  NewBasicExtractor(),
		Reranker:   NewKeywordReranker(),
		Compressor: NewBasicCompressor(BasicCompressorConfig{}),
		Quality:    NewBasicQualityEvaluator(),
	})
}

func (s *Service) Search(ctx context.Context, query string) (*SearchResult, error) {
	start := time.Now()
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, errors.New("search query is required")
	}

	if result, ok := s.cache.Get(query); ok {
		s.metrics.ObserveCache(ctx, true)
		s.metrics.ObserveSearch(ctx, "cache_hit", time.Since(start))
		s.logger.InfoContext(ctx, "search.cache_hit", logAttrsWithTrace(ctx, "query", query, "sources", len(result.Sources))...)
		return result, nil
	}
	s.metrics.ObserveCache(ctx, false)

	rewrittenQueries, err := s.rewriter.Rewrite(ctx, query)
	if err != nil {
		s.metrics.ObserveSearch(ctx, "rewrite_failed", time.Since(start))
		return nil, fmt.Errorf("rewrite search query: %w", err)
	}
	rewrittenQueries = normalizeQueries(query, rewrittenQueries)

	items, providerErr := s.searchProviders(ctx, rewrittenQueries)
	if providerErr != nil && len(items) == 0 {
		s.metrics.ObserveSearch(ctx, "provider_failed", time.Since(start))
		return nil, providerErr
	}
	if providerErr != nil {
		s.logger.WarnContext(ctx, "search.provider_partial_failure", logAttrsWithTrace(ctx, "err", providerErr)...)
	}

	docs := s.documentsFromProviderSnippets(items)
	urls := filterSearchURLs(items)
	if len(urls) > 0 {
		fetched, err := s.crawler.Fetch(ctx, urls)
		if err != nil {
			s.logger.WarnContext(ctx, "search.crawl_failed", logAttrsWithTrace(ctx, "err", err)...)
		} else if len(fetched) > 0 {
			extracted, err := s.extractor.Extract(ctx, mergeProviderMetadata(fetched, items))
			if err != nil {
				s.metrics.ObserveSearch(ctx, "extract_failed", time.Since(start))
				return nil, fmt.Errorf("extract crawled documents: %w", err)
			}
			docs = mergeDocuments(extracted, docs)
		}
	}

	ranked := s.rankDocuments(ctx, query, docs)
	compressed := s.compressor.Compress(ranked)
	quality := s.quality.Evaluate(query, compressed)
	result := buildSearchResult(query, rewrittenQueries, compressed, quality)
	s.cache.Set(query, *result)
	s.metrics.ObserveSearch(ctx, "ok", time.Since(start))
	s.logger.InfoContext(ctx, "search.done", logAttrsWithTrace(ctx, "query", query, "rewritten_queries", len(rewrittenQueries), "sources", len(result.Sources), "quality_score", quality.Score, "quality_reason", quality.Reason, "elapsed", time.Since(start))...)
	return result, nil
}

func (s *Service) rankDocuments(ctx context.Context, query string, docs []Document) []Document {
	contextual, ok := s.reranker.(ContextualReranker)
	if !ok {
		return s.reranker.Rank(query, docs)
	}
	ranked, err := contextual.RankWithContext(ctx, query, docs)
	if err != nil {
		s.logger.WarnContext(ctx, "search.rerank_failed_fallback", logAttrsWithTrace(ctx, "err", err)...)
		return NewKeywordReranker().Rank(query, docs)
	}
	return ranked
}

func (s *Service) searchProviders(ctx context.Context, queries []string) ([]SearchItem, error) {
	var items []SearchItem
	var errs []error
	seenURLs := map[string]struct{}{}

	for _, query := range queries {
		providers, err := s.router.Route(ctx, query)
		if err != nil {
			errs = append(errs, fmt.Errorf("route providers for %q: %w", query, err))
			continue
		}
		for _, registration := range providers {
			providerStart := time.Now()
			providerItems, err := registration.Provider.Search(ctx, query)
			status := "ok"
			if err != nil {
				status = "error"
				errs = append(errs, fmt.Errorf("search provider %s for %q: %w", registration.Name, query, err))
				s.metrics.ObserveProvider(ctx, string(registration.Name), status, time.Since(providerStart))
				continue
			}
			s.metrics.ObserveProvider(ctx, string(registration.Name), status, time.Since(providerStart))
			for _, item := range providerItems {
				if item.Provider == "" {
					item.Provider = registration.Name
				}
				key := canonicalURLKey(item.URL)
				if key != "" {
					if _, ok := seenURLs[key]; ok {
						continue
					}
					seenURLs[key] = struct{}{}
				}
				items = append(items, item)
				if len(items) >= s.cfg.MaxProviderResults {
					return items, errors.Join(errs...)
				}
			}
		}
	}
	return items, errors.Join(errs...)
}

func (s *Service) documentsFromProviderSnippets(items []SearchItem) []Document {
	docs := make([]Document, 0, len(items))
	for _, item := range items {
		content := strings.TrimSpace(item.Snippet)
		if content == "" {
			continue
		}
		docs = append(docs, Document{
			Title:       strings.TrimSpace(item.Title),
			URL:         strings.TrimSpace(item.URL),
			Content:     content,
			Provider:    string(item.Provider),
			PublishedAt: item.PublishedAt,
		})
	}
	return docs
}

type MockQueryRewriter struct {
	Responses map[string][]string
}

func (r MockQueryRewriter) Rewrite(ctx context.Context, query string) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("rewrite query canceled: %w", err)
	}
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, errors.New("query is required")
	}
	if len(r.Responses) > 0 {
		if out := r.Responses[query]; len(out) > 0 {
			return out, nil
		}
		if out := r.Responses[strings.ToLower(query)]; len(out) > 0 {
			return out, nil
		}
	}
	return []string{query}, nil
}

type NoopCrawler struct{}

func NewNoopCrawler() NoopCrawler {
	return NoopCrawler{}
}

func (NoopCrawler) Fetch(ctx context.Context, urls []string) ([]Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("noop crawl canceled: %w", err)
	}
	return nil, nil
}

func normalizeQueries(original string, rewritten []string) []string {
	out := make([]string, 0, len(rewritten)+1)
	seen := map[string]struct{}{}
	add := func(query string) {
		query = strings.TrimSpace(query)
		if query == "" {
			return
		}
		key := strings.ToLower(query)
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		out = append(out, query)
	}
	for _, query := range rewritten {
		add(query)
	}
	add(original)
	return out
}

func filterSearchURLs(items []SearchItem) []string {
	out := make([]string, 0, len(items))
	seen := map[string]struct{}{}
	for _, item := range items {
		key := canonicalURLKey(item.URL)
		if key == "" {
			continue
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, key)
	}
	return out
}

func canonicalURLKey(rawURL string) string {
	parsed, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil || parsed == nil {
		return ""
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return ""
	}
	if parsed.Hostname() == "" || isLocalHost(parsed.Hostname()) {
		return ""
	}
	parsed.Fragment = ""
	return parsed.String()
}

func mergeProviderMetadata(docs []Document, items []SearchItem) []Document {
	metadata := map[string]SearchItem{}
	for _, item := range items {
		key := canonicalURLKey(item.URL)
		if key != "" {
			metadata[key] = item
		}
	}
	out := make([]Document, 0, len(docs))
	for _, doc := range docs {
		key := canonicalURLKey(doc.URL)
		if item, ok := metadata[key]; ok {
			if doc.Title == "" {
				doc.Title = item.Title
			}
			if doc.Provider == "" {
				doc.Provider = string(item.Provider)
			}
			doc.PublishedAt = item.PublishedAt
		}
		out = append(out, doc)
	}
	return out
}

func mergeDocuments(primary, fallback []Document) []Document {
	out := make([]Document, 0, len(primary)+len(fallback))
	seen := map[string]struct{}{}
	for _, doc := range append(primary, fallback...) {
		key := canonicalURLKey(doc.URL)
		if key == "" {
			key = strings.ToLower(strings.TrimSpace(doc.Title + "\x00" + doc.Content))
		}
		if key == "" {
			continue
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, doc)
	}
	return out
}

func buildSearchResult(query string, rewrittenQueries []string, docs []Document, quality SearchQuality) *SearchResult {
	sources := make([]Source, 0, len(docs))
	for _, doc := range docs {
		sources = append(sources, Source{
			Title:   strings.TrimSpace(doc.Title),
			URL:     strings.TrimSpace(doc.URL),
			Content: strings.TrimSpace(doc.Content),
		})
	}

	summary := "No relevant web sources were found."
	if len(sources) > 0 {
		summary = fmt.Sprintf("Found %d relevant web source(s) for %q.", len(sources), query)
	}
	return &SearchResult{
		Summary:          summary,
		Sources:          sources,
		Query:            query,
		RewrittenQueries: rewrittenQueries,
		Quality:          quality,
	}
}
