package search

import (
	"strings"
	"sync"
	"time"
)

type MemoryCacheConfig struct {
	TTL     time.Duration
	MaxKeys int
	Now     func() time.Time
}

type MemoryCache struct {
	mu      sync.RWMutex
	ttl     time.Duration
	maxKeys int
	now     func() time.Time
	items   map[string]cacheEntry
	order   []string
}

type cacheEntry struct {
	result    SearchResult
	expiresAt time.Time
}

func NewMemoryCache(cfg MemoryCacheConfig) *MemoryCache {
	if cfg.TTL <= 0 {
		cfg.TTL = 5 * time.Minute
	}
	if cfg.MaxKeys <= 0 {
		cfg.MaxKeys = 1024
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	return &MemoryCache{
		ttl:     cfg.TTL,
		maxKeys: cfg.MaxKeys,
		now:     cfg.Now,
		items:   make(map[string]cacheEntry),
		order:   make([]string, 0, cfg.MaxKeys),
	}
}

func (c *MemoryCache) Get(query string) (*SearchResult, bool) {
	if c == nil {
		return nil, false
	}
	key := normalizeCacheKey(query)
	if key == "" {
		return nil, false
	}

	c.mu.RLock()
	entry, ok := c.items[key]
	c.mu.RUnlock()
	if !ok {
		return nil, false
	}
	if !entry.expiresAt.IsZero() && c.now().After(entry.expiresAt) {
		c.mu.Lock()
		delete(c.items, key)
		c.mu.Unlock()
		return nil, false
	}

	result := cloneSearchResult(entry.result)
	result.Cached = true
	return &result, true
}

func (c *MemoryCache) Set(query string, result SearchResult) {
	if c == nil {
		return
	}
	key := normalizeCacheKey(query)
	if key == "" {
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if _, exists := c.items[key]; !exists {
		c.order = append(c.order, key)
	}
	c.items[key] = cacheEntry{
		result:    cloneSearchResult(result),
		expiresAt: c.now().Add(c.ttl),
	}
	c.evictLocked()
}

func (c *MemoryCache) evictLocked() {
	for len(c.items) > c.maxKeys && len(c.order) > 0 {
		key := c.order[0]
		c.order = c.order[1:]
		delete(c.items, key)
	}
}

func normalizeCacheKey(query string) string {
	return strings.ToLower(strings.TrimSpace(query))
}

func cloneSearchResult(result SearchResult) SearchResult {
	out := result
	if result.Sources != nil {
		out.Sources = append([]Source(nil), result.Sources...)
	}
	if result.RewrittenQueries != nil {
		out.RewrittenQueries = append([]string(nil), result.RewrittenQueries...)
	}
	return out
}
