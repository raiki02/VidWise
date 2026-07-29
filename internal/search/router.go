package search

import (
	"context"
	"fmt"
	"reflect"
	"strings"
)

type ProviderRouter interface {
	Route(ctx context.Context, query string) ([]ProviderRegistration, error)
}

type StaticProviderRouter struct {
	providers map[ProviderName]SearchProvider
	order     []ProviderName
}

func NewProviderRouter(registrations ...ProviderRegistration) *StaticProviderRouter {
	router := &StaticProviderRouter{
		providers: make(map[ProviderName]SearchProvider, len(registrations)),
		order:     make([]ProviderName, 0, len(registrations)),
	}
	for _, registration := range registrations {
		router.Register(registration)
	}
	return router
}

func (r *StaticProviderRouter) Register(registration ProviderRegistration) {
	if r == nil || registration.Provider == nil {
		return
	}
	name := registration.Name
	if name == "" {
		name = ProviderMock
	}
	if _, exists := r.providers[name]; !exists {
		r.order = append(r.order, name)
	}
	r.providers[name] = registration.Provider
}

func (r *StaticProviderRouter) Route(ctx context.Context, query string) ([]ProviderRegistration, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("route providers: %w", err)
	}
	if r == nil || len(r.providers) == 0 {
		return nil, fmt.Errorf("no search providers registered")
	}

	candidates := providerPreferenceForQuery(query)
	out := make([]ProviderRegistration, 0, len(r.providers))
	seen := map[ProviderName]struct{}{}
	for _, name := range candidates {
		provider, ok := r.providers[name]
		if !ok || isNilProvider(provider) {
			continue
		}
		out = append(out, ProviderRegistration{Name: name, Provider: provider})
		seen[name] = struct{}{}
	}
	for _, name := range r.order {
		if _, ok := seen[name]; ok {
			continue
		}
		provider := r.providers[name]
		if isNilProvider(provider) {
			continue
		}
		out = append(out, ProviderRegistration{Name: name, Provider: provider})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no usable search providers registered")
	}
	return out, nil
}

func providerPreferenceForQuery(query string) []ProviderName {
	query = strings.ToLower(strings.TrimSpace(query))
	if isNewsQuery(query) {
		return []ProviderName{ProviderBing, ProviderTavily, ProviderDuckDuckGo, ProviderMock, ProviderInternal}
	}
	if isTechnicalQuery(query) {
		return []ProviderName{ProviderInternal, ProviderGitHub, ProviderDocumentation, ProviderDuckDuckGo, ProviderBing, ProviderMock}
	}
	return []ProviderName{ProviderInternal, ProviderBing, ProviderTavily, ProviderDuckDuckGo, ProviderMock}
}

func isNewsQuery(query string) bool {
	keywords := []string{
		"latest", "recent", "news", "announcement", "release", "today", "yesterday",
		"最近", "最新", "新闻", "发布", "公告", "今天", "昨天",
	}
	for _, keyword := range keywords {
		if strings.Contains(query, keyword) {
			return true
		}
	}
	return false
}

func isTechnicalQuery(query string) bool {
	keywords := []string{
		"github", "docs", "documentation", "api", "sdk", "error", "bug", "golang", "go ", "python",
		"文档", "代码", "接口", "报错", "错误", "库", "框架", "实现",
	}
	for _, keyword := range keywords {
		if strings.Contains(query, keyword) {
			return true
		}
	}
	return false
}

func isNilProvider(provider SearchProvider) bool {
	if provider == nil {
		return true
	}
	value := reflect.ValueOf(provider)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}
