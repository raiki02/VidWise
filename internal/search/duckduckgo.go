package search

import (
	"net/url"
	"strings"

	"golang.org/x/net/html"
)

func parseDuckDuckGoHTML(body string, maxResults int) []SearchItem {
	if maxResults <= 0 {
		maxResults = 10
	}
	root, err := html.Parse(strings.NewReader(body))
	if err != nil {
		return nil
	}

	var items []SearchItem
	var current *SearchItem
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if len(items) < maxResults && n.Type == html.ElementNode && n.Data == "a" && hasClass(n, "result__a") {
			item := SearchItem{
				Title:    normalizeWhitespace(nodeText(n)),
				URL:      normalizeDuckDuckGoURL(attr(n, "href")),
				Provider: ProviderDuckDuckGo,
			}
			items = append(items, item)
			current = &items[len(items)-1]
		}
		if current != nil && n.Type == html.ElementNode && hasClass(n, "result__snippet") {
			current.Snippet = normalizeWhitespace(nodeText(n))
		}
		for child := n.FirstChild; child != nil; child = child.NextSibling {
			walk(child)
		}
	}
	walk(root)
	return items
}

func normalizeDuckDuckGoURL(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		return raw
	}
	if parsed.Hostname() == "duckduckgo.com" && strings.HasPrefix(parsed.Path, "/l/") {
		if target := parsed.Query().Get("uddg"); target != "" {
			if decoded, err := url.QueryUnescape(target); err == nil {
				return decoded
			}
			return target
		}
	}
	if parsed.Scheme == "" && strings.HasPrefix(raw, "//") {
		return "https:" + raw
	}
	return raw
}

func hasClass(n *html.Node, className string) bool {
	classes := strings.Fields(attr(n, "class"))
	for _, class := range classes {
		if class == className {
			return true
		}
	}
	return false
}

func attr(n *html.Node, name string) string {
	for _, attr := range n.Attr {
		if attr.Key == name {
			return attr.Val
		}
	}
	return ""
}
