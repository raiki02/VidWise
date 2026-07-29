package search

import (
	"context"
	"fmt"
	"strings"
	"unicode"

	"golang.org/x/net/html"
)

type BasicExtractor struct{}

func NewBasicExtractor() *BasicExtractor {
	return &BasicExtractor{}
}

func (e *BasicExtractor) Extract(ctx context.Context, docs []Document) ([]Document, error) {
	out := make([]Document, 0, len(docs))
	for _, doc := range docs {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("extract documents: %w", err)
		}
		cleaned, err := e.extractOne(doc)
		if err != nil {
			return nil, err
		}
		if strings.TrimSpace(cleaned.Content) != "" {
			out = append(out, cleaned)
		}
	}
	return out, nil
}

func (e *BasicExtractor) extractOne(doc Document) (Document, error) {
	raw := strings.TrimSpace(doc.html)
	if raw == "" {
		doc.Content = normalizeWhitespace(doc.Content)
		return doc, nil
	}

	root, err := html.Parse(strings.NewReader(raw))
	if err != nil {
		return Document{}, fmt.Errorf("parse html for %q: %w", doc.URL, err)
	}

	title := strings.TrimSpace(extractTitle(root))
	content := strings.TrimSpace(extractVisibleText(root))
	if title != "" {
		doc.Title = title
	}
	doc.Content = normalizeWhitespace(content)
	doc.html = ""
	return doc, nil
}

func extractTitle(root *html.Node) string {
	var walk func(*html.Node) string
	walk = func(n *html.Node) string {
		if n.Type == html.ElementNode && strings.EqualFold(n.Data, "title") {
			return nodeText(n)
		}
		for child := n.FirstChild; child != nil; child = child.NextSibling {
			if title := walk(child); title != "" {
				return title
			}
		}
		return ""
	}
	return normalizeWhitespace(walk(root))
}

func extractVisibleText(root *html.Node) string {
	var parts []string
	var walk func(*html.Node, bool)
	walk = func(n *html.Node, skipped bool) {
		if n.Type == html.ElementNode && shouldDropElement(n) {
			skipped = true
		}
		if skipped {
			return
		}
		if n.Type == html.TextNode {
			text := strings.TrimSpace(n.Data)
			if text != "" {
				parts = append(parts, text)
			}
		}
		for child := n.FirstChild; child != nil; child = child.NextSibling {
			walk(child, skipped)
		}
	}
	walk(root, false)
	return strings.Join(parts, " ")
}

func shouldDropElement(n *html.Node) bool {
	switch strings.ToLower(n.Data) {
	case "head", "title", "meta", "link", "script", "style", "noscript", "nav", "header", "footer", "aside", "form", "button", "svg", "canvas", "iframe":
		return true
	}
	noise := strings.ToLower(attr(n, "class") + " " + attr(n, "id") + " " + attr(n, "role"))
	noiseMarkers := []string{"advert", "ad-", " ad", "ads", "banner", "promo", "cookie", "sidebar", "subscribe", "newsletter", "modal", "popup"}
	for _, marker := range noiseMarkers {
		if strings.Contains(noise, marker) {
			return true
		}
	}
	return false
}

func nodeText(n *html.Node) string {
	var parts []string
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.TextNode {
			parts = append(parts, n.Data)
		}
		for child := n.FirstChild; child != nil; child = child.NextSibling {
			walk(child)
		}
	}
	walk(n)
	return strings.Join(parts, " ")
}

func normalizeWhitespace(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	var sb strings.Builder
	previousSpace := false
	for _, r := range text {
		if unicode.IsSpace(r) {
			if !previousSpace {
				sb.WriteRune(' ')
				previousSpace = true
			}
			continue
		}
		sb.WriteRune(r)
		previousSpace = false
	}
	return strings.TrimSpace(sb.String())
}
