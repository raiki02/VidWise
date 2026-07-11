package rag

import (
	"fmt"
	"strconv"
	"strings"

	mdast "github.com/gomarkdown/markdown/ast"
	"github.com/gomarkdown/markdown/parser"
	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
)

const (
	MarkdownContentType = "markdown"
	PlainContentType    = "plain"

	metadataHeaderPrefix = "header_"
)

// ParseMarkdownDocuments parses a Markdown file into section documents.
//
// It follows the same shape as LangChain's Markdown header splitters: headings
// become metadata, and each heading section is merged with its paragraphs before
// the downstream chunking pass.
func ParseMarkdownDocuments(markdownText string, baseMetadata map[string]string) []Document {
	markdownText = CleanSourceText(markdownText, ContentFormatMarkdown)
	if markdownText == "" {
		return nil
	}

	metadata := cloneMetadata(baseMetadata)
	metadata[qdrantclient.FieldContentType] = MarkdownContentType

	extensions := parser.CommonExtensions | parser.AutoHeadingIDs | parser.NoEmptyLineBeforeBlock
	doc := parser.NewWithExtensions(extensions).Parse([]byte(markdownText))

	var docs []Document
	var headingStack [6]string
	section := newMarkdownSection(metadata, headingStack, 0)

	flush := func() {
		if !section.hasBody() {
			return
		}
		content := section.content()
		if content == "" {
			return
		}
		meta := cloneMetadata(section.metadata)
		meta[qdrantclient.FieldSectionIndex] = strconv.Itoa(len(docs))
		if meta[qdrantclient.FieldDocumentTitle] == "" {
			meta[qdrantclient.FieldDocumentTitle] = firstNonEmpty(meta[qdrantclient.FieldHeader1], meta[qdrantclient.FieldSourceName])
		}
		docs = append(docs, Document{
			PageContent: content,
			Metadata:    meta,
		})
	}

	for _, child := range doc.GetChildren() {
		heading, ok := child.(*mdast.Heading)
		if ok {
			flush()
			level := normalizeHeadingLevel(heading.Level)
			title := strings.TrimSpace(renderInlineText(heading))
			if title == "" {
				title = fmt.Sprintf("Untitled %d", len(docs)+1)
			}
			headingStack[level-1] = title
			for i := level; i < len(headingStack); i++ {
				headingStack[i] = ""
			}
			section = newMarkdownSection(metadata, headingStack, level)
			continue
		}

		block := strings.TrimSpace(renderMarkdownBlock(child))
		if block == "" {
			continue
		}
		section.blocks = append(section.blocks, block)
	}
	flush()

	return docs
}

type markdownSection struct {
	metadata map[string]string
	prefix   []string
	blocks   []string
}

func newMarkdownSection(base map[string]string, headingStack [6]string, currentLevel int) markdownSection {
	meta := cloneMetadata(base)
	if currentLevel > 0 {
		meta[qdrantclient.FieldHeadingLevel] = strconv.Itoa(currentLevel)
	}

	pathParts := make([]string, 0, currentLevel)
	prefix := make([]string, 0, currentLevel)
	for i := 0; i < currentLevel && i < len(headingStack); i++ {
		title := strings.TrimSpace(headingStack[i])
		if title == "" {
			continue
		}
		meta[fmt.Sprintf("%s%d", metadataHeaderPrefix, i+1)] = title
		pathParts = append(pathParts, title)
		prefix = append(prefix, fmt.Sprintf("%s %s", strings.Repeat("#", i+1), title))
	}

	if len(pathParts) > 0 {
		meta[qdrantclient.FieldHeadingPath] = strings.Join(pathParts, " > ")
		meta[qdrantclient.FieldDocumentTitle] = pathParts[0]
	}

	return markdownSection{
		metadata: meta,
		prefix:   prefix,
	}
}

func (s markdownSection) content() string {
	parts := make([]string, 0, len(s.prefix)+len(s.blocks))
	for _, block := range s.prefix {
		if block = strings.TrimSpace(block); block != "" {
			parts = append(parts, block)
		}
	}
	for _, block := range s.blocks {
		if block = strings.TrimSpace(block); block != "" {
			parts = append(parts, block)
		}
	}
	return strings.TrimSpace(strings.Join(parts, "\n\n"))
}

func (s markdownSection) hasBody() bool {
	for _, block := range s.blocks {
		if strings.TrimSpace(block) != "" {
			return true
		}
	}
	return false
}

func normalizeHeadingLevel(level int) int {
	if level < 1 {
		return 1
	}
	if level > 6 {
		return 6
	}
	return level
}

func renderMarkdownBlock(node mdast.Node) string {
	switch n := node.(type) {
	case *mdast.Heading:
		level := normalizeHeadingLevel(n.Level)
		return fmt.Sprintf("%s %s", strings.Repeat("#", level), strings.TrimSpace(renderInlineText(n)))
	case *mdast.Paragraph:
		return strings.TrimSpace(renderInlineText(n))
	case *mdast.CodeBlock:
		info := strings.TrimSpace(string(n.Info))
		content := strings.TrimRight(string(firstNonEmptyBytes(n.Literal, n.Content)), "\n")
		if info != "" {
			return fmt.Sprintf("```%s\n%s\n```", info, content)
		}
		return fmt.Sprintf("```\n%s\n```", content)
	case *mdast.HTMLBlock:
		return strings.TrimSpace(string(firstNonEmptyBytes(n.Literal, n.Content)))
	case *mdast.HorizontalRule:
		return "---"
	case *mdast.List:
		return renderList(n)
	case *mdast.ListItem:
		return renderChildren(n, "\n")
	case *mdast.BlockQuote:
		return prefixLines("> ", renderChildren(n, "\n\n"))
	case *mdast.Table:
		return renderChildren(n, "\n")
	case *mdast.TableHeader, *mdast.TableBody, *mdast.TableFooter:
		return renderChildren(n, "\n")
	case *mdast.TableRow:
		return renderChildren(n, " | ")
	case *mdast.TableCell:
		return strings.TrimSpace(renderInlineText(n))
	case *mdast.Caption, *mdast.CaptionFigure:
		return renderChildren(n, "\n")
	default:
		if leaf := node.AsLeaf(); leaf != nil {
			return strings.TrimSpace(string(firstNonEmptyBytes(leaf.Literal, leaf.Content)))
		}
		return renderChildren(node, "\n\n")
	}
}

func renderList(list *mdast.List) string {
	children := list.GetChildren()
	items := make([]string, 0, len(children))
	start := list.Start
	if start <= 0 {
		start = 1
	}
	for i, child := range children {
		text := strings.TrimSpace(renderMarkdownBlock(child))
		if text == "" {
			continue
		}
		marker := "- "
		if list.ListFlags&mdast.ListTypeOrdered != 0 {
			marker = fmt.Sprintf("%d. ", start+i)
		}
		items = append(items, prefixFirstLine(marker, text))
	}
	return strings.Join(items, "\n")
}

func renderInlineText(node mdast.Node) string {
	switch n := node.(type) {
	case *mdast.Text:
		return string(n.Literal)
	case *mdast.Code:
		return "`" + string(n.Literal) + "`"
	case *mdast.HTMLSpan:
		return string(firstNonEmptyBytes(n.Literal, n.Content))
	case *mdast.Softbreak, *mdast.Hardbreak:
		return "\n"
	case *mdast.NonBlockingSpace:
		return " "
	case *mdast.Image:
		alt := strings.TrimSpace(renderInlineChildren(n))
		dest := strings.TrimSpace(string(n.Destination))
		if alt == "" {
			return dest
		}
		return alt
	case *mdast.Link:
		return renderInlineChildren(n)
	case *mdast.Math:
		return strings.TrimSpace(string(n.Literal))
	default:
		if leaf := node.AsLeaf(); leaf != nil {
			return string(firstNonEmptyBytes(leaf.Literal, leaf.Content))
		}
		return renderInlineChildren(node)
	}
}

func renderInlineChildren(node mdast.Node) string {
	children := node.GetChildren()
	if len(children) == 0 {
		return ""
	}
	var b strings.Builder
	for _, child := range children {
		b.WriteString(renderInlineText(child))
	}
	return b.String()
}

func renderChildren(node mdast.Node, separator string) string {
	children := node.GetChildren()
	if len(children) == 0 {
		return ""
	}
	parts := make([]string, 0, len(children))
	for _, child := range children {
		text := strings.TrimSpace(renderMarkdownBlock(child))
		if text == "" {
			text = strings.TrimSpace(renderInlineText(child))
		}
		if text != "" {
			parts = append(parts, text)
		}
	}
	return strings.TrimSpace(strings.Join(parts, separator))
}

func prefixFirstLine(prefix, text string) string {
	lines := strings.Split(text, "\n")
	for i := range lines {
		if i == 0 {
			lines[i] = prefix + lines[i]
			continue
		}
		if strings.TrimSpace(lines[i]) != "" {
			lines[i] = strings.Repeat(" ", len(prefix)) + lines[i]
		}
	}
	return strings.Join(lines, "\n")
}

func prefixLines(prefix, text string) string {
	lines := strings.Split(text, "\n")
	for i := range lines {
		if strings.TrimSpace(lines[i]) != "" {
			lines[i] = prefix + lines[i]
		}
	}
	return strings.Join(lines, "\n")
}

func firstNonEmptyBytes(values ...[]byte) []byte {
	for _, value := range values {
		if len(value) > 0 {
			return value
		}
	}
	return nil
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func normalizeMarkdownNewlines(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	return strings.ReplaceAll(text, "\r", "\n")
}
