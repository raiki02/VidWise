package paragraph

import (
	"context"
	"strings"
	"unicode/utf8"

	einomodel "github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/raiki02/video-extractor/internal/appconfig"
)

func FormatText(ctx context.Context, rawText string, cfg appconfig.LLMConfig) (string, error) {
	rawText = strings.TrimSpace(rawText)
	if rawText == "" {
		return "", nil
	}

	cm, err := NewChatModel(ctx, cfg)
	if err != nil {
		return "", err
	}

	chunks := splitByRunes(rawText, cfg.ChunkRunes)
	formatted := make([]string, 0, len(chunks))

	for _, chunk := range chunks {
		resp, err := cm.Generate(ctx, []*schema.Message{
			schema.SystemMessage(cfg.Prompt.System),
			schema.UserMessage(renderUserPrompt(cfg.Prompt.UserTemplate, chunk)),
		}, einomodel.WithTemperature(cfg.Temperature), einomodel.WithMaxTokens(cfg.MaxTokens))
		if err != nil {
			return "", err
		}

		content := strings.TrimSpace(resp.Content)
		if content != "" {
			formatted = append(formatted, content)
		}
	}

	return strings.Join(formatted, "\n\n"), nil
}

func renderUserPrompt(template, text string) string {
	if template == "" {
		return text
	}
	return strings.ReplaceAll(template, "{{text}}", text)
}

func splitByRunes(text string, limit int) []string {
	if limit <= 0 {
		limit = utf8.RuneCountInString(text)
	}
	if utf8.RuneCountInString(text) <= limit {
		return []string{text}
	}

	var chunks []string
	runes := []rune(text)
	for start := 0; start < len(runes); start += limit {
		end := start + limit
		if end > len(runes) {
			end = len(runes)
		}
		chunk := strings.TrimSpace(string(runes[start:end]))
		if chunk != "" {
			chunks = append(chunks, chunk)
		}
	}
	return chunks
}
