package paragraph

import (
	"context"
	"strings"
	"unicode/utf8"

	einomodel "github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/raiki02/video-extractor/internal/appconfig"
)

const (
	defaultChunkRunes  = 8000
	paragraphSystemMsg = `你是专业的中文转写稿编辑。你的任务是只对转写文本进行自然段划分和轻微格式整理。

要求：
1. 保留原文语义，不总结、不扩写、不改写事实。
2. 修正明显的断句和空白、错别字问题。
3. 按话题、语义停顿和上下文划分段落。
4. 段落之间使用一个空行分隔。
5. 不要添加标题、列表、Markdown 标记或解释。`
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

	chunks := splitByRunes(rawText, defaultChunkRunes)
	formatted := make([]string, 0, len(chunks))

	for _, chunk := range chunks {
		resp, err := cm.Generate(ctx, []*schema.Message{
			schema.SystemMessage(paragraphSystemMsg),
			schema.UserMessage("请为下面的转写文本划分自然段，只返回处理后的正文：\n\n" + chunk),
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

func splitByRunes(text string, limit int) []string {
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
