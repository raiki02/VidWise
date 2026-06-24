package tool

import (
	"context"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/raiki02/video-extractor/internal/appconfig"
	"github.com/raiki02/video-extractor/internal/paragraph"
)

// TextFormatInput is the input for the LLM paragraph formatting tool.
type TextFormatInput struct {
	RawText string `json:"raw_text" jsonschema:"required" jsonschema_description:"The raw ASR transcript text to format."`
}

type TextFormatOutput struct {
	FormattedText string `json:"formatted_text"`
}

func NewTextFormatTool(cfg appconfig.LLMConfig) (tool.InvokableTool, *Wrapper, error) {
	inner, err := utils.InferTool(
		"format_transcript",
		"Format raw ASR transcript text using an LLM. Fixes typos, converts traditional to simplified Chinese, and organizes into semantic paragraphs.",
		func(ctx context.Context, input TextFormatInput) (TextFormatOutput, error) {
			formatted, err := paragraph.FormatTextWithFallback(ctx, input.RawText, cfg)
			if err != nil {
				return TextFormatOutput{}, err
			}
			return TextFormatOutput{FormattedText: formatted}, nil
		},
	)
	if err != nil {
		return nil, nil, err
	}
	wrapper := NewWrapper(inner, WrapperConfig{Name: "format_transcript", Timeout: 0})
	return inner, wrapper, nil
}
