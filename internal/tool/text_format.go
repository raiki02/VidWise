package tool

import (
	"context"
	"errors"
	"strings"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/paragraph"
)

// TextFormatInput is the input for the LLM paragraph formatting tool.
type TextFormatInput struct {
	RawText string `json:"raw_text" jsonschema:"required" jsonschema_description:"The raw ASR transcript text to format."`
}

type TextFormatOutput struct {
	FormattedText string `json:"formatted_text"`
	Formatted     bool   `json:"formatted"`
	Status        string `json:"status"`
	Reason        string `json:"reason,omitempty"`
}

func NewTextFormatTool(cfg appconfig.LLMConfig) (tool.InvokableTool, *Wrapper, error) {
	inner, err := utils.InferTool(
		"format_transcript",
		"Format raw ASR transcript text using an LLM. Fixes typos, converts traditional to simplified Chinese, and organizes into semantic paragraphs.",
		func(ctx context.Context, input TextFormatInput) (TextFormatOutput, error) {
			rawText := strings.TrimSpace(input.RawText)
			if rawText == "" {
				return TextFormatOutput{Status: "empty", Reason: "raw text is empty"}, nil
			}
			if reason := unavailableLLMFormatReason(cfg); reason != "" {
				return TextFormatOutput{
					FormattedText: rawText,
					Formatted:     false,
					Status:        "skipped",
					Reason:        reason,
				}, nil
			}
			formatted, err := paragraph.FormatText(ctx, rawText, cfg)
			if err != nil {
				return TextFormatOutput{}, err
			}
			if strings.TrimSpace(formatted) == "" {
				return TextFormatOutput{}, errors.New("formatted text is empty")
			}
			return TextFormatOutput{
				FormattedText: formatted,
				Formatted:     true,
				Status:        "formatted",
			}, nil
		},
	)
	if err != nil {
		return nil, nil, err
	}
	wrapper := NewWrapper(inner, WrapperConfig{Name: "format_transcript", Timeout: 0})
	return inner, wrapper, nil
}

func unavailableLLMFormatReason(cfg appconfig.LLMConfig) string {
	if cfg.Enabled != nil && !*cfg.Enabled {
		return "llm disabled"
	}
	if strings.TrimSpace(cfg.Model) == "" {
		return "llm model is not configured"
	}
	return ""
}
