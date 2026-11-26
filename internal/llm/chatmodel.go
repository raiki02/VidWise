package llm

import (
	"context"
	"errors"

	"github.com/cloudwego/eino-ext/components/model/deepseek"
	"github.com/cloudwego/eino-ext/components/model/ollama"
	"github.com/cloudwego/eino-ext/components/model/openai"
	einomodel "github.com/cloudwego/eino/components/model"
	"github.com/raiki02/vidwise/internal/appconfig"
)

// NewChatModel builds a tool-calling chat model from config. All three
// eino-ext providers (openai, ollama, deepseek) implement WithTools, so the
// returned model can be used as a react agent's ToolCallingModel.
func NewChatModel(ctx context.Context, cfg appconfig.LLMConfig) (einomodel.ToolCallingChatModel, error) {
	timeout, err := cfg.TimeoutDuration()
	if err != nil {
		return nil, err
	}

	switch cfg.Provider {
	case "openai":
		apiKey := cfg.ResolvedAPIKey()
		if apiKey == "" {
			return nil, errors.New("llm.api_key or llm.api_key_env is required for openai")
		}
		temperature := cfg.Temperature
		return openai.NewChatModel(ctx, &openai.ChatModelConfig{
			APIKey:              apiKey,
			BaseURL:             cfg.BaseURL,
			Model:               cfg.Model,
			Timeout:             timeout,
			Temperature:         &temperature,
			MaxCompletionTokens: &cfg.MaxTokens,
		})
	case "ollama":
		keepAlive, err := cfg.KeepAliveDuration()
		if err != nil {
			return nil, err
		}
		thinkFalse := ollama.ThinkValue{Value: false}
		return ollama.NewChatModel(ctx, &ollama.ChatModelConfig{
			BaseURL:   cfg.BaseURL,
			Timeout:   timeout,
			Model:     cfg.Model,
			KeepAlive: keepAlive,
			Options:   &ollama.Options{Temperature: cfg.Temperature},
			Thinking:  &thinkFalse,
		})
	case "deepseek":
		apiKey := cfg.ResolvedAPIKey()
		if apiKey == "" {
			return nil, errors.New("llm.api_key or llm.api_key_env is required for deepseek")
		}
		return deepseek.NewChatModel(ctx, &deepseek.ChatModelConfig{
			APIKey:      apiKey,
			BaseURL:     cfg.BaseURL,
			Path:        cfg.Path,
			Model:       cfg.Model,
			Timeout:     timeout,
			Temperature: cfg.Temperature,
			MaxTokens:   cfg.MaxTokens,
		})
	default:
		return nil, errors.New("unsupported llm.provider")
	}
}
