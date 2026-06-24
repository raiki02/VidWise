package paragraph

import (
	"context"
	"errors"

	"github.com/cloudwego/eino-ext/components/model/deepseek"
	"github.com/cloudwego/eino-ext/components/model/ollama"
	"github.com/cloudwego/eino-ext/components/model/openai"
	einomodel "github.com/cloudwego/eino/components/model"
	"github.com/raiki02/vidwise/internal/appconfig"
)

func NewChatModel(ctx context.Context, cfg appconfig.LLMConfig) (einomodel.BaseChatModel, error) {
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
		return openai.NewChatModel(ctx, &openai.ChatModelConfig{
			APIKey:              apiKey,
			BaseURL:             cfg.BaseURL,
			Model:               cfg.Model,
			Timeout:             timeout,
			Temperature:         &cfg.Temperature,
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
