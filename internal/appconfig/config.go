package appconfig

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server  ServerConfig  `yaml:"server"`
	Whisper WhisperConfig `yaml:"whisper"`
	ASR     ASRConfig     `yaml:"asr"`
	LLM     LLMConfig     `yaml:"llm"`
}

type ServerConfig struct {
	Addr string `yaml:"addr"`
}

type ASRConfig struct {
	BaseURL  string `yaml:"base_url"`
	Timeout  string `yaml:"timeout"`
	Language string `yaml:"language"`
}

type WhisperConfig struct {
	ModelPath string `yaml:"model_path"`
}

type LLMConfig struct {
	Provider    string  `yaml:"provider"`
	BaseURL     string  `yaml:"base_url"`
	APIKey      string  `yaml:"api_key"`
	APIKeyEnv   string  `yaml:"api_key_env"`
	Path        string  `yaml:"path"`
	Model       string  `yaml:"model"`
	Timeout     string  `yaml:"timeout"`
	Temperature float32 `yaml:"temperature"`
	MaxTokens   int     `yaml:"max_tokens"`
	KeepAlive   string  `yaml:"keep_alive"`
}

func Load(path string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return Config{}, err
	}

	cfg.applyDefaults()
	if err := cfg.validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func (c *Config) applyDefaults() {
	if c.Server.Addr == "" {
		c.Server.Addr = ":8080"
	}
	if c.ASR.BaseURL == "" {
		c.ASR.BaseURL = "http://localhost:8001"
	}
	if c.ASR.Timeout == "" {
		c.ASR.Timeout = "10m"
	}
	if c.ASR.Language == "" {
		c.ASR.Language = "zh"
	}
	if c.Whisper.ModelPath == "" {
		c.Whisper.ModelPath = "./models/ggml-small.bin"
	}
	c.LLM.Provider = strings.ToLower(strings.TrimSpace(c.LLM.Provider))
	if c.LLM.Provider == "" {
		c.LLM.Provider = "openai"
	}
	if c.LLM.Provider == "ollama" && c.LLM.BaseURL == "" {
		c.LLM.BaseURL = "http://localhost:11434"
	}
	if c.LLM.Timeout == "" {
		c.LLM.Timeout = "2m"
	}
	if c.LLM.Temperature == 0 {
		c.LLM.Temperature = 0.2
	}
	if c.LLM.MaxTokens == 0 {
		c.LLM.MaxTokens = 4096
	}
}

func (c Config) validate() error {
	if c.LLM.Model == "" {
		return errors.New("llm.model is required")
	}
	switch c.LLM.Provider {
	case "openai", "ollama", "deepseek":
	default:
		return fmt.Errorf("llm.provider must be one of: openai, ollama, deepseek")
	}
	if _, err := c.LLM.TimeoutDuration(); err != nil {
		return fmt.Errorf("invalid llm.timeout: %w", err)
	}
	if _, err := c.LLM.KeepAliveDuration(); err != nil {
		return fmt.Errorf("invalid llm.keep_alive: %w", err)
	}
	if _, err := c.ASR.TimeoutDuration(); err != nil {
		return fmt.Errorf("invalid asr.timeout: %w", err)
	}
	return nil
}

func (c ASRConfig) TimeoutDuration() (time.Duration, error) {
	return time.ParseDuration(c.Timeout)
}

func (c LLMConfig) TimeoutDuration() (time.Duration, error) {
	return time.ParseDuration(c.Timeout)
}

func (c LLMConfig) KeepAliveDuration() (*time.Duration, error) {
	if strings.TrimSpace(c.KeepAlive) == "" {
		return nil, nil
	}
	d, err := time.ParseDuration(c.KeepAlive)
	if err != nil {
		return nil, err
	}
	return &d, nil
}

func (c LLMConfig) ResolvedAPIKey() string {
	if c.APIKey != "" {
		return c.APIKey
	}
	if c.APIKeyEnv == "" {
		return ""
	}
	return strings.TrimSpace(os.Getenv(c.APIKeyEnv))
}
