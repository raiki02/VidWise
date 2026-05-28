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
	BaseURL    string              `yaml:"base_url"`
	Timeout    string              `yaml:"timeout"`
	Language   string              `yaml:"language"`
	Model      ASRModelConfig      `yaml:"model"`
	Transcribe ASRTranscribeConfig `yaml:"transcribe"`
}

type WhisperConfig struct {
	ModelPath string `yaml:"model_path"`
	Language  string `yaml:"language"`
	Prompt    string `yaml:"prompt"`
}

type ASRModelConfig struct {
	Name        string `yaml:"name"`
	Device      string `yaml:"device"`
	ComputeType string `yaml:"compute_type"`
	CPUThreads  int    `yaml:"cpu_threads"`
	Workers     int    `yaml:"workers"`
}

type ASRTranscribeConfig struct {
	BeamSize      int    `yaml:"beam_size"`
	VADFilter     *bool  `yaml:"vad_filter"`
	InitialPrompt string `yaml:"initial_prompt"`
}

type LLMConfig struct {
	Provider    string       `yaml:"provider"`
	BaseURL     string       `yaml:"base_url"`
	APIKey      string       `yaml:"api_key"`
	APIKeyEnv   string       `yaml:"api_key_env"`
	Path        string       `yaml:"path"`
	Model       string       `yaml:"model"`
	Timeout     string       `yaml:"timeout"`
	Temperature float32      `yaml:"temperature"`
	MaxTokens   int          `yaml:"max_tokens"`
	KeepAlive   string       `yaml:"keep_alive"`
	Prompt      PromptConfig `yaml:"prompt"`
	ChunkRunes  int          `yaml:"chunk_runes"`
}

type PromptConfig struct {
	System       string `yaml:"system"`
	UserTemplate string `yaml:"user_template"`
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
	if c.ASR.Model.Name == "" {
		c.ASR.Model.Name = "small"
	}
	if c.ASR.Model.Device == "" {
		c.ASR.Model.Device = "auto"
	}
	if c.ASR.Model.ComputeType == "" {
		c.ASR.Model.ComputeType = "default"
	}
	if c.ASR.Model.Workers == 0 {
		c.ASR.Model.Workers = 1
	}
	if c.ASR.Transcribe.BeamSize == 0 {
		c.ASR.Transcribe.BeamSize = 5
	}
	if c.Whisper.ModelPath == "" {
		c.Whisper.ModelPath = "./models/ggml-small.bin"
	}
	if c.Whisper.Language == "" {
		c.Whisper.Language = c.ASR.Language
	}
	if c.Whisper.Prompt == "" {
		c.Whisper.Prompt = c.ASR.Transcribe.InitialPrompt
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
	if c.LLM.ChunkRunes == 0 {
		c.LLM.ChunkRunes = 8000
	}
	if c.LLM.Prompt.System == "" {
		c.LLM.Prompt.System = defaultParagraphSystemPrompt
	}
	if c.LLM.Prompt.UserTemplate == "" {
		c.LLM.Prompt.UserTemplate = defaultParagraphUserTemplate
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

const defaultParagraphSystemPrompt = `你是专业的中文转写稿编辑。你的任务是只对转写文本进行自然段划分和轻微格式整理。

要求：
1. 保留原文语义，不总结、不扩写、不改写事实。
2. 修正明显的断句和空白、错别字问题。
3. 按话题、语义停顿和上下文划分段落。
4. 段落之间使用一个空行分隔。
5. 不要添加标题、列表、Markdown 标记或解释。`

const defaultParagraphUserTemplate = "请为下面的转写文本划分自然段，只返回处理后的正文：\n\n{{text}}"
