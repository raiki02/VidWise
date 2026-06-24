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
	Server       ServerConfig       `yaml:"server"`
	Download     DownloadConfig     `yaml:"download"`
	ASR          ASRConfig          `yaml:"asr"`
	VideoSummary VideoSummaryConfig `yaml:"video_summary"`
	LLM          LLMConfig          `yaml:"llm"`
	MySQL        MySQLConfig        `yaml:"mysql"`
	Qdrant       QdrantConfig       `yaml:"qdrant"`
	Embedding    EmbeddingConfig    `yaml:"embedding"`
	Rerank       RerankConfig       `yaml:"rerank"`
	MCP          MCPConfig          `yaml:"mcp"`
}

type ServerConfig struct {
	Addr string `yaml:"addr"`
}

type DownloadConfig struct {
	CookiesPath string `yaml:"cookies_path"`
}

type ASRConfig struct {
	BaseURL    string              `yaml:"base_url"`
	Timeout    string              `yaml:"timeout"`
	Language   string              `yaml:"language"`
	Model      ASRModelConfig      `yaml:"model"`
	Transcribe ASRTranscribeConfig `yaml:"transcribe"`
}

type ASRModelConfig struct {
	Provider    string `yaml:"provider"`
	Name        string `yaml:"name"`
	Device      string `yaml:"device"`
	TorchDType  string `yaml:"torch_dtype"`
	ComputeType string `yaml:"compute_type"`
	CPUThreads  int    `yaml:"cpu_threads"`
	Workers     int    `yaml:"workers"`
}

type ASRTranscribeConfig struct {
	BeamSize      int    `yaml:"beam_size"`
	VADFilter     *bool  `yaml:"vad_filter"`
	InitialPrompt string `yaml:"initial_prompt"`
}

type VideoSummaryConfig struct {
	BaseURL   string                      `yaml:"base_url"`
	Timeout   string                      `yaml:"timeout"`
	Model     VideoSummaryModelConfig     `yaml:"model"`
	Summarize VideoSummarySummarizeConfig `yaml:"summarize"`
}

type VideoSummaryModelConfig struct {
	Name     string `yaml:"name"`
	Provider string `yaml:"provider"`
	Device   string `yaml:"device"`
	DType    string `yaml:"dtype"`
	Compile  bool   `yaml:"compile"`
}

type VideoSummarySummarizeConfig struct {
	MaxNewTokens int     `yaml:"max_new_tokens"`
	Prompt       string  `yaml:"prompt"`
	DoSample     *bool   `yaml:"do_sample"`
	Temperature  float32 `yaml:"temperature"`
	TopP         float32 `yaml:"top_p"`
}

type LLMConfig struct {
	Enabled *bool `yaml:"enabled"`
	// FallbackToRawOnError controls whether the service should return the raw ASR
	// transcript when the LLM paragraph formatter is unavailable (misconfigured,
	// network errors, provider downtime, etc.).
	FallbackToRawOnError *bool        `yaml:"fallback_to_raw_on_error"`
	Provider             string       `yaml:"provider"`
	BaseURL              string       `yaml:"base_url"`
	APIKey               string       `yaml:"api_key"`
	APIKeyEnv            string       `yaml:"api_key_env"`
	Path                 string       `yaml:"path"`
	Model                string       `yaml:"model"`
	Timeout              string       `yaml:"timeout"`
	Temperature          float32      `yaml:"temperature"`
	MaxTokens            int          `yaml:"max_tokens"`
	KeepAlive            string       `yaml:"keep_alive"`
	Prompt               PromptConfig `yaml:"prompt"`
	ChunkRunes           int          `yaml:"chunk_runes"`
	// TwoStep enables a two-pass pipeline:
	//   Step 1: per-chunk typo fix + traditional→simplified conversion (strict, no merging)
	//   Step 2: semantic paragraph organization of the full step-1 output
	TwoStep         bool         `yaml:"two_step"`
	Step1Prompt     PromptConfig `yaml:"step1_prompt"`
	Step1ChunkRunes int          `yaml:"step1_chunk_runes"`
}

type PromptConfig struct {
	System       string `yaml:"system"`
	UserTemplate string `yaml:"user_template"`
}

type MySQLConfig struct {
	DSN     string `yaml:"dsn"`
	MaxOpen int    `yaml:"max_open"`
	MaxIdle int    `yaml:"max_idle"`
}

type QdrantConfig struct {
	Host       string `yaml:"host"`
	Port       int    `yaml:"port"`
	APIKey     string `yaml:"api_key"`
	UseTLS     bool   `yaml:"use_tls"`
	Collection string `yaml:"collection"`
	VectorDim  int    `yaml:"vector_dim"`
}

type EmbeddingConfig struct {
	BaseURL string `yaml:"base_url"`
	Model   string `yaml:"model"`
	Device  string `yaml:"device"`
	Timeout string `yaml:"timeout"`
}

type RerankConfig struct {
	BaseURL string `yaml:"base_url"`
	Model   string `yaml:"model"`
	TopK    int    `yaml:"top_k"`
	Timeout string `yaml:"timeout"`
}

type MCPConfig struct {
	Enabled bool   `yaml:"enabled"`
	Addr    string `yaml:"addr"`
	Mode    string `yaml:"mode"`
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
	if c.ASR.Model.Provider == "" {
		c.ASR.Model.Provider = "whisper"
	}
	if c.ASR.Model.Name == "" {
		c.ASR.Model.Name = "./models/whisper-small"
	}
	if c.ASR.Model.Device == "" {
		c.ASR.Model.Device = "auto"
	}
	if c.ASR.Model.TorchDType == "" {
		c.ASR.Model.TorchDType = "auto"
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
	if c.VideoSummary.BaseURL == "" {
		c.VideoSummary.BaseURL = "http://localhost:8002"
	}
	if c.VideoSummary.Timeout == "" {
		c.VideoSummary.Timeout = "20m"
	}
	if c.VideoSummary.Model.Name == "" {
		c.VideoSummary.Model.Name = "./models/marlin"
	}
	if c.VideoSummary.Model.Provider == "" {
		c.VideoSummary.Model.Provider = "huggingface"
	}
	if c.VideoSummary.Model.Device == "" {
		c.VideoSummary.Model.Device = "auto"
	}
	if c.VideoSummary.Model.DType == "" {
		c.VideoSummary.Model.DType = "bfloat16"
	}
	if c.VideoSummary.Summarize.MaxNewTokens == 0 {
		c.VideoSummary.Summarize.MaxNewTokens = 2048
	}
	if c.VideoSummary.Summarize.Temperature == 0 {
		c.VideoSummary.Summarize.Temperature = 1.0
	}
	if c.VideoSummary.Summarize.TopP == 0 {
		c.VideoSummary.Summarize.TopP = 1.0
	}
	c.LLM.Provider = strings.ToLower(strings.TrimSpace(c.LLM.Provider))
	if c.LLM.Enabled == nil {
		enabled := true
		c.LLM.Enabled = &enabled
	}
	if c.LLM.FallbackToRawOnError == nil {
		v := true
		c.LLM.FallbackToRawOnError = &v
	}
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
		c.LLM.ChunkRunes = 2000
	}
	if c.LLM.Step1ChunkRunes == 0 {
		c.LLM.Step1ChunkRunes = 800
	}
	if c.LLM.Prompt.System == "" {
		c.LLM.Prompt.System = defaultParagraphSystemPrompt
	}
	if c.LLM.Prompt.UserTemplate == "" {
		c.LLM.Prompt.UserTemplate = defaultParagraphUserTemplate
	}
	if c.LLM.Step1Prompt.System == "" {
		c.LLM.Step1Prompt.System = defaultStep1SystemPrompt
	}
	if c.LLM.Step1Prompt.UserTemplate == "" {
		c.LLM.Step1Prompt.UserTemplate = defaultStep1UserTemplate
	}
	// MySQL defaults
	if c.MySQL.MaxOpen == 0 {
		c.MySQL.MaxOpen = 10
	}
	if c.MySQL.MaxIdle == 0 {
		c.MySQL.MaxIdle = 5
	}
	// Qdrant defaults
	if c.Qdrant.Host == "" {
		c.Qdrant.Host = "localhost"
	}
	if c.Qdrant.Port == 0 {
		c.Qdrant.Port = 6334
	}
	if c.Qdrant.Collection == "" {
		c.Qdrant.Collection = "transcript_chunks"
	}
	if c.Qdrant.VectorDim == 0 {
		c.Qdrant.VectorDim = 1024
	}
	// Embedding defaults
	if c.Embedding.BaseURL == "" {
		c.Embedding.BaseURL = "http://localhost:8003"
	}
	if c.Embedding.Model == "" {
		c.Embedding.Model = "qwen"
	}
	if c.Embedding.Device == "" {
		c.Embedding.Device = "auto"
	}
	if c.Embedding.Timeout == "" {
		c.Embedding.Timeout = "2m"
	}
	// Rerank defaults
	if c.Rerank.BaseURL == "" {
		c.Rerank.BaseURL = "http://localhost:8003"
	}
	if c.Rerank.Model == "" {
		c.Rerank.Model = "qwen"
	}
	if c.Rerank.TopK == 0 {
		c.Rerank.TopK = 3
	}
	if c.Rerank.Timeout == "" {
		c.Rerank.Timeout = "1m"
	}
	// MCP defaults
	if c.MCP.Addr == "" {
		c.MCP.Addr = ":8082"
	}
	if c.MCP.Mode == "" {
		c.MCP.Mode = "sse"
	}
}

func (c Config) validate() error {
	llmEnabled := c.LLM.Enabled == nil || *c.LLM.Enabled
	llmFallback := c.LLM.FallbackToRawOnError == nil || *c.LLM.FallbackToRawOnError
	if llmEnabled {
		// If fallback is enabled, we allow the service to start even when the LLM
		// config is incomplete, and fall back to raw ASR output at runtime.
		if !llmFallback {
			if c.LLM.Model == "" {
				return errors.New("llm.model is required")
			}
			switch c.LLM.Provider {
			case "openai", "ollama", "deepseek":
			default:
				return fmt.Errorf("llm.provider must be one of: openai, ollama, deepseek")
			}
		}
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
	if _, err := c.VideoSummary.TimeoutDuration(); err != nil {
		return fmt.Errorf("invalid video_summary.timeout: %w", err)
	}
	return nil
}

func (c ASRConfig) TimeoutDuration() (time.Duration, error) {
	return time.ParseDuration(c.Timeout)
}

func (c VideoSummaryConfig) TimeoutDuration() (time.Duration, error) {
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

func (c EmbeddingConfig) TimeoutDuration() (time.Duration, error) {
	return time.ParseDuration(c.Timeout)
}

func (c RerankConfig) TimeoutDuration() (time.Duration, error) {
	return time.ParseDuration(c.Timeout)
}

func (c QdrantConfig) Addr() string {
	return fmt.Sprintf("%s:%d", c.Host, c.Port)
}

const defaultParagraphSystemPrompt = `你是专业的中文转写稿编辑。你的任务是只对转写文本进行自然段划分和轻微格式整理。

要求：
1. 保留原文语义，不总结、不扩写、不改写事实。
2. 修正明显的断句和空白、错别字问题。
3. 按话题、语义停顿和上下文划分段落。
4. 段落之间使用一个空行分隔。
5. 不要添加标题、列表、Markdown 标记或解释。`

const defaultParagraphUserTemplate = "请为下面的转写文本划分自然段，只返回处理后的正文：\n\n{{text}}"

const defaultStep1SystemPrompt = `你是中文文本纠错助手。

任务：
  - 修正错别字
  - 繁体中文转简体中文

禁止：
  - 改写句子
  - 调整段落
  - 增加或删除内容

输出：
  - 仅返回纠正后的文本`

const defaultStep1UserTemplate = "{{text}}"
