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
	Server    ServerConfig    `yaml:"server"`
	Download  DownloadConfig  `yaml:"download"`
	ASR       ASRConfig       `yaml:"asr"`
	Embedding EmbeddingConfig `yaml:"embedding"`
	LLM       LLMConfig       `yaml:"llm"`
	Qdrant    QdrantConfig    `yaml:"qdrant"`
	RAG       RAGConfig       `yaml:"rag"`
	MySQL     MySQLConfig     `yaml:"mysql"`
	Agent     AgentConfig     `yaml:"agent"`
}

type ServerConfig struct {
	Addr              string `yaml:"addr"`
	ReadHeaderTimeout string `yaml:"read_header_timeout"`
	ReadTimeout       string `yaml:"read_timeout"`
	// WriteTimeout stays 0 so long-running download/transcription requests
	// are not interrupted by the HTTP server.
	WriteTimeout    string `yaml:"write_timeout"`
	IdleTimeout     string `yaml:"idle_timeout"`
	ShutdownTimeout string `yaml:"shutdown_timeout"`
}

type DownloadConfig struct {
	CookiesPath string `yaml:"cookies_path"`
	// WorkDir is the sandbox directory where downloaded audio and transcript
	// files are written. Tools only accept paths inside this directory.
	WorkDir string `yaml:"work_dir"`
}

type ASRConfig struct {
	BaseURL  string `yaml:"base_url"`
	Timeout  string `yaml:"timeout"`
	Language string `yaml:"language"`
	// Model keeps the provider block for the Python ASR service. The Go side
	// only needs the transcribe tuning options below.
	Model      ASRModelConfig      `yaml:"model"`
	Transcribe ASRTranscribeConfig `yaml:"transcribe"`
}

// ASRModelConfig mirrors the config the Python ASR service reads. Go itself
// does not use these fields beyond validation.
type ASRModelConfig struct {
	Provider string `yaml:"provider"`
	Name     string `yaml:"name"`
	Device   string `yaml:"device"`
}

type ASRTranscribeConfig struct {
	BeamSize      int    `yaml:"beam_size"`
	VADFilter     *bool  `yaml:"vad_filter"`
	InitialPrompt string `yaml:"initial_prompt"`
}

type EmbeddingConfig struct {
	BaseURL string `yaml:"base_url"`
	// Provider/Model/Device are read by the Python embedding service. Model is
	// also sent to the service on every /embed request.
	Provider string `yaml:"provider"`
	Model    string `yaml:"model"`
	Device   string `yaml:"device"`
	Timeout  string `yaml:"timeout"`
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

type QdrantConfig struct {
	Host       string `yaml:"host"`
	Port       int    `yaml:"port"`
	APIKey     string `yaml:"api_key"`
	UseTLS     bool   `yaml:"use_tls"`
	Collection string `yaml:"collection"`
	VectorDim  int    `yaml:"vector_dim"`
}

type RAGConfig struct {
	ChunkRunes   int `yaml:"chunk_runes"`
	OverlapRunes int `yaml:"overlap_runes"`
	TopK         int `yaml:"top_k"`
}

type MySQLConfig struct {
	DSN     string `yaml:"dsn"`
	MaxOpen int    `yaml:"max_open"`
	MaxIdle int    `yaml:"max_idle"`
}

type AgentConfig struct {
	MaxStep      int `yaml:"max_step"`
	HistoryLimit int `yaml:"history_limit"`
}

// Load reads the YAML config at path, applies defaults, and validates it.
func Load(path string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("read config %s: %w", path, err)
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse config %s: %w", path, err)
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
	if c.Server.ReadHeaderTimeout == "" {
		c.Server.ReadHeaderTimeout = "5s"
	}
	if c.Server.ReadTimeout == "" {
		c.Server.ReadTimeout = "30s"
	}
	if c.Server.WriteTimeout == "" {
		c.Server.WriteTimeout = "0s"
	}
	if c.Server.IdleTimeout == "" {
		c.Server.IdleTimeout = "120s"
	}
	if c.Server.ShutdownTimeout == "" {
		c.Server.ShutdownTimeout = "10s"
	}
	if c.Download.WorkDir == "" {
		c.Download.WorkDir = ".vidwise/media"
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
	if c.ASR.Transcribe.BeamSize == 0 {
		c.ASR.Transcribe.BeamSize = 5
	}
	if c.Embedding.BaseURL == "" {
		c.Embedding.BaseURL = "http://localhost:8003"
	}
	if c.Embedding.Model == "" {
		c.Embedding.Model = "qwen"
	}
	if c.Embedding.Timeout == "" {
		c.Embedding.Timeout = "2m"
	}
	if c.LLM.Provider == "" {
		c.LLM.Provider = "openai"
	}
	if c.LLM.Timeout == "" {
		c.LLM.Timeout = "2m"
	}
	if c.LLM.Temperature == 0 {
		c.LLM.Temperature = 0.2
	}
	if c.LLM.MaxTokens == 0 {
		c.LLM.MaxTokens = 2048
	}
	if c.Qdrant.Host == "" {
		c.Qdrant.Host = "localhost"
	}
	if c.Qdrant.Port == 0 {
		c.Qdrant.Port = 6334
	}
	if c.Qdrant.Collection == "" {
		c.Qdrant.Collection = "transcript_chunks"
	}
	if c.RAG.ChunkRunes == 0 {
		c.RAG.ChunkRunes = 800
	}
	if c.RAG.OverlapRunes == 0 {
		c.RAG.OverlapRunes = 100
	}
	if c.RAG.TopK == 0 {
		c.RAG.TopK = 5
	}
	if c.MySQL.MaxOpen == 0 {
		c.MySQL.MaxOpen = 10
	}
	if c.MySQL.MaxIdle == 0 {
		c.MySQL.MaxIdle = 5
	}
	if c.Agent.MaxStep == 0 {
		c.Agent.MaxStep = 20
	}
	if c.Agent.HistoryLimit == 0 {
		c.Agent.HistoryLimit = 20
	}
}

func (c Config) validate() error {
	if _, err := c.Server.ReadHeaderTimeoutDuration(); err != nil {
		return fmt.Errorf("invalid server.read_header_timeout: %w", err)
	}
	if _, err := c.Server.ReadTimeoutDuration(); err != nil {
		return fmt.Errorf("invalid server.read_timeout: %w", err)
	}
	if _, err := c.Server.WriteTimeoutDuration(); err != nil {
		return fmt.Errorf("invalid server.write_timeout: %w", err)
	}
	if _, err := c.Server.IdleTimeoutDuration(); err != nil {
		return fmt.Errorf("invalid server.idle_timeout: %w", err)
	}
	if _, err := c.Server.ShutdownTimeoutDuration(); err != nil {
		return fmt.Errorf("invalid server.shutdown_timeout: %w", err)
	}
	if c.LLM.Model == "" {
		return errors.New("llm.model is required")
	}
	switch strings.ToLower(strings.TrimSpace(c.LLM.Provider)) {
	case "openai", "ollama", "deepseek":
	default:
		return errors.New("llm.provider must be one of: openai, ollama, deepseek")
	}
	if c.LLM.Provider == "openai" && c.LLM.ResolvedAPIKey() == "" {
		return errors.New("llm.api_key or llm.api_key_env is required for provider openai")
	}
	if c.LLM.Provider == "deepseek" && c.LLM.ResolvedAPIKey() == "" {
		return errors.New("llm.api_key or llm.api_key_env is required for provider deepseek")
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
	if _, err := c.Embedding.TimeoutDuration(); err != nil {
		return fmt.Errorf("invalid embedding.timeout: %w", err)
	}
	if c.Qdrant.VectorDim <= 0 {
		return errors.New("qdrant.vector_dim must be greater than 0")
	}
	if c.RAG.ChunkRunes <= 0 {
		return errors.New("rag.chunk_runes must be greater than 0")
	}
	if c.RAG.OverlapRunes < 0 || c.RAG.OverlapRunes >= c.RAG.ChunkRunes {
		return errors.New("rag.overlap_runes must be >= 0 and < rag.chunk_runes")
	}
	if c.RAG.TopK <= 0 {
		return errors.New("rag.top_k must be greater than 0")
	}
	if c.Agent.MaxStep <= 0 {
		return errors.New("agent.max_step must be greater than 0")
	}
	return nil
}

func (c ServerConfig) ReadHeaderTimeoutDuration() (time.Duration, error) {
	return serverDuration(c.ReadHeaderTimeout, "5s", false)
}

func (c ServerConfig) ReadTimeoutDuration() (time.Duration, error) {
	return serverDuration(c.ReadTimeout, "30s", true)
}

func (c ServerConfig) WriteTimeoutDuration() (time.Duration, error) {
	return serverDuration(c.WriteTimeout, "0s", true)
}

func (c ServerConfig) IdleTimeoutDuration() (time.Duration, error) {
	return serverDuration(c.IdleTimeout, "120s", false)
}

func (c ServerConfig) ShutdownTimeoutDuration() (time.Duration, error) {
	return serverDuration(c.ShutdownTimeout, "10s", false)
}

func (c ASRConfig) TimeoutDuration() (time.Duration, error) {
	return time.ParseDuration(c.Timeout)
}

// TimeoutDurationOr parses the configured timeout, falling back to def on
// parse errors (e.g. empty value after zero-value construction).
func (c ASRConfig) TimeoutDurationOr(def time.Duration) time.Duration {
	d, err := c.TimeoutDuration()
	if err != nil {
		return def
	}
	return d
}

func (c EmbeddingConfig) TimeoutDuration() (time.Duration, error) {
	return time.ParseDuration(c.Timeout)
}

// TimeoutDurationOr parses the configured timeout, falling back to def on
// parse errors.
func (c EmbeddingConfig) TimeoutDurationOr(def time.Duration) time.Duration {
	d, err := c.TimeoutDuration()
	if err != nil {
		return def
	}
	return d
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

func (c QdrantConfig) Addr() string {
	return fmt.Sprintf("%s:%d", c.Host, c.Port)
}

func serverDuration(raw, fallback string, allowZero bool) (time.Duration, error) {
	value := strings.TrimSpace(raw)
	if value == "" {
		value = fallback
	}
	d, err := time.ParseDuration(value)
	if err != nil {
		return 0, err
	}
	if d < 0 {
		return 0, errors.New("must be greater than or equal to 0")
	}
	if !allowZero && d == 0 {
		return 0, errors.New("must be greater than 0")
	}
	return d, nil
}
