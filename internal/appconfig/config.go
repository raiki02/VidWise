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
	RAG          RAGConfig          `yaml:"rag"`
	Embedding    EmbeddingConfig    `yaml:"embedding"`
	Rerank       RerankConfig       `yaml:"rerank"`
	MCP          MCPConfig          `yaml:"mcp"`
	Upload       UploadConfig       `yaml:"upload"`
	Task         TaskConfig         `yaml:"task"`
}

type ServerConfig struct {
	Addr              string `yaml:"addr"`
	ReadHeaderTimeout string `yaml:"read_header_timeout"`
	ReadTimeout       string `yaml:"read_timeout"`
	WriteTimeout      string `yaml:"write_timeout"`
	IdleTimeout       string `yaml:"idle_timeout"`
	ShutdownTimeout   string `yaml:"shutdown_timeout"`
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
	Provider                  string `yaml:"provider"`
	Name                      string `yaml:"name"`
	Device                    string `yaml:"device"`
	TorchDType                string `yaml:"torch_dtype"`
	ComputeType               string `yaml:"compute_type"`
	CPUThreads                int    `yaml:"cpu_threads"`
	Workers                   int    `yaml:"workers"`
	APIBaseURL                string `yaml:"api_base_url"`
	APIKey                    string `yaml:"api_key"`
	APIKeyEnv                 string `yaml:"api_key_env"`
	APITimeoutSeconds         int    `yaml:"api_timeout_seconds"`
	MaxFileBytes              int64  `yaml:"max_file_bytes"`
	XFYunAPIBaseURL           string `yaml:"xfyun_api_base_url"`
	XFYunAppID                string `yaml:"xfyun_app_id"`
	XFYunAppIDEnv             string `yaml:"xfyun_app_id_env"`
	XFYunAccessKeyID          string `yaml:"xfyun_access_key_id"`
	XFYunAccessKeyIDEnv       string `yaml:"xfyun_access_key_id_env"`
	XFYunAccessKeySecret      string `yaml:"xfyun_access_key_secret"`
	XFYunAccessKeySecretEnv   string `yaml:"xfyun_access_key_secret_env"`
	XFYunLanguage             string `yaml:"xfyun_language"`
	XFYunResultType           string `yaml:"xfyun_result_type"`
	XFYunAPITimeoutSeconds    int    `yaml:"xfyun_api_timeout_seconds"`
	XFYunPollIntervalSeconds  int    `yaml:"xfyun_poll_interval_seconds"`
	XFYunMaxPollSeconds       int    `yaml:"xfyun_max_poll_seconds"`
	XFYunMaxFileBytes         int64  `yaml:"xfyun_max_file_bytes"`
	XFYunDurationCheckDisable bool   `yaml:"xfyun_duration_check_disable"`
	BaiduTokenURL             string `yaml:"baidu_token_url"`
	BaiduAPIBaseURL           string `yaml:"baidu_api_base_url"`
	BaiduAPIKey               string `yaml:"baidu_api_key"`
	BaiduAPIKeyEnv            string `yaml:"baidu_api_key_env"`
	BaiduSecretKey            string `yaml:"baidu_secret_key"`
	BaiduSecretKeyEnv         string `yaml:"baidu_secret_key_env"`
	BaiduCUID                 string `yaml:"baidu_cuid"`
	BaiduDevPID               int    `yaml:"baidu_dev_pid"`
	BaiduRate                 int    `yaml:"baidu_rate"`
	BaiduChannel              int    `yaml:"baidu_channel"`
	BaiduAPITimeoutSeconds    int    `yaml:"baidu_api_timeout_seconds"`
	BaiduChunkSeconds         int    `yaml:"baidu_chunk_seconds"`
	BaiduMaxChunkBytes        int64  `yaml:"baidu_max_chunk_bytes"`
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

type RAGConfig struct {
	Retrieval RAGRetrievalConfig `yaml:"retrieval"`
	Context   RAGContextConfig   `yaml:"context"`
}

type RAGRetrievalConfig struct {
	SearchTopK int     `yaml:"search_top_k"`
	TopK       int     `yaml:"top_k"`
	MinScore   float64 `yaml:"min_score"`
}

type RAGContextConfig struct {
	MaxRunes int `yaml:"max_runes"`
}

type EmbeddingConfig struct {
	BaseURL           string `yaml:"base_url"`
	Provider          string `yaml:"provider"`
	Model             string `yaml:"model"`
	Device            string `yaml:"device"`
	Timeout           string `yaml:"timeout"`
	APIBaseURL        string `yaml:"api_base_url"`
	APIKey            string `yaml:"api_key"`
	APIKeyEnv         string `yaml:"api_key_env"`
	APITimeoutSeconds int    `yaml:"api_timeout_seconds"`
	Dimensions        int    `yaml:"dimensions"`
	BatchSize         int    `yaml:"batch_size"`
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

type UploadConfig struct {
	ChunkRunes   int   `yaml:"chunk_runes"`
	OverlapRunes int   `yaml:"overlap_runes"`
	MaxFileBytes int64 `yaml:"max_file_bytes"` // 0 = no limit
}

type TaskConfig struct {
	MaxTracked int    `yaml:"max_tracked"`
	RetainFor  string `yaml:"retain_for"`
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
	if c.ASR.Model.APIBaseURL == "" {
		c.ASR.Model.APIBaseURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
	}
	if c.ASR.Model.APIKeyEnv == "" {
		c.ASR.Model.APIKeyEnv = "DASHSCOPE_API_KEY"
	}
	if c.ASR.Model.APITimeoutSeconds == 0 {
		c.ASR.Model.APITimeoutSeconds = 300
	}
	if c.ASR.Model.MaxFileBytes == 0 {
		c.ASR.Model.MaxFileBytes = 7_500_000
	}
	if c.ASR.Model.XFYunAPIBaseURL == "" {
		c.ASR.Model.XFYunAPIBaseURL = "https://office-api-ist-dx.iflyaisol.com/v2"
	}
	if c.ASR.Model.XFYunAppIDEnv == "" {
		c.ASR.Model.XFYunAppIDEnv = "XFYUN_APP_ID"
	}
	if c.ASR.Model.XFYunAccessKeyIDEnv == "" {
		c.ASR.Model.XFYunAccessKeyIDEnv = "XFYUN_API_KEY"
	}
	if c.ASR.Model.XFYunAccessKeySecretEnv == "" {
		c.ASR.Model.XFYunAccessKeySecretEnv = "XFYUN_API_SECRET"
	}
	if c.ASR.Model.XFYunLanguage == "" {
		c.ASR.Model.XFYunLanguage = "autodialect"
	}
	if c.ASR.Model.XFYunResultType == "" {
		c.ASR.Model.XFYunResultType = "transfer"
	}
	if c.ASR.Model.XFYunAPITimeoutSeconds == 0 {
		c.ASR.Model.XFYunAPITimeoutSeconds = 300
	}
	if c.ASR.Model.XFYunPollIntervalSeconds == 0 {
		c.ASR.Model.XFYunPollIntervalSeconds = 3
	}
	if c.ASR.Model.XFYunMaxPollSeconds == 0 {
		c.ASR.Model.XFYunMaxPollSeconds = 600
	}
	if c.ASR.Model.XFYunMaxFileBytes == 0 {
		c.ASR.Model.XFYunMaxFileBytes = 500_000_000
	}
	if c.ASR.Model.BaiduTokenURL == "" {
		c.ASR.Model.BaiduTokenURL = "https://aip.baidubce.com/oauth/2.0/token"
	}
	if c.ASR.Model.BaiduAPIBaseURL == "" {
		c.ASR.Model.BaiduAPIBaseURL = "https://vop.baidu.com"
	}
	if c.ASR.Model.BaiduAPIKeyEnv == "" {
		c.ASR.Model.BaiduAPIKeyEnv = "BAIDU_ASR_API_KEY"
	}
	if c.ASR.Model.BaiduSecretKeyEnv == "" {
		c.ASR.Model.BaiduSecretKeyEnv = "BAIDU_ASR_SECRET_KEY"
	}
	if c.ASR.Model.BaiduCUID == "" {
		c.ASR.Model.BaiduCUID = "vidwise"
	}
	if c.ASR.Model.BaiduDevPID == 0 {
		c.ASR.Model.BaiduDevPID = 1537
	}
	if c.ASR.Model.BaiduRate == 0 {
		c.ASR.Model.BaiduRate = 16000
	}
	if c.ASR.Model.BaiduChannel == 0 {
		c.ASR.Model.BaiduChannel = 1
	}
	if c.ASR.Model.BaiduAPITimeoutSeconds == 0 {
		c.ASR.Model.BaiduAPITimeoutSeconds = 60
	}
	if c.ASR.Model.BaiduChunkSeconds == 0 {
		c.ASR.Model.BaiduChunkSeconds = 55
	}
	if c.ASR.Model.BaiduMaxChunkBytes == 0 {
		c.ASR.Model.BaiduMaxChunkBytes = 10_000_000
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
	// RAG retrieval defaults
	if c.RAG.Retrieval.SearchTopK == 0 {
		c.RAG.Retrieval.SearchTopK = 20
	}
	if c.RAG.Retrieval.TopK == 0 {
		c.RAG.Retrieval.TopK = 8
	}
	if c.RAG.Context.MaxRunes == 0 {
		c.RAG.Context.MaxRunes = 12000
	}
	// Embedding defaults
	if c.Embedding.BaseURL == "" {
		c.Embedding.BaseURL = "http://localhost:8003"
	}
	if c.Embedding.Provider == "" {
		c.Embedding.Provider = "local"
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
	if c.Embedding.APIBaseURL == "" {
		c.Embedding.APIBaseURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
	}
	if c.Embedding.APIKeyEnv == "" {
		c.Embedding.APIKeyEnv = "DASHSCOPE_API_KEY"
	}
	if c.Embedding.APITimeoutSeconds == 0 {
		c.Embedding.APITimeoutSeconds = 120
	}
	if c.Embedding.BatchSize == 0 {
		c.Embedding.BatchSize = 10
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
	// Upload defaults
	if c.Upload.ChunkRunes == 0 {
		c.Upload.ChunkRunes = 1024
	}
	if c.Upload.OverlapRunes == 0 {
		c.Upload.OverlapRunes = 128
	}
	if c.Upload.MaxFileBytes == 0 {
		c.Upload.MaxFileBytes = 10 * 1024 * 1024 // 10 MB
	}
	// Task tracker defaults
	if c.Task.MaxTracked == 0 {
		c.Task.MaxTracked = 1000
	}
	if c.Task.RetainFor == "" {
		c.Task.RetainFor = "24h"
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
	switch strings.ToLower(strings.TrimSpace(c.ASR.Model.Provider)) {
	case "whisper", "faster-whisper", "faster_whisper", "aliyun", "dashscope", "xfyun", "iflytek", "baidu", "baiducloud", "baidu-cloud":
	default:
		return fmt.Errorf("asr.model.provider must be one of: whisper, faster-whisper, aliyun, dashscope, xfyun, iflytek, baidu")
	}
	if _, err := c.VideoSummary.TimeoutDuration(); err != nil {
		return fmt.Errorf("invalid video_summary.timeout: %w", err)
	}
	switch strings.ToLower(strings.TrimSpace(c.Embedding.Provider)) {
	case "local", "sentence-transformers", "sentence_transformers", "huggingface", "hf", "aliyun", "dashscope":
	default:
		return fmt.Errorf("embedding.provider must be one of: local, sentence-transformers, huggingface, aliyun, dashscope")
	}
	if c.Embedding.BatchSize <= 0 {
		return errors.New("embedding.batch_size must be greater than 0")
	}
	if c.RAG.Retrieval.SearchTopK <= 0 {
		return errors.New("rag.retrieval.search_top_k must be greater than 0")
	}
	if c.RAG.Retrieval.TopK <= 0 {
		return errors.New("rag.retrieval.top_k must be greater than 0")
	}
	if c.RAG.Retrieval.TopK > c.RAG.Retrieval.SearchTopK {
		return errors.New("rag.retrieval.top_k must be less than or equal to rag.retrieval.search_top_k")
	}
	if c.RAG.Retrieval.MinScore < 0 {
		return errors.New("rag.retrieval.min_score must be greater than or equal to 0")
	}
	if c.RAG.Context.MaxRunes <= 0 {
		return errors.New("rag.context.max_runes must be greater than 0")
	}
	if c.Task.MaxTracked <= 0 {
		return errors.New("task.max_tracked must be greater than 0")
	}
	if _, err := c.Task.RetentionDuration(); err != nil {
		return fmt.Errorf("invalid task.retain_for: %w", err)
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

func (c TaskConfig) RetentionDuration() (time.Duration, error) {
	retainFor := strings.TrimSpace(c.RetainFor)
	if retainFor == "" {
		retainFor = "24h"
	}
	d, err := time.ParseDuration(retainFor)
	if err != nil {
		return 0, err
	}
	if d <= 0 {
		return 0, errors.New("must be greater than 0")
	}
	return d, nil
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
