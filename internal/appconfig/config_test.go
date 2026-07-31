package appconfig

import (
	"strings"
	"testing"
	"time"
)

func TestApplyDefaultsSetsRAGRetrievalDefaults(t *testing.T) {
	var cfg Config
	cfg.applyDefaults()

	if cfg.RAG.Retrieval.SearchTopK != 20 {
		t.Fatalf("SearchTopK = %d, want 20", cfg.RAG.Retrieval.SearchTopK)
	}
	if cfg.RAG.Retrieval.TopK != 8 {
		t.Fatalf("TopK = %d, want 8", cfg.RAG.Retrieval.TopK)
	}
	if cfg.RAG.Retrieval.MinScore != 0 {
		t.Fatalf("MinScore = %f, want 0", cfg.RAG.Retrieval.MinScore)
	}
	if cfg.RAG.Context.MaxRunes != 12000 {
		t.Fatalf("MaxRunes = %d, want 12000", cfg.RAG.Context.MaxRunes)
	}
}

func TestApplyDefaultsSetsSearchDefaults(t *testing.T) {
	var cfg Config
	cfg.applyDefaults()

	if cfg.Search.Provider != "mock" {
		t.Fatalf("Search Provider = %q, want mock", cfg.Search.Provider)
	}
	if strings.Join(cfg.Search.Providers, ",") != "mock" {
		t.Fatalf("Search Providers = %#v, want mock", cfg.Search.Providers)
	}
	if cfg.Search.QueryRewriteProvider != "mock" {
		t.Fatalf("Search QueryRewriteProvider = %q, want mock", cfg.Search.QueryRewriteProvider)
	}
	if cfg.Search.QueryRewriteMaxQueries != 3 {
		t.Fatalf("Search QueryRewriteMaxQueries = %d, want 3", cfg.Search.QueryRewriteMaxQueries)
	}
	if cfg.Search.RerankProvider != "keyword" {
		t.Fatalf("Search RerankProvider = %q, want keyword", cfg.Search.RerankProvider)
	}
	if cfg.Search.CacheProvider != "memory" {
		t.Fatalf("Search CacheProvider = %q, want memory", cfg.Search.CacheProvider)
	}
	if cfg.Search.Timeout != "10s" {
		t.Fatalf("Search Timeout = %q, want 10s", cfg.Search.Timeout)
	}
	if cfg.Search.CacheTTL != "5m" {
		t.Fatalf("Search CacheTTL = %q, want 5m", cfg.Search.CacheTTL)
	}
	if cfg.Search.MaxResults != 10 {
		t.Fatalf("Search MaxResults = %d, want 10", cfg.Search.MaxResults)
	}
	if cfg.Search.MaxDocuments != 5 {
		t.Fatalf("Search MaxDocuments = %d, want 5", cfg.Search.MaxDocuments)
	}
	if cfg.Search.MaxContentRunes != 1200 {
		t.Fatalf("Search MaxContentRunes = %d, want 1200", cfg.Search.MaxContentRunes)
	}
	if cfg.Search.MaxTotalRunes != 6000 {
		t.Fatalf("Search MaxTotalRunes = %d, want 6000", cfg.Search.MaxTotalRunes)
	}
	if cfg.Search.MaxResponseBytes != 2*1024*1024 {
		t.Fatalf("Search MaxResponseBytes = %d, want 2097152", cfg.Search.MaxResponseBytes)
	}
	if cfg.Search.MaxConcurrency != 4 {
		t.Fatalf("Search MaxConcurrency = %d, want 4", cfg.Search.MaxConcurrency)
	}
	if cfg.Search.UserAgent != "VidwiseSearchBot/0.1" {
		t.Fatalf("Search UserAgent = %q", cfg.Search.UserAgent)
	}
	if cfg.Search.Bing.APIKeyEnv != "BING_SEARCH_API_KEY" {
		t.Fatalf("Search Bing APIKeyEnv = %q", cfg.Search.Bing.APIKeyEnv)
	}
	if cfg.Search.Tavily.APIKeyEnv != "TAVILY_API_KEY" {
		t.Fatalf("Search Tavily APIKeyEnv = %q", cfg.Search.Tavily.APIKeyEnv)
	}
	if cfg.Search.DuckDuckGo.BaseURL != "https://duckduckgo.com/html/" {
		t.Fatalf("Search DuckDuckGo BaseURL = %q", cfg.Search.DuckDuckGo.BaseURL)
	}
	if cfg.Search.Redis.KeyPrefix != "vidwise:search:" {
		t.Fatalf("Search Redis KeyPrefix = %q", cfg.Search.Redis.KeyPrefix)
	}
	if err := cfg.validate(); err != nil {
		t.Fatalf("validate returned error: %v", err)
	}
}

func TestApplyDefaultsUsesEnabledSearchProviders(t *testing.T) {
	var cfg Config
	cfg.Search.DuckDuckGo.Enabled = true
	cfg.Search.Internal.Enabled = true
	cfg.applyDefaults()

	if strings.Join(cfg.Search.Providers, ",") != "duckduckgo,internal" {
		t.Fatalf("Search Providers = %#v", cfg.Search.Providers)
	}
	if cfg.Search.Provider != "duckduckgo" {
		t.Fatalf("Search Provider = %q, want duckduckgo", cfg.Search.Provider)
	}
}

func TestApplyDefaultsNormalizesSearchProviders(t *testing.T) {
	var cfg Config
	cfg.Search.Providers = []string{" Bing ", "bing", "Tavily"}
	cfg.applyDefaults()

	if strings.Join(cfg.Search.Providers, ",") != "bing,tavily" {
		t.Fatalf("Search Providers = %#v", cfg.Search.Providers)
	}
}

func TestApplyDefaultsSetsModelProviderDefaults(t *testing.T) {
	var cfg Config
	cfg.applyDefaults()

	if cfg.ASR.Model.APIBaseURL != "https://dashscope.aliyuncs.com/compatible-mode/v1" {
		t.Fatalf("ASR APIBaseURL = %q", cfg.ASR.Model.APIBaseURL)
	}
	if cfg.ASR.Model.APIKeyEnv != "DASHSCOPE_API_KEY" {
		t.Fatalf("ASR APIKeyEnv = %q, want DASHSCOPE_API_KEY", cfg.ASR.Model.APIKeyEnv)
	}
	if cfg.ASR.Model.APITimeoutSeconds != 300 {
		t.Fatalf("ASR APITimeoutSeconds = %d, want 300", cfg.ASR.Model.APITimeoutSeconds)
	}
	if cfg.ASR.Model.MaxFileBytes != 7_500_000 {
		t.Fatalf("ASR MaxFileBytes = %d, want 7500000", cfg.ASR.Model.MaxFileBytes)
	}
	if cfg.ASR.Model.XFYunAPIBaseURL != "https://office-api-ist-dx.iflyaisol.com/v2" {
		t.Fatalf("ASR XFYunAPIBaseURL = %q", cfg.ASR.Model.XFYunAPIBaseURL)
	}
	if cfg.ASR.Model.XFYunAppIDEnv != "XFYUN_APP_ID" {
		t.Fatalf("ASR XFYunAppIDEnv = %q, want XFYUN_APP_ID", cfg.ASR.Model.XFYunAppIDEnv)
	}
	if cfg.ASR.Model.XFYunAccessKeyIDEnv != "XFYUN_API_KEY" {
		t.Fatalf("ASR XFYunAccessKeyIDEnv = %q, want XFYUN_API_KEY", cfg.ASR.Model.XFYunAccessKeyIDEnv)
	}
	if cfg.ASR.Model.XFYunAccessKeySecretEnv != "XFYUN_API_SECRET" {
		t.Fatalf("ASR XFYunAccessKeySecretEnv = %q, want XFYUN_API_SECRET", cfg.ASR.Model.XFYunAccessKeySecretEnv)
	}
	if cfg.ASR.Model.XFYunLanguage != "autodialect" {
		t.Fatalf("ASR XFYunLanguage = %q, want autodialect", cfg.ASR.Model.XFYunLanguage)
	}
	if cfg.ASR.Model.XFYunResultType != "transfer" {
		t.Fatalf("ASR XFYunResultType = %q, want transfer", cfg.ASR.Model.XFYunResultType)
	}
	if cfg.ASR.Model.XFYunAPITimeoutSeconds != 300 {
		t.Fatalf("ASR XFYunAPITimeoutSeconds = %d, want 300", cfg.ASR.Model.XFYunAPITimeoutSeconds)
	}
	if cfg.ASR.Model.XFYunPollIntervalSeconds != 3 {
		t.Fatalf("ASR XFYunPollIntervalSeconds = %d, want 3", cfg.ASR.Model.XFYunPollIntervalSeconds)
	}
	if cfg.ASR.Model.XFYunMaxPollSeconds != 600 {
		t.Fatalf("ASR XFYunMaxPollSeconds = %d, want 600", cfg.ASR.Model.XFYunMaxPollSeconds)
	}
	if cfg.ASR.Model.XFYunMaxFileBytes != 500_000_000 {
		t.Fatalf("ASR XFYunMaxFileBytes = %d, want 500000000", cfg.ASR.Model.XFYunMaxFileBytes)
	}
	if cfg.ASR.Model.BaiduTokenURL != "https://aip.baidubce.com/oauth/2.0/token" {
		t.Fatalf("ASR BaiduTokenURL = %q", cfg.ASR.Model.BaiduTokenURL)
	}
	if cfg.ASR.Model.BaiduAPIBaseURL != "https://vop.baidu.com" {
		t.Fatalf("ASR BaiduAPIBaseURL = %q", cfg.ASR.Model.BaiduAPIBaseURL)
	}
	if cfg.ASR.Model.BaiduAPIKeyEnv != "BAIDU_ASR_API_KEY" {
		t.Fatalf("ASR BaiduAPIKeyEnv = %q, want BAIDU_ASR_API_KEY", cfg.ASR.Model.BaiduAPIKeyEnv)
	}
	if cfg.ASR.Model.BaiduSecretKeyEnv != "BAIDU_ASR_SECRET_KEY" {
		t.Fatalf("ASR BaiduSecretKeyEnv = %q, want BAIDU_ASR_SECRET_KEY", cfg.ASR.Model.BaiduSecretKeyEnv)
	}
	if cfg.ASR.Model.BaiduCUID != "vidwise" {
		t.Fatalf("ASR BaiduCUID = %q, want vidwise", cfg.ASR.Model.BaiduCUID)
	}
	if cfg.ASR.Model.BaiduDevPID != 1537 {
		t.Fatalf("ASR BaiduDevPID = %d, want 1537", cfg.ASR.Model.BaiduDevPID)
	}
	if cfg.ASR.Model.BaiduRate != 16000 {
		t.Fatalf("ASR BaiduRate = %d, want 16000", cfg.ASR.Model.BaiduRate)
	}
	if cfg.ASR.Model.BaiduChannel != 1 {
		t.Fatalf("ASR BaiduChannel = %d, want 1", cfg.ASR.Model.BaiduChannel)
	}
	if cfg.ASR.Model.BaiduAPITimeoutSeconds != 60 {
		t.Fatalf("ASR BaiduAPITimeoutSeconds = %d, want 60", cfg.ASR.Model.BaiduAPITimeoutSeconds)
	}
	if cfg.ASR.Model.BaiduChunkSeconds != 55 {
		t.Fatalf("ASR BaiduChunkSeconds = %d, want 55", cfg.ASR.Model.BaiduChunkSeconds)
	}
	if cfg.ASR.Model.BaiduMaxChunkBytes != 10_000_000 {
		t.Fatalf("ASR BaiduMaxChunkBytes = %d, want 10000000", cfg.ASR.Model.BaiduMaxChunkBytes)
	}
	if cfg.Embedding.Provider != "local" {
		t.Fatalf("Embedding Provider = %q, want local", cfg.Embedding.Provider)
	}
	if cfg.Embedding.APIKeyEnv != "DASHSCOPE_API_KEY" {
		t.Fatalf("Embedding APIKeyEnv = %q, want DASHSCOPE_API_KEY", cfg.Embedding.APIKeyEnv)
	}
	if cfg.Embedding.APITimeoutSeconds != 120 {
		t.Fatalf("Embedding APITimeoutSeconds = %d, want 120", cfg.Embedding.APITimeoutSeconds)
	}
	if cfg.Embedding.BatchSize != 10 {
		t.Fatalf("Embedding BatchSize = %d, want 10", cfg.Embedding.BatchSize)
	}
	if cfg.Rerank.Provider != "local" {
		t.Fatalf("Rerank Provider = %q, want local", cfg.Rerank.Provider)
	}
	if cfg.Rerank.Model != "BAAI/bge-reranker-v2-m3" {
		t.Fatalf("Rerank Model = %q, want BAAI/bge-reranker-v2-m3", cfg.Rerank.Model)
	}
	if cfg.Rerank.Device != "auto" {
		t.Fatalf("Rerank Device = %q, want auto", cfg.Rerank.Device)
	}
	if cfg.Rerank.APITimeoutSeconds != 120 {
		t.Fatalf("Rerank APITimeoutSeconds = %d, want 120", cfg.Rerank.APITimeoutSeconds)
	}
}

func TestApplyDefaultsSetsSiliconFlowEmbeddingDefaults(t *testing.T) {
	var cfg Config
	cfg.Embedding.Provider = "siliconflow"
	cfg.applyDefaults()

	if cfg.Embedding.APIBaseURL != "https://api.siliconflow.cn/v1" {
		t.Fatalf("Embedding APIBaseURL = %q", cfg.Embedding.APIBaseURL)
	}
	if cfg.Embedding.APIKeyEnv != "SILICONFLOW_API_KEY" {
		t.Fatalf("Embedding APIKeyEnv = %q, want SILICONFLOW_API_KEY", cfg.Embedding.APIKeyEnv)
	}
	if err := cfg.validate(); err != nil {
		t.Fatalf("validate returned error: %v", err)
	}
}

func TestApplyDefaultsSetsSiliconFlowRerankDefaults(t *testing.T) {
	var cfg Config
	cfg.Rerank.Provider = "SiliconFlow"
	cfg.applyDefaults()

	if cfg.Rerank.Provider != "siliconflow" {
		t.Fatalf("Rerank Provider = %q, want siliconflow", cfg.Rerank.Provider)
	}
	if cfg.Rerank.Model != "BAAI/bge-reranker-v2-m3" {
		t.Fatalf("Rerank Model = %q, want BAAI/bge-reranker-v2-m3", cfg.Rerank.Model)
	}
	if cfg.Rerank.APIBaseURL != "https://api.siliconflow.cn/v1" {
		t.Fatalf("Rerank APIBaseURL = %q", cfg.Rerank.APIBaseURL)
	}
	if cfg.Rerank.APIKeyEnv != "SILICONFLOW_API_KEY" {
		t.Fatalf("Rerank APIKeyEnv = %q, want SILICONFLOW_API_KEY", cfg.Rerank.APIKeyEnv)
	}
	if err := cfg.validate(); err != nil {
		t.Fatalf("validate returned error: %v", err)
	}
}

func TestApplyDefaultsSetsAliyunRerankDefaults(t *testing.T) {
	var cfg Config
	cfg.Rerank.Provider = "dashscope"
	cfg.applyDefaults()

	if cfg.Rerank.Model != "qwen3-rerank" {
		t.Fatalf("Rerank Model = %q, want qwen3-rerank", cfg.Rerank.Model)
	}
	if cfg.Rerank.APIBaseURL != "https://dashscope.aliyuncs.com/compatible-mode/v1" {
		t.Fatalf("Rerank APIBaseURL = %q", cfg.Rerank.APIBaseURL)
	}
	if cfg.Rerank.APIKeyEnv != "DASHSCOPE_API_KEY" {
		t.Fatalf("Rerank APIKeyEnv = %q, want DASHSCOPE_API_KEY", cfg.Rerank.APIKeyEnv)
	}
	if err := cfg.validate(); err != nil {
		t.Fatalf("validate returned error: %v", err)
	}
}

func TestApplyDefaultsSetsAliyunGTERerankAPIBaseURL(t *testing.T) {
	var cfg Config
	cfg.Rerank.Provider = "aliyun"
	cfg.Rerank.Model = "gte-rerank-v2"
	cfg.applyDefaults()

	if cfg.Rerank.APIBaseURL != "https://dashscope.aliyuncs.com/api/v1" {
		t.Fatalf("Rerank APIBaseURL = %q", cfg.Rerank.APIBaseURL)
	}
	if err := cfg.validate(); err != nil {
		t.Fatalf("validate returned error: %v", err)
	}
}

func TestApplyDefaultsSetsServerTimeoutDefaults(t *testing.T) {
	var cfg Config
	cfg.applyDefaults()

	if cfg.Server.ReadHeaderTimeout != "5s" {
		t.Fatalf("ReadHeaderTimeout = %q, want 5s", cfg.Server.ReadHeaderTimeout)
	}
	if cfg.Server.ReadTimeout != "30s" {
		t.Fatalf("ReadTimeout = %q, want 30s", cfg.Server.ReadTimeout)
	}
	if cfg.Server.WriteTimeout != "0s" {
		t.Fatalf("WriteTimeout = %q, want 0s", cfg.Server.WriteTimeout)
	}
	if cfg.Server.IdleTimeout != "120s" {
		t.Fatalf("IdleTimeout = %q, want 120s", cfg.Server.IdleTimeout)
	}
	if cfg.Server.ShutdownTimeout != "10s" {
		t.Fatalf("ShutdownTimeout = %q, want 10s", cfg.Server.ShutdownTimeout)
	}
}

func TestApplyDefaultsSetsTaskDefaults(t *testing.T) {
	var cfg Config
	cfg.applyDefaults()

	if cfg.Task.MaxTracked != 1000 {
		t.Fatalf("MaxTracked = %d, want 1000", cfg.Task.MaxTracked)
	}
	if cfg.Task.MaxConcurrentVideo != 0 {
		t.Fatalf("MaxConcurrentVideo = %d, want 0", cfg.Task.MaxConcurrentVideo)
	}
	if cfg.Task.RetainFor != "24h" {
		t.Fatalf("RetainFor = %q, want 24h", cfg.Task.RetainFor)
	}
	retention, err := cfg.Task.RetentionDuration()
	if err != nil {
		t.Fatalf("RetentionDuration returned error: %v", err)
	}
	if retention != 24*time.Hour {
		t.Fatalf("RetentionDuration = %s, want 24h", retention)
	}
}

func TestValidateRejectsInvalidSearchConfig(t *testing.T) {
	var cfg Config
	cfg.applyDefaults()
	cfg.Search.Provider = "unknown"

	err := cfg.validate()
	if err == nil || !strings.Contains(err.Error(), "search.provider") {
		t.Fatalf("validate error = %v, want search.provider error", err)
	}
}

func TestLLMConfigSamplingParamsDisabled(t *testing.T) {
	disabled := true
	enabled := false

	tests := []struct {
		name string
		cfg  LLMConfig
		want bool
	}{
		{
			name: "manual true",
			cfg:  LLMConfig{DisableSamplingParams: &disabled},
			want: true,
		},
		{
			name: "manual false overrides reasoning model",
			cfg:  LLMConfig{Provider: "openai", Model: "gpt-5-mini", DisableSamplingParams: &enabled},
			want: false,
		},
		{
			name: "gpt 5 model",
			cfg:  LLMConfig{Provider: "openai", Model: "gpt-5"},
			want: true,
		},
		{
			name: "o series model",
			cfg:  LLMConfig{Provider: "openai", Model: "o4-mini"},
			want: true,
		},
		{
			name: "non reasoning openai model",
			cfg:  LLMConfig{Provider: "openai", Model: "gpt-4o-mini"},
			want: false,
		},
		{
			name: "non openai provider",
			cfg:  LLMConfig{Provider: "deepseek", Model: "gpt-5"},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.cfg.SamplingParamsDisabled(); got != tt.want {
				t.Fatalf("SamplingParamsDisabled() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestServerConfigDurations(t *testing.T) {
	cfg := ServerConfig{
		ReadHeaderTimeout: "3s",
		ReadTimeout:       "45s",
		WriteTimeout:      "0s",
		IdleTimeout:       "2m",
		ShutdownTimeout:   "15s",
	}

	if got, err := cfg.ReadHeaderTimeoutDuration(); err != nil || got != 3*time.Second {
		t.Fatalf("ReadHeaderTimeoutDuration = %s, %v; want 3s", got, err)
	}
	if got, err := cfg.ReadTimeoutDuration(); err != nil || got != 45*time.Second {
		t.Fatalf("ReadTimeoutDuration = %s, %v; want 45s", got, err)
	}
	if got, err := cfg.WriteTimeoutDuration(); err != nil || got != 0 {
		t.Fatalf("WriteTimeoutDuration = %s, %v; want 0s", got, err)
	}
	if got, err := cfg.IdleTimeoutDuration(); err != nil || got != 2*time.Minute {
		t.Fatalf("IdleTimeoutDuration = %s, %v; want 2m", got, err)
	}
	if got, err := cfg.ShutdownTimeoutDuration(); err != nil || got != 15*time.Second {
		t.Fatalf("ShutdownTimeoutDuration = %s, %v; want 15s", got, err)
	}
}

func TestValidateRejectsInvalidServerConfig(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantErr string
	}{
		{
			name: "invalid read header timeout",
			mutate: func(cfg *Config) {
				cfg.Server.ReadHeaderTimeout = "slow"
			},
			wantErr: "server.read_header_timeout",
		},
		{
			name: "zero read header timeout",
			mutate: func(cfg *Config) {
				cfg.Server.ReadHeaderTimeout = "0s"
			},
			wantErr: "server.read_header_timeout",
		},
		{
			name: "negative read timeout",
			mutate: func(cfg *Config) {
				cfg.Server.ReadTimeout = "-1s"
			},
			wantErr: "server.read_timeout",
		},
		{
			name: "negative write timeout",
			mutate: func(cfg *Config) {
				cfg.Server.WriteTimeout = "-1s"
			},
			wantErr: "server.write_timeout",
		},
		{
			name: "zero idle timeout",
			mutate: func(cfg *Config) {
				cfg.Server.IdleTimeout = "0s"
			},
			wantErr: "server.idle_timeout",
		},
		{
			name: "zero shutdown timeout",
			mutate: func(cfg *Config) {
				cfg.Server.ShutdownTimeout = "0s"
			},
			wantErr: "server.shutdown_timeout",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cfg Config
			cfg.applyDefaults()
			tt.mutate(&cfg)

			err := cfg.validate()
			if err == nil {
				t.Fatal("expected validation error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected error containing %q, got %v", tt.wantErr, err)
			}
		})
	}
}

func TestValidateRejectsInvalidRAGRetrievalConfig(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantErr string
	}{
		{
			name: "top k greater than search top k",
			mutate: func(cfg *Config) {
				cfg.RAG.Retrieval.SearchTopK = 5
				cfg.RAG.Retrieval.TopK = 6
			},
			wantErr: "rag.retrieval.top_k",
		},
		{
			name: "negative min score",
			mutate: func(cfg *Config) {
				cfg.RAG.Retrieval.MinScore = -0.1
			},
			wantErr: "rag.retrieval.min_score",
		},
		{
			name: "invalid context budget",
			mutate: func(cfg *Config) {
				cfg.RAG.Context.MaxRunes = 0
			},
			wantErr: "rag.context.max_runes",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cfg Config
			cfg.applyDefaults()
			tt.mutate(&cfg)

			err := cfg.validate()
			if err == nil {
				t.Fatal("expected validation error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected error containing %q, got %v", tt.wantErr, err)
			}
		})
	}
}

func TestValidateRejectsInvalidModelProviders(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantErr string
	}{
		{
			name: "invalid asr provider",
			mutate: func(cfg *Config) {
				cfg.ASR.Model.Provider = "unknown"
			},
			wantErr: "asr.model.provider",
		},
		{
			name: "invalid embedding provider",
			mutate: func(cfg *Config) {
				cfg.Embedding.Provider = "unknown"
			},
			wantErr: "embedding.provider",
		},
		{
			name: "invalid embedding batch size",
			mutate: func(cfg *Config) {
				cfg.Embedding.BatchSize = 0
			},
			wantErr: "embedding.batch_size",
		},
		{
			name: "invalid rerank provider",
			mutate: func(cfg *Config) {
				cfg.Rerank.Provider = "unknown"
			},
			wantErr: "rerank.provider",
		},
		{
			name: "invalid rerank top k",
			mutate: func(cfg *Config) {
				cfg.Rerank.TopK = 0
			},
			wantErr: "rerank.top_k",
		},
		{
			name: "negative rerank max chunks",
			mutate: func(cfg *Config) {
				cfg.Rerank.MaxChunksPerDoc = -1
			},
			wantErr: "rerank.max_chunks_per_doc",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cfg Config
			cfg.applyDefaults()
			tt.mutate(&cfg)

			err := cfg.validate()
			if err == nil {
				t.Fatal("expected validation error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected error containing %q, got %v", tt.wantErr, err)
			}
		})
	}
}

func TestTaskConfigRetentionDuration(t *testing.T) {
	tests := []struct {
		name    string
		cfg     TaskConfig
		want    time.Duration
		wantErr string
	}{
		{
			name: "empty uses default",
			cfg:  TaskConfig{},
			want: 24 * time.Hour,
		},
		{
			name: "custom duration",
			cfg:  TaskConfig{RetainFor: "90m"},
			want: 90 * time.Minute,
		},
		{
			name:    "invalid duration",
			cfg:     TaskConfig{RetainFor: "forever"},
			wantErr: "invalid duration",
		},
		{
			name:    "non-positive duration",
			cfg:     TaskConfig{RetainFor: "0s"},
			wantErr: "greater than 0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.cfg.RetentionDuration()
			if tt.wantErr != "" {
				if err == nil {
					t.Fatal("expected validation error")
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("expected error containing %q, got %v", tt.wantErr, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("RetentionDuration returned error: %v", err)
			}
			if got != tt.want {
				t.Fatalf("RetentionDuration = %s, want %s", got, tt.want)
			}
		})
	}
}

func TestValidateRejectsInvalidTaskConfig(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantErr string
	}{
		{
			name: "invalid max tracked",
			mutate: func(cfg *Config) {
				cfg.Task.MaxTracked = -1
			},
			wantErr: "task.max_tracked",
		},
		{
			name: "invalid max concurrent video",
			mutate: func(cfg *Config) {
				cfg.Task.MaxConcurrentVideo = -1
			},
			wantErr: "task.max_concurrent_video",
		},
		{
			name: "invalid retention duration",
			mutate: func(cfg *Config) {
				cfg.Task.RetainFor = "forever"
			},
			wantErr: "task.retain_for",
		},
		{
			name: "non-positive retention duration",
			mutate: func(cfg *Config) {
				cfg.Task.RetainFor = "0s"
			},
			wantErr: "task.retain_for",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cfg Config
			cfg.applyDefaults()
			tt.mutate(&cfg)

			err := cfg.validate()
			if err == nil {
				t.Fatal("expected validation error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected error containing %q, got %v", tt.wantErr, err)
			}
		})
	}
}
