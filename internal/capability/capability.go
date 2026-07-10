package capability

import (
	"sort"
	"strings"

	"github.com/raiki02/vidwise/internal/appconfig"
)

// Name identifies a runtime capability the gateway can expose or degrade.
type Name string

const (
	ChatSessionStore Name = "chat"
	MemoryStore      Name = "memory"
	VectorStore      Name = "vector_store"
	VectorCollection Name = "vector_collection"
	Embedding        Name = "embedding"
	Rerank           Name = "rerank"
	ASR              Name = "asr"
	VideoSummary     Name = "video_summary"
	RAG              Name = "rag"
	LLM              Name = "llm"
)

// Status captures whether a capability can serve traffic.
type Status string

const (
	Available   Status = "available"
	Degraded    Status = "degraded"
	Unavailable Status = "unavailable"
)

// Capability is the health record for one runtime capability.
type Capability struct {
	Name   Name   `json:"name"`
	Status Status `json:"status"`
	Reason string `json:"reason,omitempty"`
}

// Readiness is the deployment readiness view for required capabilities.
type Readiness struct {
	Ready    bool         `json:"ready"`
	Status   Status       `json:"status"`
	Required []Capability `json:"required"`
	Blocking []Capability `json:"blocking,omitempty"`
}

// Snapshot is an immutable capability view for the current gateway process.
type Snapshot struct {
	items map[Name]Capability
}

// RuntimeDeps are the adapters already constructed by gateway boot.
type RuntimeDeps struct {
	ChatSessionStore bool
	MemoryStore      bool
	VectorStore      bool
	VectorCollection bool
	Embedding        bool
	Rerank           bool
	ASR              bool
	VideoSummary     bool
	LLMConfig        appconfig.LLMConfig
}

// FromRuntime converts constructed adapters and config into a capability view.
func FromRuntime(deps RuntimeDeps) Snapshot {
	items := map[Name]Capability{
		ChatSessionStore: boolCapability(ChatSessionStore, deps.ChatSessionStore, "mysql chat session store unavailable"),
		MemoryStore:      boolCapability(MemoryStore, deps.MemoryStore, "mysql memory store unavailable"),
		VectorStore:      boolCapability(VectorStore, deps.VectorStore, "qdrant vector store unavailable"),
		VectorCollection: boolCapability(VectorCollection, deps.VectorCollection, "qdrant collection unavailable"),
		Embedding:        boolCapability(Embedding, deps.Embedding, "embedding adapter unavailable"),
		Rerank:           boolCapability(Rerank, deps.Rerank, "rerank adapter unavailable"),
		ASR:              boolCapability(ASR, deps.ASR, "asr service unavailable"),
		VideoSummary:     boolCapability(VideoSummary, deps.VideoSummary, "video summary service unavailable"),
		LLM:              llmCapability(deps.LLMConfig),
	}
	items[RAG] = ragCapability(items[VectorStore], items[VectorCollection], items[Embedding], items[Rerank])
	return Snapshot{items: items}
}

func boolCapability(name Name, ok bool, reason string) Capability {
	if ok {
		return Capability{Name: name, Status: Available}
	}
	return Capability{Name: name, Status: Unavailable, Reason: reason}
}

func llmCapability(cfg appconfig.LLMConfig) Capability {
	if cfg.Enabled != nil && !*cfg.Enabled {
		return Capability{Name: LLM, Status: Unavailable, Reason: "llm disabled by config"}
	}
	if strings.TrimSpace(cfg.Model) == "" {
		return Capability{Name: LLM, Status: Degraded, Reason: "llm enabled without configured model"}
	}
	return Capability{Name: LLM, Status: Available}
}

func ragCapability(vectorStore, vectorCollection, embedding, rerank Capability) Capability {
	if vectorStore.Status == Unavailable && embedding.Status == Unavailable {
		return Capability{Name: RAG, Status: Unavailable, Reason: "vector store and embedding unavailable"}
	}
	if vectorStore.Status == Unavailable {
		return Capability{Name: RAG, Status: Unavailable, Reason: vectorStore.Reason}
	}
	if embedding.Status == Unavailable {
		return Capability{Name: RAG, Status: Unavailable, Reason: embedding.Reason}
	}
	if vectorCollection.Status == Unavailable {
		return Capability{Name: RAG, Status: Unavailable, Reason: vectorCollection.Reason}
	}
	if rerank.Status == Unavailable {
		return Capability{Name: RAG, Status: Degraded, Reason: "rerank unavailable; vector search only"}
	}
	return Capability{Name: RAG, Status: Available}
}

// Get returns a capability record. Unknown capabilities are unavailable.
func (s Snapshot) Get(name Name) Capability {
	if s.items == nil {
		return Capability{Name: name, Status: Unavailable, Reason: "capability snapshot not initialized"}
	}
	if c, ok := s.items[name]; ok {
		return c
	}
	return Capability{Name: name, Status: Unavailable, Reason: "unknown capability"}
}

// Available reports whether a capability is fully available.
func (s Snapshot) Available(name Name) bool {
	return s.Get(name).Status == Available
}

// Usable reports whether a capability can serve traffic, possibly degraded.
func (s Snapshot) Usable(name Name) bool {
	status := s.Get(name).Status
	return status == Available || status == Degraded
}

// Readiness evaluates the capabilities that must be usable before this process
// should receive production traffic.
func (s Snapshot) Readiness(required ...Name) Readiness {
	out := Readiness{
		Ready:  true,
		Status: Available,
	}
	for _, name := range required {
		c := s.Get(name)
		out.Required = append(out.Required, c)
		if c.Status == Unavailable {
			out.Ready = false
			out.Status = Unavailable
			out.Blocking = append(out.Blocking, c)
		}
	}
	return out
}

// LegacyStatus keeps existing health clients working: degraded still means
// traffic can be served, so legacy fields report it as available.
func (s Snapshot) LegacyStatus(name Name) Status {
	status := s.Get(name).Status
	if status == Degraded {
		return Available
	}
	return status
}

// Map returns a deterministic string-keyed view suitable for JSON responses.
func (s Snapshot) Map() map[string]Capability {
	out := map[string]Capability{}
	if s.items == nil {
		return out
	}
	names := make([]string, 0, len(s.items))
	for name := range s.items {
		names = append(names, string(name))
	}
	sort.Strings(names)
	for _, name := range names {
		c := s.items[Name(name)]
		out[name] = c
	}
	return out
}
