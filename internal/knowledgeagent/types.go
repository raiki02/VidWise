package knowledgeagent

import (
	"context"
	"time"

	"github.com/raiki02/vidwise/internal/chat"
	"github.com/raiki02/vidwise/internal/chatagent"
	"github.com/raiki02/vidwise/internal/rag"
)

type ActionType string

const (
	ActionAnswerFromKnowledge ActionType = "answer_from_knowledge"
	ActionProcessVideo        ActionType = "process_video"
	ActionIndexTaskTranscript ActionType = "index_task_transcript"
	ActionListSources         ActionType = "list_sources"
	ActionDeleteSource        ActionType = "delete_source"
	ActionFormatText          ActionType = "format_text"
)

type RiskLevel string

const (
	RiskReadOnly    RiskLevel = "read_only"
	RiskExpensive   RiskLevel = "expensive"
	RiskDestructive RiskLevel = "destructive"
)

type AgentAction struct {
	ID                   string         `json:"id"`
	Type                 ActionType     `json:"type"`
	Title                string         `json:"title"`
	Summary              string         `json:"summary"`
	RequiresConfirmation bool           `json:"requires_confirmation"`
	Input                map[string]any `json:"input,omitempty"`
	RiskLevel            RiskLevel      `json:"risk_level"`
}

type ExecutedAction struct {
	AgentAction
	Status string         `json:"status"`
	Output map[string]any `json:"output,omitempty"`
	Error  string         `json:"error,omitempty"`
}

type TurnRequest struct {
	UserID      string   `json:"user_id"`
	SessionID   string   `json:"session_id,omitempty"`
	Message     string   `json:"message"`
	SourceIDs   []string `json:"source_ids,omitempty"`
	DocumentIDs []string `json:"document_ids,omitempty"`
	TraceID     string   `json:"-"`
}

type TurnResponse struct {
	TraceID         string           `json:"trace_id,omitempty"`
	SessionID       string           `json:"session_id,omitempty"`
	Answer          string           `json:"answer"`
	PendingActions  []AgentAction    `json:"pending_actions,omitempty"`
	ExecutedActions []ExecutedAction `json:"executed_actions,omitempty"`
	TaskIDs         []string         `json:"task_ids,omitempty"`
	RAGTrace        *RAGTrace        `json:"rag_trace,omitempty"`
	Chunks          []TraceChunk     `json:"chunks,omitempty"`
}

type ConfirmRequest struct {
	UserID  string `json:"user_id"`
	TraceID string `json:"-"`
}

type IntentClassifyRequest struct {
	UserID      string
	SessionID   string
	Message     string
	SourceIDs   []string
	DocumentIDs []string
}

type IntentClassifyResult struct {
	Type       ActionType
	VideoURL   string
	VideoName  string
	TaskID     string
	SourceIDs  []string
	Text       string
	Confidence float64
	Reason     string
}

type RAGTrace struct {
	Triggered               bool         `json:"triggered"`
	Reason                  string       `json:"reason,omitempty"`
	Status                  string       `json:"status,omitempty"`
	Query                   string       `json:"query,omitempty"`
	Queries                 []string     `json:"queries,omitempty"`
	SourceIDs               []string     `json:"source_ids,omitempty"`
	DocumentIDs             []string     `json:"document_ids,omitempty"`
	ChunkCount              int          `json:"chunk_count"`
	ContextUsedChunks       int          `json:"context_used_chunks"`
	ContextSkippedDuplicate int          `json:"context_skipped_duplicates"`
	ContextTruncated        bool         `json:"context_truncated"`
	ContextChunks           []TraceChunk `json:"context_chunks,omitempty"`
	AnswerStatus            string       `json:"answer_status,omitempty"`
	CitationRequired        bool         `json:"citation_required"`
	HasCitations            bool         `json:"has_citations"`
	CitedSnippets           []int        `json:"cited_snippets,omitempty"`
	InvalidSnippets         []int        `json:"invalid_snippets,omitempty"`
}

type TraceChunk struct {
	SnippetNumber int     `json:"snippet_number,omitempty"`
	Text          string  `json:"text"`
	Score         float64 `json:"score"`
	SourceID      string  `json:"source_id,omitempty"`
	DocumentID    string  `json:"document_id,omitempty"`
	ChunkID       string  `json:"chunk_id,omitempty"`
	ContentHash   string  `json:"content_hash,omitempty"`
	TaskID        string  `json:"task_id,omitempty"`
	SessionID     string  `json:"session_id,omitempty"`
	ChunkIdx      int64   `json:"chunk_idx,omitempty"`
	SourceName    string  `json:"source_name,omitempty"`
	SourceURL     string  `json:"source_url,omitempty"`
	ContentType   string  `json:"content_type,omitempty"`
	DocumentTitle string  `json:"document_title,omitempty"`
	HeadingPath   string  `json:"heading_path,omitempty"`
}

type VideoProcessRequest struct {
	URL       string
	Name      string
	UserID    string
	SessionID string
	Language  string
}

type VideoProcessResult struct {
	TaskID    string
	TraceID   string
	Status    string
	SessionID string
}

type TranscriptIndexResult struct {
	Status      string
	TaskID      string
	ChunkCount  int
	ContentType string
	SourceIDs   []string
}

type TextFormatResult struct {
	Text string
}

type SessionStore interface {
	CreateSessionForUser(ctx context.Context, userID, title string) (*chat.Session, error)
	AddMessage(ctx context.Context, sessionID, role, content string) (*chat.Message, error)
	GetRecentMessages(ctx context.Context, sessionID string, limit int) ([]chat.Message, error)
}

type AnswerRunner interface {
	RunTurn(ctx context.Context, req chatagent.TurnRequest) (chatagent.TurnResult, error)
}

type SourceService interface {
	ListSources(ctx context.Context, req rag.SourceListRequest) ([]rag.SourceSummary, error)
	DeleteSourcesWithOptions(ctx context.Context, req rag.DeleteRequest) (rag.DeleteResult, error)
}

type VideoProcessor interface {
	StartVideoProcess(ctx context.Context, req VideoProcessRequest, traceID string) (VideoProcessResult, error)
}

type TranscriptIndexer interface {
	IndexTranscriptTask(ctx context.Context, taskID string) (TranscriptIndexResult, error)
}

type TextFormatter interface {
	FormatText(ctx context.Context, text string) (TextFormatResult, error)
}

type TaskReader interface {
	Get(id string) (TaskSnapshot, bool)
}

type IntentClassifier interface {
	ClassifyIntent(ctx context.Context, req IntentClassifyRequest) (IntentClassifyResult, error)
}

type TaskSnapshot struct {
	ID        string
	Type      string
	Status    string
	UserID    string
	SessionID string
	Output    map[string]any
	UpdatedAt time.Time
}
