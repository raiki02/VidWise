package chat

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"gorm.io/gorm"
)

// Repo provides CRUD for chat sessions and messages.
type Repo struct {
	db *gorm.DB
}

func NewRepo(db *gorm.DB) *Repo {
	return &Repo{db: db}
}

func (r *Repo) AutoMigrate() error {
	return r.db.AutoMigrate(&Session{}, &Message{})
}

// ---- Sessions ----

func (r *Repo) CreateSession(ctx context.Context, title string) (*Session, error) {
	return r.CreateSessionForUser(ctx, "", title)
}

func (r *Repo) CreateSessionForUser(ctx context.Context, userID, title string) (*Session, error) {
	s := &Session{UserID: userID, Title: title}
	if err := r.db.WithContext(ctx).Create(s).Error; err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}
	return s, nil
}

func (r *Repo) UpdateSessionTitle(ctx context.Context, sessionID, title string) error {
	return r.db.WithContext(ctx).Model(&Session{}).Where("id = ?", sessionID).Update("title", title).Error
}

func (r *Repo) ListSessions(ctx context.Context, limit int) ([]Session, error) {
	if limit <= 0 {
		limit = 50
	}
	var sessions []Session
	if err := r.db.WithContext(ctx).Order("updated_at DESC").Limit(limit).Find(&sessions).Error; err != nil {
		return nil, fmt.Errorf("list sessions: %w", err)
	}
	return sessions, nil
}

func (r *Repo) ListSessionsByUser(ctx context.Context, userID string, limit int) ([]Session, error) {
	if limit <= 0 {
		limit = 50
	}
	var sessions []Session
	if err := r.db.WithContext(ctx).
		Where("user_id = ?", userID).
		Order("updated_at DESC").
		Limit(limit).
		Find(&sessions).Error; err != nil {
		return nil, fmt.Errorf("list sessions by user: %w", err)
	}
	return sessions, nil
}

func (r *Repo) GetSession(ctx context.Context, id string) (*Session, error) {
	var s Session
	if err := r.db.WithContext(ctx).First(&s, "id = ?", id).Error; err != nil {
		return nil, fmt.Errorf("get session: %w", err)
	}
	return &s, nil
}

// ---- Messages ----

func (r *Repo) AddMessage(ctx context.Context, sessionID, role, content string) (*Message, error) {
	m := &Message{SessionID: sessionID, Role: role, Content: content}
	if err := r.db.WithContext(ctx).Create(m).Error; err != nil {
		return nil, fmt.Errorf("add message: %w", err)
	}
	// Touch session updated_at.
	_ = r.db.WithContext(ctx).Model(&Session{}).Where("id = ?", sessionID).Update("updated_at", time.Now()).Error
	return m, nil
}

func (r *Repo) GetMessages(ctx context.Context, sessionID string, limit int) ([]Message, error) {
	if limit <= 0 {
		limit = 200
	}
	var msgs []Message
	if err := r.db.WithContext(ctx).Where("session_id = ?", sessionID).Order("created_at ASC").Limit(limit).Find(&msgs).Error; err != nil {
		return nil, fmt.Errorf("get messages: %w", err)
	}
	return msgs, nil
}

func (r *Repo) GetRecentMessages(ctx context.Context, sessionID string, limit int) ([]Message, error) {
	if limit <= 0 {
		limit = 20
	}
	var msgs []Message
	if err := r.db.WithContext(ctx).
		Where("session_id = ?", sessionID).
		Order("created_at DESC").
		Limit(limit).
		Find(&msgs).Error; err != nil {
		return nil, fmt.Errorf("get recent messages: %w", err)
	}
	// Reverse to chronological order
	for i, j := 0, len(msgs)-1; i < j; i, j = i+1, j-1 {
		msgs[i], msgs[j] = msgs[j], msgs[i]
	}
	return msgs, nil
}

// GenerateTitle uses the LLM to generate a session title from the first few messages.
func GenerateTitle(ctx context.Context, llmConfig interface{}, messages []Message) string {
	if len(messages) == 0 {
		return "新对话"
	}
	// Use the first user message to generate title
	firstUserMsg := ""
	for _, m := range messages {
		if m.Role == "user" {
			firstUserMsg = m.Content
			break
		}
	}
	if firstUserMsg == "" {
		return "新对话"
	}
	// Truncate long messages for title generation
	runes := []rune(firstUserMsg)
	if len(runes) > 50 {
		firstUserMsg = string(runes[:50]) + "..."
	}
	slog.Info("chat.generate_title", "preview", firstUserMsg)
	return firstUserMsg
}
