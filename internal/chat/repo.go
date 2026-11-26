package chat

import (
	"context"
	"fmt"
	"time"

	"gorm.io/gorm"
)

// Repo is the MySQL-backed HistoryStore.
type Repo struct {
	db *gorm.DB
}

func NewRepo(db *gorm.DB) *Repo {
	return &Repo{db: db}
}

func (r *Repo) AutoMigrate() error {
	return r.db.AutoMigrate(&Session{}, &Message{})
}

func (r *Repo) CreateSession(ctx context.Context, title string) (string, error) {
	s := &Session{Title: title}
	if err := r.db.WithContext(ctx).Create(s).Error; err != nil {
		return "", fmt.Errorf("create session: %w", err)
	}
	return s.ID, nil
}

func (r *Repo) SessionExists(ctx context.Context, id string) (bool, error) {
	var count int64
	if err := r.db.WithContext(ctx).Model(&Session{}).Where("id = ?", id).Count(&count).Error; err != nil {
		return false, fmt.Errorf("count session: %w", err)
	}
	return count > 0, nil
}

func (r *Repo) AddMessage(ctx context.Context, sessionID, role, content string) error {
	m := &Message{SessionID: sessionID, Role: role, Content: content}
	if err := r.db.WithContext(ctx).Create(m).Error; err != nil {
		return fmt.Errorf("add message: %w", err)
	}
	_ = r.db.WithContext(ctx).Model(&Session{}).Where("id = ?", sessionID).Update("updated_at", time.Now()).Error
	return nil
}

func (r *Repo) RecentMessages(ctx context.Context, sessionID string, limit int) ([]Message, error) {
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
	for i, j := 0, len(msgs)-1; i < j; i, j = i+1, j-1 {
		msgs[i], msgs[j] = msgs[j], msgs[i]
	}
	return msgs, nil
}
