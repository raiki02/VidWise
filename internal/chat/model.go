package chat

import (
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

// Session represents a chat conversation session.
type Session struct {
	ID        string    `gorm:"type:varchar(36);primaryKey" json:"id"`
	UserID    string    `gorm:"type:varchar(36);index:idx_cs_user" json:"user_id,omitempty"`
	Title     string    `gorm:"type:varchar(512)" json:"title"`
	CreatedAt time.Time `gorm:"autoCreateTime" json:"created_at"`
	UpdatedAt time.Time `gorm:"autoUpdateTime" json:"updated_at"`
}

func (Session) TableName() string { return "chat_sessions" }

func (s *Session) BeforeCreate(tx *gorm.DB) error {
	if s.ID == "" {
		s.ID = uuid.New().String()
	}
	return nil
}

// Message is a single message in a chat session.
type Message struct {
	ID        string    `gorm:"type:varchar(36);primaryKey" json:"id"`
	SessionID string    `gorm:"type:varchar(36);not null;index:idx_msg_session" json:"session_id"`
	Role      string    `gorm:"type:varchar(16);not null" json:"role"` // "user" or "assistant"
	Content   string    `gorm:"type:mediumtext;not null" json:"content"`
	CreatedAt time.Time `gorm:"autoCreateTime" json:"created_at"`
}

func (Message) TableName() string { return "chat_messages" }

func (m *Message) BeforeCreate(tx *gorm.DB) error {
	if m.ID == "" {
		m.ID = uuid.New().String()
	}
	return nil
}
