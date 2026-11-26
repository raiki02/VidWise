package chat

import (
	"context"
	"errors"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

var errSessionNotFound = errors.New("chat session not found")

// MemoryStore is the in-memory HistoryStore used when MySQL is not configured.
// Data is lost on restart.
type MemoryStore struct {
	mu       sync.Mutex
	sessions map[string]bool      // sessionID -> exists
	messages map[string][]Message // sessionID -> chronological user/assistant messages
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		sessions: map[string]bool{},
		messages: map[string][]Message{},
	}
}

func (s *MemoryStore) CreateSession(_ context.Context, title string) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	id := uuid.New().String()
	s.sessions[id] = true
	s.messages[id] = nil
	_ = title
	return id, nil
}

func (s *MemoryStore) SessionExists(_ context.Context, id string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ok := s.sessions[id]
	return ok, nil
}

func (s *MemoryStore) AddMessage(_ context.Context, sessionID, role, content string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.sessions[sessionID]; !ok {
		return errSessionNotFound
	}
	s.messages[sessionID] = append(s.messages[sessionID], Message{
		ID:        uuid.NewString(),
		SessionID: sessionID,
		Role:      role,
		Content:   content,
		CreatedAt: time.Now(),
	})
	return nil
}

func (s *MemoryStore) RecentMessages(_ context.Context, sessionID string, limit int) ([]Message, error) {
	if limit <= 0 {
		limit = 20
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.sessions[sessionID]; !ok {
		return nil, errSessionNotFound
	}
	msgs := s.messages[sessionID]
	if len(msgs) > limit {
		msgs = msgs[len(msgs)-limit:]
	}
	out := make([]Message, len(msgs))
	copy(out, msgs)
	return out, nil
}

// TitleFor returns a short title derived from the first user message.
func TitleFor(message string) string {
	message = strings.TrimSpace(message)
	runes := []rune(message)
	if len(runes) > 30 {
		return string(runes[:30])
	}
	if message == "" {
		return "新对话"
	}
	return message
}
