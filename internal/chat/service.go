package chat

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/cloudwego/eino/schema"
)

// ErrSessionNotFound is returned when the client references an unknown session.
var ErrSessionNotFound = errors.New("chat session not found")

// HistoryStore persists sessions and messages.
type HistoryStore interface {
	CreateSession(ctx context.Context, title string) (string, error)
	SessionExists(ctx context.Context, id string) (bool, error)
	AddMessage(ctx context.Context, sessionID, role, content string) error
	RecentMessages(ctx context.Context, sessionID string, limit int) ([]Message, error)
}

// Generator produces an assistant reply. It is satisfied by the eino react
// agent (via a small adapter) and by test doubles.
type Generator interface {
	Generate(ctx context.Context, messages []*schema.Message) (*schema.Message, error)
}

// Service is the application layer between the HTTP handler, the history
// store, and the agent.
type Service struct {
	store  HistoryStore
	model  Generator
	limits int // max history messages fed to the model
}

func NewService(store HistoryStore, generator Generator, historyLimit int) *Service {
	if historyLimit <= 0 {
		historyLimit = 20
	}
	return &Service{store: store, model: generator, limits: historyLimit}
}

// Chat runs one agent turn. When sessionID is empty a new session is created.
// The user message and the assistant reply are persisted only on success.
func (s *Service) Chat(ctx context.Context, sessionID, message string) (string, string, error) {
	if strings.TrimSpace(message) == "" {
		return "", "", errors.New("message is required")
	}

	if sessionID == "" {
		id, err := s.store.CreateSession(ctx, TitleFor(message))
		if err != nil {
			return "", "", fmt.Errorf("create session: %w", err)
		}
		sessionID = id
	} else {
		ok, err := s.store.SessionExists(ctx, sessionID)
		if err != nil {
			return "", "", fmt.Errorf("check session: %w", err)
		}
		if !ok {
			return "", "", ErrSessionNotFound
		}
	}

	history, err := s.store.RecentMessages(ctx, sessionID, s.limits)
	if err != nil {
		return "", "", fmt.Errorf("load history: %w", err)
	}

	inputs := make([]*schema.Message, 0, len(history)+1)
	for _, m := range history {
		role := schema.User
		if m.Role == "assistant" {
			role = schema.Assistant
		}
		inputs = append(inputs, &schema.Message{Role: role, Content: m.Content})
	}
	inputs = append(inputs, schema.UserMessage(message))

	out, err := s.model.Generate(ctx, inputs)
	if err != nil {
		return "", "", fmt.Errorf("agent generate: %w", err)
	}

	if err := s.store.AddMessage(ctx, sessionID, "user", message); err != nil {
		return "", "", fmt.Errorf("persist user message: %w", err)
	}
	if err := s.store.AddMessage(ctx, sessionID, "assistant", out.Content); err != nil {
		return "", "", fmt.Errorf("persist assistant message: %w", err)
	}
	return sessionID, out.Content, nil
}
