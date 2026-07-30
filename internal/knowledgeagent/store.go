package knowledgeagent

import (
	"errors"
	"sync"
	"time"

	"github.com/google/uuid"
)

var (
	ErrActionNotFound        = errors.New("action not found")
	ErrActionAlreadyExecuted = errors.New("action already executed")
	ErrActionUserMismatch    = errors.New("action does not belong to user")
)

const (
	defaultMaxActions      = 1000
	defaultActionRetainFor = 24 * time.Hour
	actionStatusPending    = "pending"
	actionStatusExecuting  = "executing"
	actionStatusExecuted   = "executed"
	actionStatusFailed     = "failed"
)

type ActionStoreOptions struct {
	MaxActions int
	RetainFor  time.Duration
	Now        func() time.Time
}

type ActionStore struct {
	mu        sync.Mutex
	actions   map[string]storedAction
	max       int
	retainFor time.Duration
	now       func() time.Time
}

type storedAction struct {
	Action    AgentAction
	UserID    string
	SessionID string
	Status    string
	Result    *ExecutedAction
	CreatedAt time.Time
	UpdatedAt time.Time
}

func NewActionStore(opts ActionStoreOptions) *ActionStore {
	if opts.MaxActions <= 0 {
		opts.MaxActions = defaultMaxActions
	}
	if opts.RetainFor <= 0 {
		opts.RetainFor = defaultActionRetainFor
	}
	if opts.Now == nil {
		opts.Now = time.Now
	}
	return &ActionStore{
		actions:   make(map[string]storedAction),
		max:       opts.MaxActions,
		retainFor: opts.RetainFor,
		now:       opts.Now,
	}
}

func (s *ActionStore) Add(action AgentAction, userID, sessionID string) AgentAction {
	if s == nil {
		return action
	}
	now := s.now()
	if action.ID == "" {
		action.ID = uuid.New().String()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.actions == nil {
		s.actions = make(map[string]storedAction)
	}
	s.actions[action.ID] = storedAction{
		Action:    action,
		UserID:    userID,
		SessionID: sessionID,
		Status:    actionStatusPending,
		CreatedAt: now,
		UpdatedAt: now,
	}
	s.pruneLocked(now)
	return action
}

func (s *ActionStore) Reserve(actionID, userID string) (AgentAction, error) {
	if s == nil {
		return AgentAction{}, ErrActionNotFound
	}
	now := s.now()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pruneLocked(now)
	record, ok := s.actions[actionID]
	if !ok {
		return AgentAction{}, ErrActionNotFound
	}
	if record.UserID != "" && userID != "" && record.UserID != userID {
		return AgentAction{}, ErrActionUserMismatch
	}
	if record.Status != actionStatusPending {
		return AgentAction{}, ErrActionAlreadyExecuted
	}
	record.Status = actionStatusExecuting
	record.UpdatedAt = now
	s.actions[actionID] = record
	return record.Action, nil
}

func (s *ActionStore) Complete(actionID string, result ExecutedAction) {
	if s == nil {
		return
	}
	now := s.now()
	s.mu.Lock()
	defer s.mu.Unlock()
	record, ok := s.actions[actionID]
	if !ok {
		return
	}
	result.Status = actionStatusExecuted
	record.Result = &result
	record.Status = actionStatusExecuted
	record.UpdatedAt = now
	s.actions[actionID] = record
}

func (s *ActionStore) Fail(actionID string, result ExecutedAction) {
	if s == nil {
		return
	}
	now := s.now()
	s.mu.Lock()
	defer s.mu.Unlock()
	record, ok := s.actions[actionID]
	if !ok {
		return
	}
	result.Status = actionStatusFailed
	record.Result = &result
	record.Status = actionStatusFailed
	record.UpdatedAt = now
	s.actions[actionID] = record
}

func (s *ActionStore) pruneLocked(now time.Time) {
	if len(s.actions) == 0 {
		return
	}
	for id, action := range s.actions {
		if now.Sub(action.UpdatedAt) >= s.retainFor {
			delete(s.actions, id)
		}
	}
	if s.max <= 0 || len(s.actions) <= s.max {
		return
	}
	for len(s.actions) > s.max {
		var oldestID string
		var oldest time.Time
		for id, action := range s.actions {
			if oldestID == "" || action.UpdatedAt.Before(oldest) {
				oldestID = id
				oldest = action.UpdatedAt
			}
		}
		if oldestID == "" {
			return
		}
		delete(s.actions, oldestID)
	}
}
