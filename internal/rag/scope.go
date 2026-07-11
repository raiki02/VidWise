package rag

import (
	"errors"
	"strings"
)

var ErrScopeRequired = errors.New("rag scope is required")

// ScopePolicy turns caller identity into an explicit RAG retrieval scope.
type ScopePolicy struct {
	AllowUnscoped         bool
	PreferUserOverSession bool
}

// Scope is the normalized result of applying a ScopePolicy.
type Scope struct {
	UserID    string
	SessionID string
	Unscoped  bool
}

func DefaultScopePolicy() ScopePolicy {
	return ScopePolicy{AllowUnscoped: true}
}

// PersonalKnowledgeScopePolicy is for user-facing RAG operations: user_id is
// the stable personal knowledge-base boundary, while session_id is a fallback
// for temporary or anonymous use.
func PersonalKnowledgeScopePolicy() ScopePolicy {
	return ScopePolicy{
		AllowUnscoped:         false,
		PreferUserOverSession: true,
	}
}

func ChatScopePolicy() ScopePolicy {
	return PersonalKnowledgeScopePolicy()
}

func StrictScopePolicy() ScopePolicy {
	return ScopePolicy{AllowUnscoped: false}
}

// ResolveScope normalizes user/session identifiers and makes unscoped access
// an explicit policy decision instead of an accidental empty-string default.
func ResolveScope(userID, sessionID string, policy ScopePolicy) (Scope, error) {
	userID = strings.TrimSpace(userID)
	sessionID = strings.TrimSpace(sessionID)

	if policy.PreferUserOverSession && userID != "" {
		sessionID = ""
	}

	if userID == "" && sessionID == "" {
		if !policy.AllowUnscoped {
			return Scope{}, ErrScopeRequired
		}
		return Scope{Unscoped: true}, nil
	}

	return Scope{
		UserID:    userID,
		SessionID: sessionID,
	}, nil
}

func (s Scope) RetrieveFilter() *RetrieveFilter {
	if s.Unscoped || (s.UserID == "" && s.SessionID == "") {
		return nil
	}
	return &RetrieveFilter{
		UserID:    s.UserID,
		SessionID: s.SessionID,
	}
}

func NewRetrieveFilterWithPolicy(userID, sessionID string, policy ScopePolicy) (*RetrieveFilter, error) {
	scope, err := ResolveScope(userID, sessionID, policy)
	if err != nil {
		return nil, err
	}
	return scope.RetrieveFilter(), nil
}
