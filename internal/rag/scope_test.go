package rag

import (
	"errors"
	"testing"
)

func TestResolveScopeChatPolicyPrefersUserOverSession(t *testing.T) {
	scope, err := ResolveScope(" user-1 ", " session-1 ", ChatScopePolicy())
	if err != nil {
		t.Fatalf("ResolveScope: %v", err)
	}
	if scope.UserID != "user-1" {
		t.Fatalf("UserID = %q, want user-1", scope.UserID)
	}
	if scope.SessionID != "" {
		t.Fatalf("SessionID = %q, want empty when user is present", scope.SessionID)
	}
	if scope.Unscoped {
		t.Fatal("expected scoped decision")
	}
}

func TestResolveScopeAllowsSessionOnly(t *testing.T) {
	scope, err := ResolveScope("", " session-1 ", ChatScopePolicy())
	if err != nil {
		t.Fatalf("ResolveScope: %v", err)
	}
	if scope.UserID != "" {
		t.Fatalf("UserID = %q, want empty", scope.UserID)
	}
	if scope.SessionID != "session-1" {
		t.Fatalf("SessionID = %q, want session-1", scope.SessionID)
	}
}

func TestResolveScopeChatPolicyRequiresScope(t *testing.T) {
	_, err := ResolveScope("", "", ChatScopePolicy())
	if !errors.Is(err, ErrScopeRequired) {
		t.Fatalf("error = %v, want ErrScopeRequired", err)
	}
}

func TestResolveScopeAllowsExplicitUnscopedByPolicy(t *testing.T) {
	scope, err := ResolveScope(" ", "\t", DefaultScopePolicy())
	if err != nil {
		t.Fatalf("ResolveScope: %v", err)
	}
	if !scope.Unscoped {
		t.Fatalf("expected unscoped decision, got %#v", scope)
	}
	if filter := scope.RetrieveFilter(); filter != nil {
		t.Fatalf("expected nil retrieve filter for unscoped decision, got %#v", filter)
	}
}

func TestResolveScopeRejectsUnscopedWhenStrict(t *testing.T) {
	_, err := ResolveScope("", "", StrictScopePolicy())
	if !errors.Is(err, ErrScopeRequired) {
		t.Fatalf("error = %v, want ErrScopeRequired", err)
	}
}

func TestDefaultPolicyKeepsCombinedUserAndSessionScope(t *testing.T) {
	filter, err := NewRetrieveFilterWithPolicy(" user-1 ", " session-1 ", DefaultScopePolicy())
	if err != nil {
		t.Fatalf("NewRetrieveFilterWithPolicy: %v", err)
	}
	if filter == nil {
		t.Fatal("expected filter")
	}
	if filter.UserID != "user-1" {
		t.Fatalf("UserID = %q, want user-1", filter.UserID)
	}
	if filter.SessionID != "session-1" {
		t.Fatalf("SessionID = %q, want session-1", filter.SessionID)
	}
}
