package user

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	mysqlclient "github.com/raiki02/vidwise/internal/storage/mysql"
	"gorm.io/gorm"
)

type Repo struct {
	db *gorm.DB
}

func NewRepo(client *mysqlclient.Client) *Repo {
	return &Repo{db: client.DB}
}

// GetOrCreateUser returns an existing user by name or creates one.
func (r *Repo) GetOrCreateUser(ctx context.Context, name string) (*User, error) {
	var u User
	err := r.db.WithContext(ctx).Where("name = ?", name).First(&u).Error
	if err == nil {
		return &u, nil
	}
	if !errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, fmt.Errorf("get user: %w", err)
	}

	u = User{Name: name}
	if err := r.db.WithContext(ctx).Create(&u).Error; err != nil {
		return nil, fmt.Errorf("create user: %w", err)
	}
	return &u, nil
}

// CreateSession creates a new session for a user.
func (r *Repo) CreateSession(ctx context.Context, userID, title string, metadata map[string]any) (*Session, error) {
	s := &Session{
		UserID: userID,
		Status: "active",
	}
	if title != "" {
		s.Title = &title
	}
	if metadata != nil {
		b, _ := json.Marshal(metadata)
		raw := string(b)
		s.Metadata = &raw
	}
	if err := r.db.WithContext(ctx).Create(s).Error; err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}
	return s, nil
}

// GetSession returns a session by ID.
func (r *Repo) GetSession(ctx context.Context, id string) (*Session, error) {
	var s Session
	err := r.db.WithContext(ctx).Where("id = ?", id).First(&s).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, fmt.Errorf("session not found: %s", id)
		}
		return nil, fmt.Errorf("get session: %w", err)
	}
	return &s, nil
}

// GetSessionsByUser returns all sessions for a user.
func (r *Repo) GetSessionsByUser(ctx context.Context, userID string) ([]Session, error) {
	var sessions []Session
	err := r.db.WithContext(ctx).
		Where("user_id = ?", userID).
		Order("created_at DESC").
		Find(&sessions).Error
	if err != nil {
		return nil, fmt.Errorf("get sessions: %w", err)
	}
	return sessions, nil
}
