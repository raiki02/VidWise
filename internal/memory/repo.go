package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

// Repo provides CRUD operations for cross-session user memory facts.
type Repo struct {
	db *gorm.DB
}

func NewRepo(db *gorm.DB) *Repo {
	return &Repo{db: db}
}

func (r *Repo) AutoMigrate() error {
	return r.db.AutoMigrate(&MemoryFact{})
}

// GetFactsByUser returns all active (non-superseded) facts for a user.
func (r *Repo) GetFactsByUser(ctx context.Context, userID string) ([]MemoryFact, error) {
	var facts []MemoryFact
	if err := r.db.WithContext(ctx).
		Where("user_id = ? AND superseded_by IS NULL", userID).
		Order(memoryFactOrderBy()).
		Find(&facts).Error; err != nil {
		return nil, fmt.Errorf("get memory facts: %w", err)
	}
	return facts, nil
}

func memoryFactOrderBy() clause.OrderBy {
	return clause.OrderBy{Columns: []clause.OrderByColumn{
		{Column: clause.Column{Name: "category"}},
		{Column: clause.Column{Name: "key"}},
	}}
}

// GetFactSummaries returns lightweight views for the LLM prompt.
func (r *Repo) GetFactSummaries(ctx context.Context, userID string) ([]MemoryFactSummary, error) {
	facts, err := r.GetFactsByUser(ctx, userID)
	if err != nil {
		return nil, err
	}
	summaries := make([]MemoryFactSummary, 0, len(facts))
	for _, f := range facts {
		summaries = append(summaries, MemoryFactSummary{
			ID:         f.ID,
			Category:   f.Category,
			Key:        f.Key,
			Value:      f.Value,
			Confidence: string(f.Confidence),
		})
	}
	return summaries, nil
}

// UpsertFact creates or updates a memory fact. When updating, it bumps the mention count.
func (r *Repo) UpsertFact(ctx context.Context, userID string, ef ExtractedFact, sessionID string) (*MemoryFact, error) {
	// Try to find existing by user + category + key
	var existing MemoryFact
	err := r.db.WithContext(ctx).
		Where("user_id = ? AND category = ? AND `key` = ? AND superseded_by IS NULL", userID, ef.Category, ef.Key).
		First(&existing).Error

	if err != nil && err != gorm.ErrRecordNotFound {
		return nil, fmt.Errorf("lookup existing fact: %w", err)
	}

	if existing.ID != "" {
		// Update existing fact
		existing.Value = ef.Value
		existing.Evidence = ef.Evidence
		existing.Source = ef.Source
		existing.Confidence = ef.Confidence
		existing.MentionCount++
		existing.LastMentionedAt = time.Now()
		existing.ExtractedFrom = sessionID

		if err := r.db.WithContext(ctx).Save(&existing).Error; err != nil {
			return nil, fmt.Errorf("update fact: %w", err)
		}
		slog.Info("memory.fact_updated", "user_id", userID, "category", ef.Category, "key", ef.Key)
		return &existing, nil
	}

	// Create new fact
	fact := &MemoryFact{
		UserID:          userID,
		Category:        ef.Category,
		Key:             ef.Key,
		Value:           ef.Value,
		Evidence:        ef.Evidence,
		Source:          ef.Source,
		Confidence:      ef.Confidence,
		MentionCount:    1,
		LastMentionedAt: time.Now(),
		ExtractedFrom:   sessionID,
	}
	if err := r.db.WithContext(ctx).Create(fact).Error; err != nil {
		return nil, fmt.Errorf("create fact: %w", err)
	}
	slog.Info("memory.fact_created", "user_id", userID, "category", ef.Category, "key", ef.Key)
	return fact, nil
}

// SupersedeFact marks an old fact as superseded by a new one.
// The userID parameter scopes the operation to prevent cross-user tampering.
func (r *Repo) SupersedeFact(ctx context.Context, userID, oldID, newID, reason string) error {
	now := time.Now()
	return r.db.WithContext(ctx).Model(&MemoryFact{}).
		Where("id = ? AND user_id = ?", oldID, userID).
		Updates(map[string]any{
			"superseded_by": newID,
			"superseded_at": now,
			"revision_note": reason,
		}).Error
}

// ConfirmFact bumps the mention count and last_mentioned_at of an existing fact.
// The userID parameter scopes the operation to prevent cross-user tampering.
func (r *Repo) ConfirmFact(ctx context.Context, userID, factID string) error {
	return r.db.WithContext(ctx).Model(&MemoryFact{}).
		Where("id = ? AND user_id = ?", factID, userID).
		Updates(map[string]any{
			"mention_count":     gorm.Expr("mention_count + 1"),
			"last_mentioned_at": time.Now(),
		}).Error
}

// ApplyResult summarizes how many extracted memory operations were applied.
type ApplyResult struct {
	Applied int
	Skipped int
}

// ApplyExtractedFacts applies LLM-produced memory operations for a completed
// turn. Unknown actions and incomplete target-dependent actions are skipped so
// a single bad extraction cannot fail the whole background task.
func (r *Repo) ApplyExtractedFacts(ctx context.Context, userID, sessionID string, facts []ExtractedFact) (ApplyResult, error) {
	return applyExtractedFacts(ctx, repoFactApplier{repo: r}, userID, sessionID, facts)
}

type factApplier interface {
	UpsertFact(ctx context.Context, userID string, ef ExtractedFact, sessionID string) (*MemoryFact, error)
	SupersedeFact(ctx context.Context, userID, oldID, newID, reason string) error
	ConfirmFact(ctx context.Context, userID, factID string) error
}

type repoFactApplier struct {
	repo *Repo
}

func (a repoFactApplier) UpsertFact(ctx context.Context, userID string, ef ExtractedFact, sessionID string) (*MemoryFact, error) {
	return a.repo.UpsertFact(ctx, userID, ef, sessionID)
}

func (a repoFactApplier) SupersedeFact(ctx context.Context, userID, oldID, newID, reason string) error {
	return a.repo.SupersedeFact(ctx, userID, oldID, newID, reason)
}

func (a repoFactApplier) ConfirmFact(ctx context.Context, userID, factID string) error {
	return a.repo.ConfirmFact(ctx, userID, factID)
}

func applyExtractedFacts(ctx context.Context, applier factApplier, userID, sessionID string, facts []ExtractedFact) (ApplyResult, error) {
	var result ApplyResult
	for _, f := range facts {
		switch f.Action {
		case "create":
			if _, err := applier.UpsertFact(ctx, userID, f, sessionID); err != nil {
				return result, fmt.Errorf("create memory fact: %w", err)
			}
			result.Applied++
		case "update":
			if _, err := applier.UpsertFact(ctx, userID, f, sessionID); err != nil {
				return result, fmt.Errorf("update memory fact: %w", err)
			}
			result.Applied++
		case "supersede":
			if f.TargetID == "" {
				result.Skipped++
				continue
			}
			newFact, err := applier.UpsertFact(ctx, userID, f, sessionID)
			if err != nil {
				return result, fmt.Errorf("create superseding memory fact: %w", err)
			}
			if newFact == nil || newFact.ID == "" {
				return result, fmt.Errorf("create superseding memory fact: empty id")
			}
			if err := applier.SupersedeFact(ctx, userID, f.TargetID, newFact.ID, f.Reason); err != nil {
				return result, fmt.Errorf("supersede memory fact: %w", err)
			}
			result.Applied++
		case "confirm":
			if f.TargetID == "" {
				result.Skipped++
				continue
			}
			if err := applier.ConfirmFact(ctx, userID, f.TargetID); err != nil {
				return result, fmt.Errorf("confirm memory fact: %w", err)
			}
			result.Applied++
		default:
			result.Skipped++
			slog.Warn("memory.unknown_action", "action", f.Action)
		}
	}
	return result, nil
}

// FormatForPrompt returns a formatted string of user facts for inclusion in the system prompt.
func (r *Repo) FormatForPrompt(ctx context.Context, userID string) string {
	facts, err := r.GetFactsByUser(ctx, userID)
	if err != nil || len(facts) == 0 {
		return ""
	}

	s := "## 用户画像（跨会话记忆）\n"
	s += "以下是从之前对话中提取的关于该用户的信息。你可以利用这些信息更好地理解用户，但要注意：\n"
	s += "- 这些信息可能不完全准确，需要结合当前对话判断\n"
	s += "- 如果用户在当前对话中表达了矛盾的信息，以用户最新表达为准\n"
	s += "- 不要直接复述这些信息，自然地运用它们来个性化回复\n\n"

	// Group by category
	categories := map[string][]MemoryFact{}
	for _, f := range facts {
		categories[f.Category] = append(categories[f.Category], f)
	}

	for cat, catFacts := range categories {
		s += fmt.Sprintf("**%s**:\n", cat)
		for _, f := range catFacts {
			s += fmt.Sprintf("- %s: %s (置信度: %s)\n", f.Key, f.Value, f.Confidence)
		}
		s += "\n"
	}
	return s
}

// DumpForPromptJSON returns existing facts as a compact JSON string for LLM prompts.
func (r *Repo) DumpForPromptJSON(ctx context.Context, userID string) string {
	summaries, err := r.GetFactSummaries(ctx, userID)
	if err != nil || len(summaries) == 0 {
		return "[]"
	}
	b, err := json.Marshal(summaries)
	if err != nil {
		return "[]"
	}
	return string(b)
}
