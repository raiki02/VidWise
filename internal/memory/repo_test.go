package memory

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	gormmysql "gorm.io/driver/mysql"
	"gorm.io/gorm"
	gormlogger "gorm.io/gorm/logger"
)

type fakeFactApplier struct {
	upserts    []ExtractedFact
	supersedes []supersedeCall
	confirms   []string
	err        error
}

type supersedeCall struct {
	oldID  string
	newID  string
	reason string
}

func (f *fakeFactApplier) UpsertFact(_ context.Context, _ string, ef ExtractedFact, _ string) (*MemoryFact, error) {
	if f.err != nil {
		return nil, f.err
	}
	f.upserts = append(f.upserts, ef)
	return &MemoryFact{ID: ef.Key + "-id"}, nil
}

func (f *fakeFactApplier) SupersedeFact(_ context.Context, _, oldID, newID, reason string) error {
	if f.err != nil {
		return f.err
	}
	f.supersedes = append(f.supersedes, supersedeCall{oldID: oldID, newID: newID, reason: reason})
	return nil
}

func (f *fakeFactApplier) ConfirmFact(_ context.Context, _, factID string) error {
	if f.err != nil {
		return f.err
	}
	f.confirms = append(f.confirms, factID)
	return nil
}

func TestApplyExtractedFactsAppliesSupportedActionsAndSkipsInvalidOnes(t *testing.T) {
	applier := &fakeFactApplier{}
	facts := []ExtractedFact{
		{Action: "create", Category: "学习", Key: "语言", Value: "Go"},
		{Action: "update", Category: "偏好", Key: "回答风格", Value: "简洁"},
		{Action: "supersede", Category: "学习", Key: "语言", Value: "Rust", TargetID: "old-fact", Reason: "用户明确更正"},
		{Action: "confirm", TargetID: "confirmed-fact"},
		{Action: "confirm"},
		{Action: "supersede", Key: "missing-target"},
		{Action: "unknown"},
	}

	got, err := applyExtractedFacts(context.Background(), applier, "user-1", "session-1", facts)
	if err != nil {
		t.Fatalf("applyExtractedFacts returned error: %v", err)
	}

	if got.Applied != 4 || got.Skipped != 3 {
		t.Fatalf("ApplyResult = %#v, want applied=4 skipped=3", got)
	}
	if len(applier.upserts) != 3 {
		t.Fatalf("upserts = %#v, want 3", applier.upserts)
	}
	if !reflect.DeepEqual(applier.supersedes, []supersedeCall{{oldID: "old-fact", newID: "语言-id", reason: "用户明确更正"}}) {
		t.Fatalf("supersedes = %#v", applier.supersedes)
	}
	if !reflect.DeepEqual(applier.confirms, []string{"confirmed-fact"}) {
		t.Fatalf("confirms = %#v", applier.confirms)
	}
}

func TestApplyExtractedFactsStopsOnAdapterError(t *testing.T) {
	applier := &fakeFactApplier{err: errors.New("database down")}

	got, err := applyExtractedFacts(context.Background(), applier, "user-1", "session-1", []ExtractedFact{
		{Action: "create", Category: "学习", Key: "语言", Value: "Go"},
	})

	if err == nil {
		t.Fatal("expected error")
	}
	if got.Applied != 0 || got.Skipped != 0 {
		t.Fatalf("ApplyResult = %#v, want zero result on first error", got)
	}
}

func TestMemoryFactOrderByQuotesReservedKeyColumn(t *testing.T) {
	capture := &sqlCaptureLogger{}
	db, err := gorm.Open(gormmysql.New(gormmysql.Config{
		DSN:                       "user:pass@tcp(127.0.0.1:3306)/vidwise?charset=utf8mb4&parseTime=True&loc=Local",
		SkipInitializeWithVersion: true,
	}), &gorm.Config{DryRun: true, DisableAutomaticPing: true, Logger: capture})
	if err != nil {
		t.Fatalf("open dry-run db: %v", err)
	}

	if _, err := NewRepo(db).GetFactsByUser(context.Background(), "user-1"); err != nil {
		t.Fatalf("get facts by user: %v", err)
	}

	sql := capture.sql
	if !strings.Contains(sql, "ORDER BY `category`,`key`") {
		t.Fatalf("generated SQL = %q, want quoted key column in ORDER BY", sql)
	}
	if strings.Contains(sql, " key ") || strings.Contains(sql, " key,") {
		t.Fatalf("generated SQL = %q, contains unquoted key column", sql)
	}
}

type sqlCaptureLogger struct {
	sql string
}

func (l *sqlCaptureLogger) LogMode(gormlogger.LogLevel) gormlogger.Interface {
	return l
}

func (l *sqlCaptureLogger) Info(context.Context, string, ...any) {}

func (l *sqlCaptureLogger) Warn(context.Context, string, ...any) {}

func (l *sqlCaptureLogger) Error(context.Context, string, ...any) {}

func (l *sqlCaptureLogger) Trace(_ context.Context, _ time.Time, fc func() (string, int64), _ error) {
	sql, _ := fc()
	l.sql = sql
}
