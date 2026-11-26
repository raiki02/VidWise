package tool

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/raiki02/vidwise/internal/asr"
	"github.com/raiki02/vidwise/internal/vectorstore"
)

type fakeTranscriber struct {
	resp asr.TranscribeResponse
	err  error
}

func (f *fakeTranscriber) Transcribe(_ context.Context, _ string, _ string) (asr.TranscribeResponse, error) {
	return f.resp, f.err
}

type fakeEmbedder struct {
	vectors [][]float64
	err     error
	last    []string
}

func (f *fakeEmbedder) Embed(_ context.Context, texts []string) ([][]float64, error) {
	f.last = texts
	if f.err != nil {
		return nil, f.err
	}
	out := make([][]float64, len(texts))
	for i := range texts {
		out[i] = []float64{1.0, 0.0}
	}
	if f.vectors != nil {
		out = f.vectors
	}
	return out, nil
}

type fakeStore struct {
	chunks    []vectorstore.Chunk
	vectors   [][]float64
	sourceID  string
	searchHit []vectorstore.Hit
	searchVec []float32
	searchK   uint64
	err       error
}

func (f *fakeStore) ReplaceSource(_ context.Context, sourceID string, chunks []vectorstore.Chunk, vectors [][]float64) error {
	if f.err != nil {
		return f.err
	}
	f.sourceID = sourceID
	f.chunks = chunks
	f.vectors = vectors
	return nil
}

func (f *fakeStore) Search(_ context.Context, vector []float32, topK uint64) ([]vectorstore.Hit, error) {
	if f.err != nil {
		return nil, f.err
	}
	f.searchVec = vector
	f.searchK = topK
	return f.searchHit, nil
}

func TestTranscribeToolWritesFile(t *testing.T) {
	dir := t.TempDir()
	audio := filepath.Join(dir, "a.mp3")
	if err := os.WriteFile(audio, []byte("fake"), 0o644); err != nil {
		t.Fatal(err)
	}
	tr := &fakeTranscriber{resp: asr.TranscribeResponse{Text: strings.Repeat("你好", 600), Duration: 10}}
	tool, err := NewTranscribeTool(dir, tr)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := tool.InvokableRun(context.Background(), `{"audio_path":"`+audio+`"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(raw, "transcript_path") || !strings.Contains(raw, "1200") {
		t.Errorf("output missing expected fields: %s", raw)
	}
	text, err := os.ReadFile(filepath.Join(dir, "a.txt"))
	if err != nil || len(text) == 0 {
		t.Fatalf("transcript file not written: %v", err)
	}
}

func TestTranscribeToolRejectsPathOutside(t *testing.T) {
	dir := t.TempDir()
	tool, err := NewTranscribeTool(dir, &fakeTranscriber{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tool.InvokableRun(context.Background(), `{"audio_path":"/etc/hosts"}`); err == nil {
		t.Fatal("expected path escape error")
	}
}

func TestTranscribeToolEmptyTranscript(t *testing.T) {
	dir := t.TempDir()
	audio := filepath.Join(dir, "a.mp3")
	if err := os.WriteFile(audio, []byte("fake"), 0o644); err != nil {
		t.Fatal(err)
	}
	tool, err := NewTranscribeTool(dir, &fakeTranscriber{resp: asr.TranscribeResponse{Text: "   "}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tool.InvokableRun(context.Background(), `{"audio_path":"`+audio+`"}`); err == nil {
		t.Fatal("expected empty transcript error")
	}
}

func TestRAGIndexTool(t *testing.T) {
	dir := t.TempDir()
	transcript := filepath.Join(dir, "a.txt")
	long := strings.Repeat("这是转写内容。", 300) // 1800 runes
	if err := os.WriteFile(transcript, []byte(long), 0o644); err != nil {
		t.Fatal(err)
	}
	store := &fakeStore{}
	emb := &fakeEmbedder{}
	tool, err := NewRAGIndexTool(dir, 800, 100, emb, store)
	if err != nil {
		t.Fatal(err)
	}
	in := `{"transcript_path":"` + transcript + `","source_name":"测试视频","source_url":"https://example.com/v"}`
	raw, err := tool.InvokableRun(context.Background(), in)
	if err != nil {
		t.Fatal(err)
	}
	if store.sourceID == "" || store.chunks == nil {
		t.Fatalf("store not called: %+v", store)
	}
	if len(store.chunks) < 2 {
		t.Errorf("expected multiple chunks, got %d", len(store.chunks))
	}
	if len(store.vectors) != len(store.chunks) {
		t.Errorf("vector count %d != chunk count %d", len(store.vectors), len(store.chunks))
	}
	if !strings.Contains(raw, "chunks") {
		t.Errorf("unexpected output: %s", raw)
	}
}

func TestRAGIndexToolEmptyTranscript(t *testing.T) {
	dir := t.TempDir()
	transcript := filepath.Join(dir, "a.txt")
	if err := os.WriteFile(transcript, []byte("   "), 0o644); err != nil {
		t.Fatal(err)
	}
	tool, err := NewRAGIndexTool(dir, 800, 100, &fakeEmbedder{}, &fakeStore{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tool.InvokableRun(context.Background(), `{"transcript_path":"`+transcript+`","source_name":"x"}`); err == nil {
		t.Fatal("expected empty transcript error")
	}
}

func TestRAGQueryTool(t *testing.T) {
	store := &fakeStore{searchHit: []vectorstore.Hit{
		{Chunk: vectorstore.Chunk{Text: "片段一", SourceName: "v1", SourceURL: "https://e.com/1"}, Score: 0.9},
		{Chunk: vectorstore.Chunk{Text: "片段二", SourceName: "v1"}, Score: 0.8},
	}}
	tool, err := NewRAGQueryTool(&fakeEmbedder{}, store, 5)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := tool.InvokableRun(context.Background(), `{"query":"讲了什么","top_k":20}`)
	if err != nil {
		t.Fatal(err)
	}
	if store.searchK != 20 {
		t.Errorf("top_k = %d, want 20", store.searchK)
	}
	if !strings.Contains(raw, "片段一") || !strings.Contains(raw, "source_name") {
		t.Errorf("unexpected output: %s", raw)
	}
}

func TestRAGQueryToolClampsTopK(t *testing.T) {
	store := &fakeStore{}
	tool, err := NewRAGQueryTool(&fakeEmbedder{}, store, 5)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tool.InvokableRun(context.Background(), `{"query":"x","top_k":99}`); err != nil {
		t.Fatal(err)
	}
	if store.searchK != 20 {
		t.Errorf("top_k = %d, want clamped 20", store.searchK)
	}
}
