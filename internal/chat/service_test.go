package chat

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/cloudwego/eino/schema"
)

type fakeGenerator struct {
	reply string
	err   error
	got   []*schema.Message
}

func (f *fakeGenerator) Generate(_ context.Context, messages []*schema.Message) (*schema.Message, error) {
	f.got = messages
	if f.err != nil {
		return nil, f.err
	}
	return &schema.Message{Role: schema.Assistant, Content: f.reply}, nil
}

func TestChatCreatesSession(t *testing.T) {
	store := NewMemoryStore()
	gen := &fakeGenerator{reply: "回答"}
	svc := NewService(store, gen, 20)

	sessionID, reply, err := svc.Chat(context.Background(), "", "你好")
	if err != nil {
		t.Fatal(err)
	}
	if sessionID == "" || reply != "回答" {
		t.Fatalf("sessionID=%q reply=%q", sessionID, reply)
	}
	msgs, err := store.RecentMessages(context.Background(), sessionID, 20)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 2 || msgs[0].Role != "user" || msgs[1].Role != "assistant" {
		t.Fatalf("messages = %+v", msgs)
	}
}

func TestChatRejectsUnknownSession(t *testing.T) {
	svc := NewService(NewMemoryStore(), &fakeGenerator{reply: "x"}, 20)
	_, _, err := svc.Chat(context.Background(), "no-such-id", "hi")
	if !errors.Is(err, ErrSessionNotFound) {
		t.Fatalf("err = %v, want ErrSessionNotFound", err)
	}
}

func TestChatRejectsEmptyMessage(t *testing.T) {
	svc := NewService(NewMemoryStore(), &fakeGenerator{}, 20)
	if _, _, err := svc.Chat(context.Background(), "", "  "); err == nil {
		t.Fatal("expected error for empty message")
	}
}

func TestChatFeedsHistory(t *testing.T) {
	store := NewMemoryStore()
	id, _ := store.CreateSession(context.Background(), "t")
	_ = store.AddMessage(context.Background(), id, "user", "第一个问题")
	_ = store.AddMessage(context.Background(), id, "assistant", "第一个回答")

	gen := &fakeGenerator{reply: "第二个回答"}
	svc := NewService(store, gen, 20)
	if _, _, err := svc.Chat(context.Background(), id, "第二个问题"); err != nil {
		t.Fatal(err)
	}
	if len(gen.got) != 3 {
		t.Fatalf("model saw %d messages, want 3: %+v", len(gen.got), gen.got)
	}
	if gen.got[0].Content != "第一个问题" || gen.got[1].Content != "第一个回答" || gen.got[2].Role != schema.User {
		t.Fatalf("unexpected inputs: %+v", gen.got)
	}
}

func TestChatDoesNotPersistOnFailure(t *testing.T) {
	store := NewMemoryStore()
	gen := &fakeGenerator{err: errors.New("model down")}
	svc := NewService(store, gen, 20)

	_, _, err := svc.Chat(context.Background(), "", "会失败")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "model down") {
		t.Fatalf("err = %v", err)
	}
	// The session is created but no user/assistant messages are stored.
	found := false
	for id := range store.sessions {
		found = id != ""
		if len(store.messages[id]) != 0 {
			t.Fatalf("messages persisted on failure: %+v", store.messages[id])
		}
	}
	if !found {
		t.Fatal("session not created")
	}
}

func TestTitleFor(t *testing.T) {
	if got := TitleFor(""); got != "新对话" {
		t.Errorf("TitleFor empty = %q", got)
	}
	long := strings.Repeat("好", 50)
	if got := TitleFor(long); len([]rune(got)) != 30 {
		t.Errorf("TitleFor long = %q (%d)", got, len([]rune(got)))
	}
}
