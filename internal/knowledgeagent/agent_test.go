package knowledgeagent

import (
	"context"
	"errors"
	"strings"
	"testing"

	einomodel "github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/chat"
	"github.com/raiki02/vidwise/internal/chatagent"
	"github.com/raiki02/vidwise/internal/rag"
)

func TestTurnVideoShareCreatesPendingProcessAction(t *testing.T) {
	classifier := &fakeIntentClassifier{}
	service := NewService(ServiceConfig{
		Sessions:         newFakeSessionStore(),
		Actions:          NewActionStore(ActionStoreOptions{}),
		IntentClassifier: classifier,
	})

	shareText := `54 【满身蜱虫痒到发疯！非洲水牛下河，万条小鱼集体上门搓澡 - 狂野星球说 | 小红书 - 你的生活兴趣社区】 😆 R2mLHAQeGI3Satn 😆 https://www.xiaohongshu.com/discovery/item/6a4cc74500000000150279a4?source=webshare&xhsshare=pc_web&xsec_token=ABYkvtp8_gX1Tai24ZgtvB7r9gvNP1yaSHYWH10TE-Zv0=&xsec_source=pc_share,`
	got, err := service.Turn(context.Background(), TurnRequest{UserID: "u1", Message: shareText})
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}

	if got.SessionID != "session-1" {
		t.Fatalf("session id = %q, want session-1", got.SessionID)
	}
	if len(got.PendingActions) != 1 {
		t.Fatalf("pending actions = %#v, want one action", got.PendingActions)
	}
	action := got.PendingActions[0]
	if action.Type != ActionProcessVideo || !action.RequiresConfirmation || action.RiskLevel != RiskExpensive {
		t.Fatalf("action = %#v, want expensive confirmed process_video", action)
	}
	if action.Input["url"] != "https://www.xiaohongshu.com/discovery/item/6a4cc74500000000150279a4?source=webshare&xhsshare=pc_web&xsec_token=ABYkvtp8_gX1Tai24ZgtvB7r9gvNP1yaSHYWH10TE-Zv0=&xsec_source=pc_share" {
		t.Fatalf("action URL = %#v", action.Input["url"])
	}
	if action.Input["name"] != "满身蜱虫痒到发疯_非洲水牛下河_万条小鱼集体上门搓澡_-_狂野星球说" {
		t.Fatalf("action name = %#v", action.Input["name"])
	}
	if len(got.ExecutedActions) != 0 {
		t.Fatalf("executed actions = %#v, want none", got.ExecutedActions)
	}
	if classifier.calls != 0 {
		t.Fatalf("classifier calls = %d, want 0 for deterministic video URL", classifier.calls)
	}
}

func TestTurnKnowledgeQuestionExecutesReadOnlyAnswer(t *testing.T) {
	answerer := &fakeAnswerRunner{
		result: chatagent.TurnResult{
			Answer: "来自知识库的回答",
			Evaluation: chatagent.RetrievalEvaluation{
				ShouldRetrieve: true,
				Reason:         "knowledge_base_question",
			},
			Retrieval: chatagent.RetrievalOutcome{
				Status:     chatagent.RetrievalStatusRetrieved,
				Query:      "视频讲了什么",
				Queries:    []string{"视频讲了什么"},
				ChunkCount: 1,
			},
			Chunks: []rag.RelevantChunk{{Text: "retrieved chunk", SourceID: "source-1"}},
			RAGContext: chatagent.RAGContextOutcome{
				UsedChunks: 1,
				Citations:  []rag.ContextCitation{{SnippetNumber: 1, Chunk: rag.RelevantChunk{Text: "retrieved chunk", SourceID: "source-1"}}},
			},
			Grounding: chatagent.AnswerGroundingOutcome{
				Status:           chatagent.AnswerGroundingGrounded,
				CitationRequired: true,
				HasCitations:     true,
				CitedSnippets:    []int{1},
			},
		},
	}
	service := NewService(ServiceConfig{Answerer: answerer})

	got, err := service.Turn(context.Background(), TurnRequest{
		UserID:      "u1",
		SessionID:   "s1",
		Message:     "视频讲了什么",
		SourceIDs:   []string{"source-1"},
		DocumentIDs: []string{"doc-1"},
	})
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}

	if got.Answer != "来自知识库的回答" {
		t.Fatalf("answer = %q", got.Answer)
	}
	if len(got.ExecutedActions) != 1 || got.ExecutedActions[0].Type != ActionAnswerFromKnowledge {
		t.Fatalf("executed actions = %#v, want answer_from_knowledge", got.ExecutedActions)
	}
	if got.RAGTrace == nil || got.RAGTrace.Status != string(chatagent.RetrievalStatusRetrieved) || got.RAGTrace.ContextUsedChunks != 1 {
		t.Fatalf("rag trace = %#v", got.RAGTrace)
	}
	if strings.Join(answerer.req.SourceIDs, ",") != "source-1" || strings.Join(answerer.req.DocumentIDs, ",") != "doc-1" {
		t.Fatalf("answer request filters = %#v", answerer.req)
	}
}

func TestTurnListSourcesExecutesReadOnlyAction(t *testing.T) {
	sources := &fakeSourceService{
		sources: []rag.SourceSummary{{SourceID: "source-1", SourceName: "demo.txt", ChunkCount: 3}},
	}
	service := NewService(ServiceConfig{Sources: sources})

	got, err := service.Turn(context.Background(), TurnRequest{UserID: "u1", SessionID: "s1", Message: "列出知识库 sources"})
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}

	if len(got.ExecutedActions) != 1 || got.ExecutedActions[0].Type != ActionListSources {
		t.Fatalf("executed actions = %#v, want list_sources", got.ExecutedActions)
	}
	if !strings.Contains(got.Answer, "demo.txt") {
		t.Fatalf("answer = %q, want source name", got.Answer)
	}
	if sources.listReq.Filter == nil || sources.listReq.Filter.UserID != "u1" || sources.listReq.Filter.SessionID != "s1" {
		t.Fatalf("list filter = %#v", sources.listReq.Filter)
	}
}

func TestTurnLLMClassifierCanCreatePendingFormatAction(t *testing.T) {
	classifier := &fakeIntentClassifier{
		result: IntentClassifyResult{
			Type: ActionFormatText,
			Text: "第一句。第二句。",
		},
	}
	service := NewService(ServiceConfig{
		Actions:          NewActionStore(ActionStoreOptions{}),
		IntentClassifier: classifier,
	})

	got, err := service.Turn(context.Background(), TurnRequest{UserID: "u1", SessionID: "s1", Message: "请帮我把下面内容变得更像文章：第一句。第二句。"})
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}

	if classifier.calls != 1 {
		t.Fatalf("classifier calls = %d, want 1", classifier.calls)
	}
	if len(got.PendingActions) != 1 {
		t.Fatalf("pending actions = %#v, want one", got.PendingActions)
	}
	action := got.PendingActions[0]
	if action.Type != ActionFormatText || !action.RequiresConfirmation || action.RiskLevel != RiskExpensive {
		t.Fatalf("action = %#v, want expensive format_text", action)
	}
	if action.Input["text"] != "第一句。第二句。" {
		t.Fatalf("text input = %#v", action.Input["text"])
	}
}

func TestTurnDeleteSourceRequiresDestructiveConfirmation(t *testing.T) {
	service := NewService(ServiceConfig{Actions: NewActionStore(ActionStoreOptions{})})

	got, err := service.Turn(context.Background(), TurnRequest{
		UserID:    "u1",
		SessionID: "s1",
		Message:   "删除这个 source",
		SourceIDs: []string{"source-1"},
	})
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}
	if len(got.PendingActions) != 1 {
		t.Fatalf("pending actions = %#v, want one", got.PendingActions)
	}
	action := got.PendingActions[0]
	if action.Type != ActionDeleteSource || action.RiskLevel != RiskDestructive || !action.RequiresConfirmation {
		t.Fatalf("action = %#v, want destructive delete_source", action)
	}
}

func TestTurnDeleteSourceParsesStableHashSourceID(t *testing.T) {
	service := NewService(ServiceConfig{Actions: NewActionStore(ActionStoreOptions{})})
	sourceID := "2a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f70819"

	got, err := service.Turn(context.Background(), TurnRequest{
		UserID:    "u1",
		SessionID: "s1",
		Message:   "请删除 source_id: " + sourceID,
	})
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}

	if len(got.PendingActions) != 1 {
		t.Fatalf("pending actions = %#v, want one", got.PendingActions)
	}
	action := got.PendingActions[0]
	if action.Type != ActionDeleteSource || action.RiskLevel != RiskDestructive || !action.RequiresConfirmation {
		t.Fatalf("action = %#v, want destructive delete_source", action)
	}
	sourceIDs, ok := action.Input["source_ids"].([]string)
	if !ok || len(sourceIDs) != 1 || sourceIDs[0] != sourceID {
		t.Fatalf("source_ids input = %#v, want %s", action.Input["source_ids"], sourceID)
	}
}

func TestTurnEmptyInputReturnsValidationError(t *testing.T) {
	service := NewService(ServiceConfig{})

	if _, err := service.Turn(context.Background(), TurnRequest{UserID: "", Message: "hi"}); err == nil {
		t.Fatal("expected missing user_id error")
	}
	if _, err := service.Turn(context.Background(), TurnRequest{UserID: "u1", Message: "   "}); err == nil {
		t.Fatal("expected missing message error")
	}
}

func TestTurnAmbiguousInputFallsBackToKnowledgeAnswer(t *testing.T) {
	answerer := &fakeAnswerRunner{
		result: chatagent.TurnResult{Answer: "普通知识库回答"},
	}
	service := NewService(ServiceConfig{
		Answerer:         answerer,
		IntentClassifier: &fakeIntentClassifier{result: IntentClassifyResult{Type: ActionAnswerFromKnowledge}},
	})

	got, err := service.Turn(context.Background(), TurnRequest{UserID: "u1", SessionID: "s1", Message: "这个呢？"})
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}

	if got.Answer != "普通知识库回答" {
		t.Fatalf("answer = %q, want knowledge answer", got.Answer)
	}
	if len(got.PendingActions) != 0 {
		t.Fatalf("pending actions = %#v, want none", got.PendingActions)
	}
	if len(got.ExecutedActions) != 1 || got.ExecutedActions[0].Type != ActionAnswerFromKnowledge {
		t.Fatalf("executed actions = %#v, want answer_from_knowledge", got.ExecutedActions)
	}
	if answerer.req.Query != "这个呢？" {
		t.Fatalf("answer query = %q", answerer.req.Query)
	}
}

func TestLLMIntentClassifierParsesModelDecision(t *testing.T) {
	enabled := true
	model := &fakeIntentModel{
		response: `判断如下：{"action_type":"list_sources","confidence":0.82,"reason":"用户想看知识库 sources"}`,
	}
	classifier := NewLLMIntentClassifierWithModelFactory(
		appconfig.LLMConfig{Enabled: &enabled, Model: "qwen"},
		func(context.Context, appconfig.LLMConfig) (chatagent.ChatModel, error) {
			return model, nil
		},
	)

	got, err := classifier.ClassifyIntent(context.Background(), IntentClassifyRequest{
		UserID:      "u1",
		SessionID:   "s1",
		Message:     "能看看我现在有哪些资料吗？",
		SourceIDs:   []string{"source-1"},
		DocumentIDs: []string{"doc-1"},
	})
	if err != nil {
		t.Fatalf("ClassifyIntent: %v", err)
	}

	if got.Type != ActionListSources || got.Confidence != 0.82 {
		t.Fatalf("classification = %#v, want list_sources", got)
	}
	if len(model.input) != 2 {
		t.Fatalf("model input = %#v, want system and user messages", model.input)
	}
	prompt := model.input[1].Content
	for _, want := range []string{"source-1", "doc-1", "能看看我现在有哪些资料吗"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt missing %q:\n%s", want, prompt)
		}
	}
}

func TestTurnIndexTaskRequiresConfirmation(t *testing.T) {
	service := NewService(ServiceConfig{Actions: NewActionStore(ActionStoreOptions{})})

	got, err := service.Turn(context.Background(), TurnRequest{UserID: "u1", SessionID: "s1", Message: "把 task-1 存入知识库"})
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}
	if len(got.PendingActions) != 1 {
		t.Fatalf("pending actions = %#v, want one", got.PendingActions)
	}
	action := got.PendingActions[0]
	if action.Type != ActionIndexTaskTranscript || action.RiskLevel != RiskExpensive || !action.RequiresConfirmation {
		t.Fatalf("action = %#v, want expensive index_task_transcript", action)
	}
	if action.Input["task_id"] != "task-1" {
		t.Fatalf("task id input = %#v", action.Input["task_id"])
	}
}

func TestConfirmProcessVideoExecutesOnlyOnce(t *testing.T) {
	videos := &fakeVideoProcessor{result: VideoProcessResult{TaskID: "task-1", Status: "pending", SessionID: "s1", TraceID: "trace-1"}}
	service := NewService(ServiceConfig{
		Videos:  videos,
		Actions: NewActionStore(ActionStoreOptions{}),
	})
	turn, err := service.Turn(context.Background(), TurnRequest{UserID: "u1", SessionID: "s1", Message: "https://example.com/video"})
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}
	actionID := turn.PendingActions[0].ID

	got, err := service.Confirm(context.Background(), actionID, ConfirmRequest{UserID: "u1", TraceID: "trace-confirm"})
	if err != nil {
		t.Fatalf("Confirm: %v", err)
	}
	if len(got.TaskIDs) != 1 || got.TaskIDs[0] != "task-1" {
		t.Fatalf("task ids = %#v, want task-1", got.TaskIDs)
	}
	if videos.calls != 1 {
		t.Fatalf("video calls = %d, want 1", videos.calls)
	}

	_, err = service.Confirm(context.Background(), actionID, ConfirmRequest{UserID: "u1"})
	if !errors.Is(err, ErrActionAlreadyExecuted) {
		t.Fatalf("second confirm err = %v, want ErrActionAlreadyExecuted", err)
	}
	if videos.calls != 1 {
		t.Fatalf("video calls after second confirm = %d, want 1", videos.calls)
	}
}

func TestConfirmPersistsResultInActionSession(t *testing.T) {
	store := newFakeSessionStore()
	videos := &fakeVideoProcessor{result: VideoProcessResult{TaskID: "task-1", Status: "pending", SessionID: "s1"}}
	service := NewService(ServiceConfig{
		Sessions: store,
		Videos:   videos,
		Actions:  NewActionStore(ActionStoreOptions{}),
	})
	turn, err := service.Turn(context.Background(), TurnRequest{UserID: "u1", SessionID: "s1", Message: "https://example.com/video"})
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}
	action := turn.PendingActions[0]
	if action.Input["session_id"] != "s1" {
		t.Fatalf("action session_id = %#v, want s1", action.Input["session_id"])
	}

	if _, err := service.Confirm(context.Background(), action.ID, ConfirmRequest{UserID: "u1"}); err != nil {
		t.Fatalf("Confirm: %v", err)
	}

	messages := store.messages["s1"]
	if len(messages) != 3 {
		t.Fatalf("stored messages = %#v, want user, pending assistant, confirm assistant", messages)
	}
	last := messages[len(messages)-1]
	if last.Role != "assistant" || !strings.Contains(last.Content, "已开始处理视频") {
		t.Fatalf("last message = %#v, want confirmation result assistant message", last)
	}
}

type fakeSessionStore struct {
	nextID   string
	messages map[string][]chat.Message
}

func newFakeSessionStore() *fakeSessionStore {
	return &fakeSessionStore{nextID: "session-1", messages: map[string][]chat.Message{}}
}

func (s *fakeSessionStore) CreateSessionForUser(_ context.Context, userID, title string) (*chat.Session, error) {
	return &chat.Session{ID: s.nextID, UserID: userID, Title: title}, nil
}

func (s *fakeSessionStore) AddMessage(_ context.Context, sessionID, role, content string) (*chat.Message, error) {
	msg := chat.Message{SessionID: sessionID, Role: role, Content: content}
	s.messages[sessionID] = append(s.messages[sessionID], msg)
	return &msg, nil
}

func (s *fakeSessionStore) GetRecentMessages(_ context.Context, sessionID string, limit int) ([]chat.Message, error) {
	messages := s.messages[sessionID]
	if limit > 0 && len(messages) > limit {
		messages = messages[len(messages)-limit:]
	}
	out := make([]chat.Message, len(messages))
	copy(out, messages)
	return out, nil
}

type fakeAnswerRunner struct {
	req    chatagent.TurnRequest
	result chatagent.TurnResult
	err    error
}

type fakeIntentClassifier struct {
	req    IntentClassifyRequest
	result IntentClassifyResult
	err    error
	calls  int
}

func (c *fakeIntentClassifier) ClassifyIntent(_ context.Context, req IntentClassifyRequest) (IntentClassifyResult, error) {
	c.calls++
	c.req = req
	if c.err != nil {
		return IntentClassifyResult{}, c.err
	}
	return c.result, nil
}

type fakeIntentModel struct {
	response string
	input    []*schema.Message
}

func (m *fakeIntentModel) Generate(_ context.Context, input []*schema.Message, _ ...einomodel.Option) (*schema.Message, error) {
	m.input = input
	return schema.AssistantMessage(m.response, nil), nil
}

func (r *fakeAnswerRunner) RunTurn(_ context.Context, req chatagent.TurnRequest) (chatagent.TurnResult, error) {
	r.req = req
	if r.err != nil {
		return chatagent.TurnResult{}, r.err
	}
	return r.result, nil
}

type fakeSourceService struct {
	listReq   rag.SourceListRequest
	deleteReq rag.DeleteRequest
	sources   []rag.SourceSummary
	delete    rag.DeleteResult
	err       error
}

func (s *fakeSourceService) ListSources(_ context.Context, req rag.SourceListRequest) ([]rag.SourceSummary, error) {
	s.listReq = req
	if s.err != nil {
		return nil, s.err
	}
	return s.sources, nil
}

func (s *fakeSourceService) DeleteSourcesWithOptions(_ context.Context, req rag.DeleteRequest) (rag.DeleteResult, error) {
	s.deleteReq = req
	if s.err != nil {
		return rag.DeleteResult{}, s.err
	}
	return s.delete, nil
}

type fakeVideoProcessor struct {
	req    VideoProcessRequest
	result VideoProcessResult
	err    error
	calls  int
}

func (p *fakeVideoProcessor) StartVideoProcess(_ context.Context, req VideoProcessRequest, traceID string) (VideoProcessResult, error) {
	p.calls++
	p.req = req
	if p.err != nil {
		return VideoProcessResult{}, p.err
	}
	if p.result.TraceID == "" {
		p.result.TraceID = traceID
	}
	return p.result, nil
}
