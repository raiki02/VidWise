package chatagent

import (
	"context"
	"errors"
	"strings"
	"testing"

	einomodel "github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/rag"
)

type fakeModel struct {
	response string
	err      error
	input    []*schema.Message
}

func (m *fakeModel) Generate(_ context.Context, input []*schema.Message, _ ...einomodel.Option) (*schema.Message, error) {
	m.input = input
	if m.err != nil {
		return nil, m.err
	}
	return schema.AssistantMessage(m.response, nil), nil
}

type fakeRetriever struct {
	chunks []rag.RelevantChunk
	err    error
	req    rag.RetrieveRequest
	calls  int
}

func (r *fakeRetriever) RetrieveWithOptions(_ context.Context, req rag.RetrieveRequest) ([]rag.RelevantChunk, error) {
	r.calls++
	r.req = req
	if r.err != nil {
		return nil, r.err
	}
	return r.chunks, nil
}

func TestEvaluateRetrievalMarksKnowledgeQueryWhenRetrieverUnavailable(t *testing.T) {
	calls := 0
	agent := NewWithModelFactory(enabledLLMConfig(), rag.ContextConfig{}, func(context.Context, appconfig.LLMConfig) (ChatModel, error) {
		calls++
		return &fakeModel{response: `{"should_retrieve": true, "reason": "should not run"}`}, nil
	})

	got := agent.EvaluateRetrieval(context.Background(), RetrievalEvaluationRequest{
		Query:              "视频里讲了什么？",
		RetrieverAvailable: false,
	})

	if !got.ShouldRetrieve {
		t.Fatalf("expected knowledge-base query to still require retrieval, got %#v", got)
	}
	if got.Reason != "retriever_unavailable" {
		t.Fatalf("reason = %q, want retriever_unavailable", got.Reason)
	}
	if calls != 0 {
		t.Fatalf("model factory called %d times, want 0", calls)
	}
}

func TestEvaluateRetrievalSkipsSmallTalkWhenRetrieverUnavailable(t *testing.T) {
	agent := NewWithModelFactory(enabledLLMConfig(), rag.ContextConfig{}, func(context.Context, appconfig.LLMConfig) (ChatModel, error) {
		return &fakeModel{response: `{"should_retrieve": true, "reason": "should not run"}`}, nil
	})

	got := agent.EvaluateRetrieval(context.Background(), RetrievalEvaluationRequest{
		Query:              "你好",
		RetrieverAvailable: false,
	})

	if got.ShouldRetrieve {
		t.Fatalf("expected small talk to skip retrieval, got %#v", got)
	}
	if got.Reason != "retriever_unavailable" {
		t.Fatalf("reason = %q, want retriever_unavailable", got.Reason)
	}
}

func TestEvaluateRetrievalSkipsGenericSummaryWhenRetrieverUnavailable(t *testing.T) {
	agent := NewWithModelFactory(enabledLLMConfig(), rag.ContextConfig{}, func(context.Context, appconfig.LLMConfig) (ChatModel, error) {
		return &fakeModel{response: `{"should_retrieve": true, "reason": "should not run"}`}, nil
	})

	got := agent.EvaluateRetrieval(context.Background(), RetrievalEvaluationRequest{
		Query:              "帮我总结一下这段 Go 代码",
		RetrieverAvailable: false,
	})

	if got.ShouldRetrieve {
		t.Fatalf("expected generic summary request to skip retrieval, got %#v", got)
	}
	if got.Reason != "retriever_unavailable" {
		t.Fatalf("reason = %q, want retriever_unavailable", got.Reason)
	}
}

func TestEvaluateRetrievalDefaultsToRetrieveWhenLLMDisabled(t *testing.T) {
	agent := New(disabledLLMConfig(), rag.ContextConfig{})

	got := agent.EvaluateRetrieval(context.Background(), RetrievalEvaluationRequest{
		Query:              "查一下上传文档里的安装步骤",
		RetrieverAvailable: true,
	})

	if !got.ShouldRetrieve {
		t.Fatalf("expected conservative retrieval, got %#v", got)
	}
	if got.Reason != "llm_unavailable" {
		t.Fatalf("reason = %q, want llm_unavailable", got.Reason)
	}
}

func TestEvaluateRetrievalParsesModelDecisionAndIncludesContext(t *testing.T) {
	model := &fakeModel{response: `当然可以：{"should_retrieve": false, "reason": "small_talk"}`}
	agent := NewWithModelFactory(enabledLLMConfig(), rag.ContextConfig{}, func(context.Context, appconfig.LLMConfig) (ChatModel, error) {
		return model, nil
	})

	got := agent.EvaluateRetrieval(context.Background(), RetrievalEvaluationRequest{
		Query:              "你好",
		History:            "用户: 上次的问题\n助手: 上次的回答",
		UserFacts:          "用户画像: 喜欢简洁回答",
		RetrieverAvailable: true,
	})

	if got.ShouldRetrieve {
		t.Fatalf("expected model decision to skip retrieval, got %#v", got)
	}
	if got.Reason != "small_talk" {
		t.Fatalf("reason = %q, want small_talk", got.Reason)
	}
	if len(model.input) != 2 {
		t.Fatalf("expected system and user messages, got %#v", model.input)
	}
	userPrompt := model.input[1].Content
	for _, want := range []string{"最近对话", "用户画像信息", "你好"} {
		if !strings.Contains(userPrompt, want) {
			t.Fatalf("retrieval prompt missing %q:\n%s", want, userPrompt)
		}
	}
}

func TestEvaluateRetrievalFallsBackToRetrieveOnModelFailure(t *testing.T) {
	agent := NewWithModelFactory(enabledLLMConfig(), rag.ContextConfig{}, func(context.Context, appconfig.LLMConfig) (ChatModel, error) {
		return &fakeModel{err: errors.New("model down")}, nil
	})

	got := agent.EvaluateRetrieval(context.Background(), RetrievalEvaluationRequest{
		Query:              "视频里讲了什么？",
		RetrieverAvailable: true,
	})

	if !got.ShouldRetrieve {
		t.Fatalf("expected conservative retrieval on model failure, got %#v", got)
	}
	if got.Reason != "eval_gen_failed" {
		t.Fatalf("reason = %q, want eval_gen_failed", got.Reason)
	}
}

func TestRunTurnRetrievesWithChatScopeAndFallbackAnswer(t *testing.T) {
	retriever := &fakeRetriever{
		chunks: []rag.RelevantChunk{
			{
				Text:       "视频讲到了 Markdown RAG。",
				Score:      0.93,
				SourceName: "guide.md",
			},
		},
	}
	agent := NewWithRetriever(disabledLLMConfig(), rag.ContextConfig{MaxRunes: 1024}, retriever)

	got, err := agent.RunTurn(context.Background(), TurnRequest{
		Query:     " 视频里讲了什么？ ",
		UserID:    " u1 ",
		SessionID: " s1 ",
	})
	if err != nil {
		t.Fatalf("RunTurn returned error: %v", err)
	}

	if !got.Evaluation.ShouldRetrieve || got.Evaluation.Reason != "llm_unavailable" {
		t.Fatalf("expected conservative retrieval evaluation, got %#v", got.Evaluation)
	}
	if retriever.calls != 1 {
		t.Fatalf("retriever calls = %d, want 1", retriever.calls)
	}
	if retriever.req.Query != "视频里讲了什么？" {
		t.Fatalf("retriever query = %q", retriever.req.Query)
	}
	if retriever.req.Filter == nil || retriever.req.Filter.UserID != "u1" || retriever.req.Filter.SessionID != "" {
		t.Fatalf("unexpected retrieval filter: %#v", retriever.req.Filter)
	}
	if len(got.Chunks) != 1 {
		t.Fatalf("expected one retrieved chunk, got %#v", got.Chunks)
	}
	if got.Retrieval.Status != RetrievalStatusRetrieved || got.Retrieval.ChunkCount != 1 {
		t.Fatalf("unexpected retrieval outcome: %#v", got.Retrieval)
	}
	if !strings.Contains(got.Answer, "视频讲到了 Markdown RAG") {
		t.Fatalf("expected fallback answer to include retrieved context, got %q", got.Answer)
	}
}

func TestRunTurnRetrievesWithSessionScopeWhenUserMissing(t *testing.T) {
	retriever := &fakeRetriever{
		chunks: []rag.RelevantChunk{{Text: "session scoped chunk", Score: 0.9}},
	}
	agent := NewWithRetriever(disabledLLMConfig(), rag.ContextConfig{}, retriever)

	_, err := agent.RunTurn(context.Background(), TurnRequest{
		Query:     "查一下知识库",
		SessionID: " session-1 ",
	})
	if err != nil {
		t.Fatalf("RunTurn returned error: %v", err)
	}

	if retriever.req.Filter == nil || retriever.req.Filter.UserID != "" || retriever.req.Filter.SessionID != "session-1" {
		t.Fatalf("unexpected retrieval filter: %#v", retriever.req.Filter)
	}
}

func TestRunTurnSkipsRetrieverWhenEvaluationDeclines(t *testing.T) {
	retriever := &fakeRetriever{
		chunks: []rag.RelevantChunk{{Text: "should not be used", Score: 0.9}},
	}
	agent := NewWithModelFactoryAndRetriever(enabledLLMConfig(), rag.ContextConfig{}, func(context.Context, appconfig.LLMConfig) (ChatModel, error) {
		return &fakeModel{response: `{"should_retrieve": false, "reason": "small_talk"}`}, nil
	}, retriever)

	got, err := agent.RunTurn(context.Background(), TurnRequest{Query: "你好"})
	if err != nil {
		t.Fatalf("RunTurn returned error: %v", err)
	}

	if got.Evaluation.ShouldRetrieve || got.Evaluation.Reason != "small_talk" {
		t.Fatalf("expected classifier to skip retrieval, got %#v", got.Evaluation)
	}
	if retriever.calls != 0 {
		t.Fatalf("retriever calls = %d, want 0", retriever.calls)
	}
	if len(got.Chunks) != 0 {
		t.Fatalf("expected no chunks, got %#v", got.Chunks)
	}
	if got.Retrieval.Status != RetrievalStatusNotNeeded {
		t.Fatalf("retrieval status = %q, want not_needed", got.Retrieval.Status)
	}
}

func TestRunTurnFallsBackWhenRetrieverFails(t *testing.T) {
	retriever := &fakeRetriever{err: errors.New("qdrant down")}
	agent := NewWithRetriever(disabledLLMConfig(), rag.ContextConfig{}, retriever)

	got, err := agent.RunTurn(context.Background(), TurnRequest{Query: "查一下知识库", SessionID: "session-1"})
	if err != nil {
		t.Fatalf("RunTurn returned error: %v", err)
	}

	if retriever.calls != 1 {
		t.Fatalf("retriever calls = %d, want 1", retriever.calls)
	}
	if len(got.Chunks) != 0 {
		t.Fatalf("expected no chunks on retrieval failure, got %#v", got.Chunks)
	}
	if got.Retrieval.Status != RetrievalStatusFailed {
		t.Fatalf("retrieval status = %q, want failed", got.Retrieval.Status)
	}
	if !strings.Contains(got.Retrieval.Error, "qdrant down") {
		t.Fatalf("expected retrieval error to be recorded, got %#v", got.Retrieval)
	}
	if !strings.Contains(got.Answer, "没有在当前知识库范围内检索到足够相关的内容") {
		t.Fatalf("expected insufficient-context answer, got %q", got.Answer)
	}
}

func TestRunTurnRecordsNoResultsWhenRetrieverReturnsEmpty(t *testing.T) {
	retriever := &fakeRetriever{}
	agent := NewWithRetriever(disabledLLMConfig(), rag.ContextConfig{}, retriever)

	got, err := agent.RunTurn(context.Background(), TurnRequest{Query: "查一下上传文档", SessionID: "session-1"})
	if err != nil {
		t.Fatalf("RunTurn returned error: %v", err)
	}

	if retriever.calls != 1 {
		t.Fatalf("retriever calls = %d, want 1", retriever.calls)
	}
	if got.Retrieval.Status != RetrievalStatusNoResults {
		t.Fatalf("retrieval status = %q, want no_results", got.Retrieval.Status)
	}
	if got.Retrieval.ChunkCount != 0 {
		t.Fatalf("chunk count = %d, want 0", got.Retrieval.ChunkCount)
	}
	if !strings.Contains(got.Answer, "没有在当前知识库范围内检索到足够相关的内容") {
		t.Fatalf("expected insufficient-context answer, got %q", got.Answer)
	}
}

func TestRunTurnSkipsRetrieverWhenScopeMissing(t *testing.T) {
	retriever := &fakeRetriever{
		chunks: []rag.RelevantChunk{{Text: "should not be used", Score: 0.9}},
	}
	agent := NewWithRetriever(disabledLLMConfig(), rag.ContextConfig{}, retriever)

	got, err := agent.RunTurn(context.Background(), TurnRequest{Query: "查一下知识库"})
	if err != nil {
		t.Fatalf("RunTurn returned error: %v", err)
	}

	if retriever.calls != 0 {
		t.Fatalf("retriever calls = %d, want 0", retriever.calls)
	}
	if len(got.Chunks) != 0 {
		t.Fatalf("expected no chunks without scope, got %#v", got.Chunks)
	}
	if got.Retrieval.Status != RetrievalStatusScopeRequired {
		t.Fatalf("retrieval status = %q, want scope_required", got.Retrieval.Status)
	}
	if !strings.Contains(got.Answer, "没有在当前知识库范围内检索到足够相关的内容") {
		t.Fatalf("expected insufficient-context answer without scope, got %q", got.Answer)
	}
}

func TestRunTurnTreatsTypedNilRetrieverAsUnavailable(t *testing.T) {
	var retriever *rag.Retriever
	agent := NewWithRetriever(disabledLLMConfig(), rag.ContextConfig{}, retriever)

	got, err := agent.RunTurn(context.Background(), TurnRequest{Query: "查一下知识库"})
	if err != nil {
		t.Fatalf("RunTurn returned error: %v", err)
	}

	if !got.Evaluation.ShouldRetrieve {
		t.Fatalf("expected knowledge query to require unavailable retriever, got %#v", got.Evaluation)
	}
	if got.Evaluation.Reason != "retriever_unavailable" {
		t.Fatalf("reason = %q, want retriever_unavailable", got.Evaluation.Reason)
	}
	if got.Retrieval.Status != RetrievalStatusUnavailable {
		t.Fatalf("retrieval status = %q, want unavailable", got.Retrieval.Status)
	}
	if !strings.Contains(got.Answer, "没有在当前知识库范围内检索到足够相关的内容") {
		t.Fatalf("expected insufficient-context answer, got %q", got.Answer)
	}
}

func TestRunTurnRejectsWhitespaceQuery(t *testing.T) {
	agent := New(disabledLLMConfig(), rag.ContextConfig{})

	if _, err := agent.RunTurn(context.Background(), TurnRequest{Query: "   "}); err == nil {
		t.Fatal("expected query validation error")
	}
}

func TestAnswerUsesPackedRAGContextBudgetWhenLLMDisabled(t *testing.T) {
	agent := New(disabledLLMConfig(), rag.ContextConfig{MaxRunes: 90})

	answer := agent.Answer(context.Background(), AnswerRequest{
		Query: "question",
		Chunks: []rag.RelevantChunk{
			{
				Text:       strings.Repeat("界", 200),
				Score:      0.91,
				SourceName: "guide.md",
			},
			{
				Text:  "second chunk should not fit",
				Score: 0.8,
			},
		},
		Evaluation: RetrievalEvaluation{ShouldRetrieve: true},
	})

	if !strings.Contains(answer, "[片段 1]") {
		t.Fatalf("expected packed snippet, got %q", answer)
	}
	if !strings.Contains(answer, "部分检索上下文已按长度预算截断") {
		t.Fatalf("expected truncation note, got %q", answer)
	}
	if strings.Contains(answer, "second chunk should not fit") {
		t.Fatalf("expected second chunk to be omitted, got %q", answer)
	}
}

func TestAnswerRefusesWhenRetrievalExpectedButNoContext(t *testing.T) {
	calls := 0
	agent := NewWithModelFactory(enabledLLMConfig(), rag.ContextConfig{}, func(context.Context, appconfig.LLMConfig) (ChatModel, error) {
		calls++
		return &fakeModel{response: "hallucinated answer"}, nil
	})

	answer := agent.Answer(context.Background(), AnswerRequest{
		Query:      "视频里讲了什么？",
		Evaluation: RetrievalEvaluation{ShouldRetrieve: true, Reason: "knowledge_base_question"},
	})

	if calls != 0 {
		t.Fatalf("model factory calls = %d, want 0", calls)
	}
	if !strings.Contains(answer, "没有在当前知识库范围内检索到足够相关的内容") {
		t.Fatalf("expected insufficient-context answer, got %q", answer)
	}
}

func TestAnswerUsesLLMWhenRetrievalNotExpectedAndNoContext(t *testing.T) {
	model := &fakeModel{response: "你好，我在。"}
	agent := NewWithModelFactory(enabledLLMConfig(), rag.ContextConfig{}, func(context.Context, appconfig.LLMConfig) (ChatModel, error) {
		return model, nil
	})

	answer := agent.Answer(context.Background(), AnswerRequest{
		Query:      "你好",
		Evaluation: RetrievalEvaluation{ShouldRetrieve: false, Reason: "small_talk"},
	})

	if answer != "你好，我在。" {
		t.Fatalf("answer = %q, want LLM response", answer)
	}
	if len(model.input) != 2 {
		t.Fatalf("expected model to receive prompt, got %#v", model.input)
	}
}

func TestAnswerBuildsRAGPromptWithHistoryAndUserFacts(t *testing.T) {
	model := &fakeModel{response: "answer"}
	agent := NewWithModelFactory(enabledLLMConfig(), rag.ContextConfig{MaxRunes: 1024}, func(context.Context, appconfig.LLMConfig) (ChatModel, error) {
		return model, nil
	})

	answer := agent.Answer(context.Background(), AnswerRequest{
		Query:     "视频里说了什么？",
		History:   "用户: 之前的问题\n助手: 之前的回答",
		UserFacts: "用户画像: 喜欢 Go",
		Chunks: []rag.RelevantChunk{
			{
				Text:        "视频讲到了 Markdown RAG。",
				Score:       0.92,
				HeadingPath: "Guide > RAG",
			},
		},
		Evaluation: RetrievalEvaluation{ShouldRetrieve: true, Reason: "knowledge_base_question"},
	})

	if answer != "answer" {
		t.Fatalf("answer = %q, want LLM response", answer)
	}
	if len(model.input) != 2 {
		t.Fatalf("expected system and user messages, got %#v", model.input)
	}
	system := model.input[0].Content
	user := model.input[1].Content
	for _, want := range []string{"用户画像: 喜欢 Go", "仅根据提供的上下文回答"} {
		if !strings.Contains(system, want) {
			t.Fatalf("system prompt missing %q:\n%s", want, system)
		}
	}
	for _, want := range []string{"对话历史", "[片段 1]", "Guide > RAG", "视频里说了什么？"} {
		if !strings.Contains(user, want) {
			t.Fatalf("user prompt missing %q:\n%s", want, user)
		}
	}
}

func TestAnswerFallsBackToPackedContextWhenLLMFails(t *testing.T) {
	agent := NewWithModelFactory(enabledLLMConfig(), rag.ContextConfig{MaxRunes: 1024}, func(context.Context, appconfig.LLMConfig) (ChatModel, error) {
		return &fakeModel{err: errors.New("model down")}, nil
	})

	answer := agent.Answer(context.Background(), AnswerRequest{
		Query: "question",
		Chunks: []rag.RelevantChunk{
			{Text: "retrieved context", Score: 0.7},
		},
		Evaluation: RetrievalEvaluation{ShouldRetrieve: true},
	})

	if !strings.Contains(answer, "retrieved context") {
		t.Fatalf("expected fallback context, got %q", answer)
	}
}

func TestExtractMemoryFactsReturnsNilWhenLLMDisabled(t *testing.T) {
	agent := New(disabledLLMConfig(), rag.ContextConfig{})

	got := agent.ExtractMemoryFacts(context.Background(), MemoryExtractionRequest{
		UserMessage:    "我正在学 Go",
		AssistantReply: "很好",
	})

	if got != nil {
		t.Fatalf("expected nil facts when LLM disabled, got %#v", got)
	}
}

func TestExtractMemoryFactsParsesWrappedJSONAndBuildsPrompt(t *testing.T) {
	model := &fakeModel{response: `提取结果如下：[{"category":"学习","key":"正在学习的语言","value":"Go","evidence":"用户说正在学 Go","source":"explicit","confidence":"high","action":"create"}]`}
	agent := NewWithModelFactory(enabledLLMConfig(), rag.ContextConfig{}, func(context.Context, appconfig.LLMConfig) (ChatModel, error) {
		return model, nil
	})

	got := agent.ExtractMemoryFacts(context.Background(), MemoryExtractionRequest{
		ExistingFactsJSON: `[{"id":"fact-1","category":"偏好","key":"回答风格","value":"简洁","confidence":"high"}]`,
		History:           "用户: 你好\n助手: 你好",
		UserMessage:       "我正在学 Go",
		AssistantReply:    strings.Repeat("答", 700),
	})

	if len(got) != 1 {
		t.Fatalf("expected one fact, got %#v", got)
	}
	if got[0].Action != "create" || got[0].Key != "正在学习的语言" || got[0].Value != "Go" {
		t.Fatalf("unexpected fact: %#v", got[0])
	}
	if len(model.input) != 2 {
		t.Fatalf("expected system and user messages, got %#v", model.input)
	}
	system := model.input[0].Content
	user := model.input[1].Content
	for _, want := range []string{"用户画像提取助手", "supersede", "JSON数组"} {
		if !strings.Contains(system, want) {
			t.Fatalf("system prompt missing %q:\n%s", want, system)
		}
	}
	for _, want := range []string{"已有用户画像", "fact-1", "我正在学 Go"} {
		if !strings.Contains(user, want) {
			t.Fatalf("user prompt missing %q:\n%s", want, user)
		}
	}
	if strings.Contains(user, strings.Repeat("答", 501)) {
		t.Fatalf("assistant reply was not truncated in prompt")
	}
}

func TestExtractMemoryFactsReturnsNilOnModelFailure(t *testing.T) {
	agent := NewWithModelFactory(enabledLLMConfig(), rag.ContextConfig{}, func(context.Context, appconfig.LLMConfig) (ChatModel, error) {
		return &fakeModel{err: errors.New("model down")}, nil
	})

	got := agent.ExtractMemoryFacts(context.Background(), MemoryExtractionRequest{
		UserMessage:    "我正在学 Go",
		AssistantReply: "很好",
	})

	if got != nil {
		t.Fatalf("expected nil facts on model failure, got %#v", got)
	}
}

func TestRetrieveFilterForTurnScopesByUserBeforeSession(t *testing.T) {
	got, ok := retrieveFilterForTurn(" u1 ", " s1 ")
	if !ok {
		t.Fatal("expected scope")
	}
	if got == nil {
		t.Fatal("expected filter")
	}
	if got.UserID != "u1" {
		t.Fatalf("UserID = %q, want u1", got.UserID)
	}
	if got.SessionID != "" {
		t.Fatalf("SessionID = %q, want empty when user scope is present", got.SessionID)
	}
}

func TestRetrieveFilterForTurnUsesExplicitSessionWhenUserMissing(t *testing.T) {
	got, ok := retrieveFilterForTurn("", " s1 ")
	if !ok {
		t.Fatal("expected scope")
	}
	if got == nil {
		t.Fatal("expected filter")
	}
	if got.UserID != "" {
		t.Fatalf("UserID = %q, want empty", got.UserID)
	}
	if got.SessionID != "s1" {
		t.Fatalf("SessionID = %q, want s1", got.SessionID)
	}
}

func TestRetrieveFilterForTurnRejectsMissingScope(t *testing.T) {
	got, ok := retrieveFilterForTurn("", " ")
	if ok {
		t.Fatalf("expected missing scope to be rejected, got %#v", got)
	}
	if got != nil {
		t.Fatalf("expected nil filter on rejected scope, got %#v", got)
	}
}

func disabledLLMConfig() appconfig.LLMConfig {
	enabled := false
	return appconfig.LLMConfig{Enabled: &enabled}
}

func enabledLLMConfig() appconfig.LLMConfig {
	enabled := true
	return appconfig.LLMConfig{Enabled: &enabled, Model: "test-model"}
}
