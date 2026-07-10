package chatagent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"reflect"
	"strings"

	einomodel "github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/memory"
	"github.com/raiki02/vidwise/internal/paragraph"
	"github.com/raiki02/vidwise/internal/rag"
)

const insufficientRAGContextAnswer = "我没有在当前知识库范围内检索到足够相关的内容，因此不能可靠回答这个问题。你可以上传相关资料，或换一种更具体的问法。"

// ChatModel is the LLM interface this agent needs. Keeping it small makes the
// answer policy testable without constructing provider-specific adapters.
type ChatModel interface {
	Generate(ctx context.Context, input []*schema.Message, opts ...einomodel.Option) (*schema.Message, error)
}

// ModelFactory creates the chat model used for one request.
type ModelFactory func(context.Context, appconfig.LLMConfig) (ChatModel, error)

// Retriever is the retrieval adapter used by the turn workflow.
type Retriever interface {
	RetrieveWithOptions(ctx context.Context, req rag.RetrieveRequest) ([]rag.RelevantChunk, error)
}

// RetrievalEvaluation records why retrieval did or did not run for a turn.
type RetrievalEvaluation struct {
	ShouldRetrieve bool   `json:"should_retrieve"`
	Reason         string `json:"reason"`
	RetrievalQuery string `json:"retrieval_query,omitempty"`
}

// RetrievalStatus is the retrieval execution outcome for a turn.
type RetrievalStatus string

const (
	RetrievalStatusNotNeeded     RetrievalStatus = "not_needed"
	RetrievalStatusUnavailable   RetrievalStatus = "unavailable"
	RetrievalStatusScopeRequired RetrievalStatus = "scope_required"
	RetrievalStatusFailed        RetrievalStatus = "failed"
	RetrievalStatusNoResults     RetrievalStatus = "no_results"
	RetrievalStatusRetrieved     RetrievalStatus = "retrieved"
)

// RetrievalOutcome records what happened when the turn crossed the retrieval
// seam. Evaluation explains whether retrieval should run; Outcome explains what
// actually happened.
type RetrievalOutcome struct {
	Status     RetrievalStatus `json:"status"`
	Reason     string          `json:"reason,omitempty"`
	Error      string          `json:"error,omitempty"`
	Query      string          `json:"query,omitempty"`
	Queries    []string        `json:"queries,omitempty"`
	ChunkCount int             `json:"chunk_count"`
}

// RAGContextOutcome records how much retrieved context actually reached the
// answer prompt after packing and budget enforcement.
type RAGContextOutcome struct {
	UsedChunks        int                   `json:"used_chunks"`
	SkippedDuplicates int                   `json:"skipped_duplicates"`
	Truncated         bool                  `json:"truncated"`
	Citations         []rag.ContextCitation `json:"citations,omitempty"`
}

// RetrievalEvaluationRequest contains the inputs needed to decide whether a
// turn should use knowledge-base retrieval.
type RetrievalEvaluationRequest struct {
	Query              string
	History            string
	UserFacts          string
	RetrieverAvailable bool
}

// AnswerRequest is the stable interface for generating a chat answer from
// query, memory, conversation history, and optional retrieved chunks.
type AnswerRequest struct {
	Query      string
	History    string
	UserFacts  string
	Chunks     []rag.RelevantChunk
	Evaluation RetrievalEvaluation
}

// AnswerOutcome is the answer text plus prompt-context packing metadata.
type AnswerOutcome struct {
	Answer     string
	RAGContext RAGContextOutcome
}

// TurnRequest is the stable interface for one user turn. It deliberately keeps
// HTTP/session persistence out while owning the RAG decision and answer policy.
type TurnRequest struct {
	Query       string
	History     string
	UserFacts   string
	UserID      string
	SessionID   string
	SourceIDs   []string
	DocumentIDs []string
}

// TurnResult is the completed answer plus the retrieval trace callers can
// expose to users or logs.
type TurnResult struct {
	Answer     string
	Chunks     []rag.RelevantChunk
	Evaluation RetrievalEvaluation
	Retrieval  RetrievalOutcome
	RAGContext RAGContextOutcome
}

// MemoryExtractionRequest is the turn-level evidence used to update
// cross-session user memory.
type MemoryExtractionRequest struct {
	ExistingFactsJSON string
	History           string
	UserMessage       string
	AssistantReply    string
}

// Agent owns the RAG answer policy: context packing, prompt construction, LLM
// invocation, and non-LLM fallback behaviour.
type Agent struct {
	llmCfg     appconfig.LLMConfig
	ragContext rag.ContextConfig
	newModel   ModelFactory
	retriever  Retriever
}

func New(llmCfg appconfig.LLMConfig, ragContext rag.ContextConfig) *Agent {
	return NewWithModelFactory(llmCfg, ragContext, defaultModelFactory)
}

func NewWithRetriever(llmCfg appconfig.LLMConfig, ragContext rag.ContextConfig, retriever Retriever) *Agent {
	return NewWithModelFactoryAndRetriever(llmCfg, ragContext, defaultModelFactory, retriever)
}

func NewWithModelFactory(llmCfg appconfig.LLMConfig, ragContext rag.ContextConfig, factory ModelFactory) *Agent {
	return NewWithModelFactoryAndRetriever(llmCfg, ragContext, factory, nil)
}

func NewWithModelFactoryAndRetriever(llmCfg appconfig.LLMConfig, ragContext rag.ContextConfig, factory ModelFactory, retriever Retriever) *Agent {
	if factory == nil {
		factory = defaultModelFactory
	}
	return &Agent{
		llmCfg:     llmCfg,
		ragContext: ragContext,
		newModel:   factory,
		retriever:  normalizeRetrieverAdapter(retriever),
	}
}

func defaultModelFactory(ctx context.Context, cfg appconfig.LLMConfig) (ChatModel, error) {
	return paragraph.NewChatModel(ctx, cfg)
}

func normalizeRetrieverAdapter(retriever Retriever) Retriever {
	if retriever == nil {
		return nil
	}
	value := reflect.ValueOf(retriever)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		if value.IsNil() {
			return nil
		}
	}
	return retriever
}

// RunTurn evaluates retrieval, searches the knowledge base when useful, and
// produces the final answer. HTTP adapters should call this once per user turn
// instead of duplicating RAG ordering rules.
func (a *Agent) RunTurn(ctx context.Context, req TurnRequest) (TurnResult, error) {
	req.Query = strings.TrimSpace(req.Query)
	if req.Query == "" {
		return TurnResult{}, errors.New("query is required")
	}

	eval := a.EvaluateRetrieval(ctx, RetrievalEvaluationRequest{
		Query:              req.Query,
		History:            req.History,
		UserFacts:          req.UserFacts,
		RetrieverAvailable: a.retriever != nil,
	})

	var chunks []rag.RelevantChunk
	retrieval := RetrievalOutcome{
		Status: RetrievalStatusNotNeeded,
		Reason: eval.Reason,
	}
	retrievalQuery := retrievalQueryForTurn(eval, req.Query)
	if eval.ShouldRetrieve && a.retriever != nil {
		retrieval.Query = retrievalQuery
		filter, ok := retrieveFilterForTurn(req.UserID, req.SessionID, req.SourceIDs, req.DocumentIDs)
		if !ok {
			slog.Info("chat.agent.retrieve_skipped", "reason", "scope_required")
			retrieval.Status = RetrievalStatusScopeRequired
			retrieval.Reason = "scope_required"
		} else {
			retrievedChunks, queries, err := a.retrieveWithFallbackQueries(ctx, retrievalQuery, req.Query, filter)
			retrieval.Queries = queries
			if len(queries) > 0 {
				retrieval.Query = queries[len(queries)-1]
			}
			if err != nil {
				slog.Error("chat.agent.retrieve_failed", "err", err)
				retrieval.Status = RetrievalStatusFailed
				retrieval.Reason = "retriever_error"
				retrieval.Error = err.Error()
			} else {
				chunks = retrievedChunks
				retrieval.ChunkCount = len(chunks)
				if len(chunks) == 0 {
					retrieval.Status = RetrievalStatusNoResults
					retrieval.Reason = "no_results"
				} else {
					retrieval.Status = RetrievalStatusRetrieved
					retrieval.Reason = "retrieved"
				}
			}
		}
	} else if eval.ShouldRetrieve {
		retrieval.Status = RetrievalStatusUnavailable
		retrieval.Reason = "retriever_unavailable"
		retrieval.Query = retrievalQuery
		retrieval.Queries = []string{retrievalQuery}
	}

	answer := a.AnswerWithOutcome(ctx, AnswerRequest{
		Query:      req.Query,
		History:    req.History,
		UserFacts:  req.UserFacts,
		Chunks:     chunks,
		Evaluation: eval,
	})

	return TurnResult{
		Answer:     answer.Answer,
		Chunks:     chunks,
		Evaluation: eval,
		Retrieval:  retrieval,
		RAGContext: answer.RAGContext,
	}, nil
}

func (a *Agent) retrieveWithFallbackQueries(ctx context.Context, retrievalQuery, originalQuery string, filter *rag.RetrieveFilter) ([]rag.RelevantChunk, []string, error) {
	queries := retrievalQueriesForTurn(retrievalQuery, originalQuery)
	for i, query := range queries {
		chunks, err := a.retriever.RetrieveWithOptions(ctx, rag.RetrieveRequest{
			Query:  query,
			Filter: filter,
		})
		if err != nil {
			return nil, queries[:i+1], err
		}
		if len(chunks) > 0 {
			return chunks, queries[:i+1], nil
		}
	}
	return nil, queries, nil
}

// EvaluateRetrieval decides whether the agent should query the knowledge base
// for this turn. It uses the LLM as a query classifier when available, and
// falls back to conservative retrieval on model failures.
func (a *Agent) EvaluateRetrieval(ctx context.Context, req RetrievalEvaluationRequest) RetrievalEvaluation {
	req.Query = strings.TrimSpace(req.Query)
	if !req.RetrieverAvailable {
		if isLikelyKnowledgeBaseQuery(req.Query) {
			return RetrievalEvaluation{ShouldRetrieve: true, Reason: "retriever_unavailable", RetrievalQuery: req.Query}
		}
		return RetrievalEvaluation{ShouldRetrieve: false, Reason: "retriever_unavailable"}
	}
	if !a.llmEnabled() {
		return RetrievalEvaluation{ShouldRetrieve: true, Reason: "llm_unavailable", RetrievalQuery: req.Query}
	}

	cm, err := a.newModel(ctx, a.llmCfg)
	if err != nil {
		slog.Warn("chat.agent.rag_eval_llm_failed", "err", err)
		return RetrievalEvaluation{ShouldRetrieve: true, Reason: "llm_init_failed", RetrievalQuery: req.Query}
	}

	resp, genErr := cm.Generate(ctx, buildRetrievalEvaluationMessages(req))
	if genErr != nil || resp == nil || strings.TrimSpace(resp.Content) == "" {
		slog.Warn("chat.agent.rag_eval_gen_failed", "err", genErr)
		return RetrievalEvaluation{ShouldRetrieve: true, Reason: "eval_gen_failed", RetrievalQuery: req.Query}
	}

	var result RetrievalEvaluation
	if err := extractJSON(resp.Content, &result); err != nil {
		slog.Warn("chat.agent.rag_eval_parse_failed", "content", resp.Content, "err", err)
		return RetrievalEvaluation{ShouldRetrieve: true, Reason: "eval_parse_failed", RetrievalQuery: req.Query}
	}

	result = normalizeRetrievalEvaluation(result, req.Query)
	slog.Info("chat.agent.rag_eval", "should_retrieve", result.ShouldRetrieve, "reason", result.Reason)
	return result
}

// ExtractMemoryFacts asks the LLM to turn a completed chat turn into structured
// memory fact operations. It returns nil when memory extraction is unavailable,
// unnecessary, or the model output cannot be trusted.
func (a *Agent) ExtractMemoryFacts(ctx context.Context, req MemoryExtractionRequest) []memory.ExtractedFact {
	if !a.llmEnabled() {
		return nil
	}

	cm, err := a.newModel(ctx, a.llmCfg)
	if err != nil {
		slog.Warn("chat.agent.memory_extract_llm_failed", "err", err)
		return nil
	}

	resp, genErr := cm.Generate(ctx, buildMemoryExtractionMessages(req))
	if genErr != nil || resp == nil || strings.TrimSpace(resp.Content) == "" {
		slog.Warn("chat.agent.memory_extract_gen_failed", "err", genErr)
		return nil
	}

	var facts []memory.ExtractedFact
	if err := extractJSON(resp.Content, &facts); err != nil {
		slog.Warn("chat.agent.memory_extract_parse_failed", "content", resp.Content, "err", err)
		return nil
	}
	return facts
}

func (a *Agent) Answer(ctx context.Context, req AnswerRequest) string {
	return a.AnswerWithOutcome(ctx, req).Answer
}

func (a *Agent) AnswerWithOutcome(ctx context.Context, req AnswerRequest) AnswerOutcome {
	req.Query = strings.TrimSpace(req.Query)
	packedContext := rag.PackContext(req.Chunks, a.ragContext)
	outcome := AnswerOutcome{
		RAGContext: RAGContextOutcome{
			UsedChunks:        packedContext.UsedChunks,
			SkippedDuplicates: packedContext.SkippedDuplicates,
			Truncated:         packedContext.Truncated,
			Citations:         packedContext.Citations,
		},
	}
	hasRAGContext := packedContext.HasContext()

	if req.Evaluation.ShouldRetrieve && !hasRAGContext {
		outcome.Answer = insufficientRAGContextAnswer
		return outcome
	}

	if a.llmEnabled() {
		cm, err := a.newModel(ctx, a.llmCfg)
		if err == nil {
			msgs := buildMessages(answerPromptInput{
				Query:         req.Query,
				History:       req.History,
				UserFacts:     req.UserFacts,
				HasRAGContext: hasRAGContext,
				RAGContext:    packedContext.Text,
			})
			resp, genErr := cm.Generate(ctx, msgs)
			if genErr == nil && resp != nil && strings.TrimSpace(resp.Content) != "" {
				outcome.Answer = resp.Content
				return outcome
			}
			slog.Warn("chat.agent.llm_gen_failed", "err", genErr)
		} else {
			slog.Warn("chat.agent.llm_unavailable", "err", err)
		}
	}

	outcome.Answer = fallbackAnswer(packedContext)
	return outcome
}

func (a *Agent) llmEnabled() bool {
	return a.llmCfg.Enabled != nil && *a.llmCfg.Enabled && strings.TrimSpace(a.llmCfg.Model) != ""
}

func retrieveFilterForTurn(userID, sessionID string, sourceIDs, documentIDs []string) (*rag.RetrieveFilter, bool) {
	filter, err := rag.NewRetrieveFilterWithPolicy(userID, sessionID, rag.ChatScopePolicy())
	if err != nil {
		return nil, false
	}
	if filter != nil {
		filter.SourceIDs = sourceIDs
		filter.DocumentIDs = documentIDs
	}
	return rag.NormalizeRetrieveFilter(filter), true
}

func normalizeRetrievalEvaluation(eval RetrievalEvaluation, fallbackQuery string) RetrievalEvaluation {
	eval.Reason = strings.TrimSpace(eval.Reason)
	eval.RetrievalQuery = strings.TrimSpace(eval.RetrievalQuery)
	if eval.ShouldRetrieve {
		if eval.Reason == "" {
			eval.Reason = "knowledge_base_question"
		}
		if eval.RetrievalQuery == "" {
			eval.RetrievalQuery = strings.TrimSpace(fallbackQuery)
		}
	} else {
		if eval.Reason == "" {
			eval.Reason = "not_needed"
		}
		eval.RetrievalQuery = ""
	}
	return eval
}

func retrievalQueryForTurn(eval RetrievalEvaluation, fallbackQuery string) string {
	if query := strings.TrimSpace(eval.RetrievalQuery); query != "" {
		return query
	}
	return strings.TrimSpace(fallbackQuery)
}

func retrievalQueriesForTurn(retrievalQuery, originalQuery string) []string {
	queries := make([]string, 0, 2)
	seen := map[string]struct{}{}
	for _, query := range []string{retrievalQuery, originalQuery} {
		query = strings.TrimSpace(query)
		if query == "" {
			continue
		}
		key := strings.ToLower(query)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		queries = append(queries, query)
	}
	return queries
}

func isLikelyKnowledgeBaseQuery(query string) bool {
	query = strings.ToLower(strings.TrimSpace(query))
	if query == "" {
		return false
	}
	keywords := []string{
		"知识库",
		"上传",
		"文档",
		"资料",
		"视频",
		"转录",
		"字幕",
		"片段",
		"讲了什么",
		"knowledge base",
		"uploaded",
		"document",
		"video",
		"transcript",
		"caption",
	}
	for _, keyword := range keywords {
		if strings.Contains(query, keyword) {
			return true
		}
	}
	return false
}

func buildRetrievalEvaluationMessages(input RetrievalEvaluationRequest) []*schema.Message {
	systemPrompt := `你是一个RAG（检索增强生成）系统的查询评估器。你的任务是判断用户的最新问题是否需要从视频知识库中检索相关信息，并在需要检索时生成一条适合向量检索的独立查询。

需要触发RAG的情况（回答"是"）：
1. 问题涉及视频的具体内容、转录文本、或之前上传到知识库的文档
2. 用户询问关于某个视频的话题、细节、或片段
3. 用户想知道知识库中是否包含某个主题
4. 用户要求引用、查找、搜索具体信息

不需要触发RAG的情况（回答"否"）：
1. 简单的寒暄和闲聊（"你好"、"今天天气怎么样"）
2. 纯技术咨询，不涉及知识库内容（"go语言的goroutine怎么用"）
3. 用户在谈论自己的情况（"我正在学go"——这是用户个人信息，不是对知识库的查询）
4. 用户对AI能力的询问（"你能做什么"、"你怎么工作的"）
5. 用户要求总结当前对话（不涉及外部知识）

retrieval_query规则：
1. 只有should_retrieve为true时才填写。
2. 结合最近对话，把"它"、"这个"、"那具体怎么做"这类追问改写成可以单独检索的查询。
3. 保留关键实体、视频/文档主题、用户真正要找的信息，不要加入不存在的细节。

重要：用户谈论自己的个人信息（如学习计划、偏好等）不需要触发RAG，因为这些不是对知识库的查询。`

	userPrompt := fmt.Sprintf("用户最新问题：%s\n\n请判断是否需要从视频知识库检索。返回JSON格式：{\"should_retrieve\": true/false, \"reason\": \"判断理由\", \"retrieval_query\": \"需要检索时，将用户问题改写成可独立向量检索的查询；不需要检索时留空\"}", input.Query)
	if strings.TrimSpace(input.History) != "" {
		userPrompt = fmt.Sprintf("最近对话：\n%s\n\n%s", input.History, userPrompt)
	}
	if strings.TrimSpace(input.UserFacts) != "" {
		userPrompt = fmt.Sprintf("%s\n\n用户画像信息：\n%s", userPrompt, input.UserFacts)
	}

	return []*schema.Message{
		schema.SystemMessage(systemPrompt),
		schema.UserMessage(userPrompt),
	}
}

func buildMemoryExtractionMessages(input MemoryExtractionRequest) []*schema.Message {
	existingJSON := strings.TrimSpace(input.ExistingFactsJSON)
	if existingJSON == "" {
		existingJSON = "[]"
	}

	systemPrompt := `你是一个用户画像提取助手。你的任务是从用户与AI的对话中，提取关于用户的重要信息作为跨会话记忆。

提取规则：
1. 从用户的消息中提取用户的个人信息、偏好、状态、计划、技能等
2. 信息必须是用户明确表达或强烈暗示的，不要无中生有
3. 每条信息需要分类（category），例如：学习、工作、兴趣、偏好、目标、技能、个人情况等
4. 为每条信息设置置信度：
   - "high": 用户明确陈述（如"我正在学go"）
   - "medium": 从对话中合理推断（如多次问go相关问题，推断用户在学习go）
   - "low": 弱信号，仅作参考

动作（action）说明：
- "create": 这是一个新发现的信息，之前没有记录
- "update": 更新已有信息（用户提供了更具体的内容）
- "confirm": 用户在对话中再次提及，确认该信息仍然有效
- "supersede": 用户的新表达与旧信息矛盾，新信息更可靠（如用户先说"学go"，后来说"改用rust了"）

矛盾处理规则（重要）：
1. 当新旧事实矛盾时，必须判断哪个更可靠：
   - 用户的直接最新陈述 > 旧的直接陈述
   - 被多次提及的事实 > 只提过一次的事实
   - 有具体上下文支撑的事实 > 孤立陈述
   - 用户明确更正 > 推断
2. 不要"无脑去旧留新"——如果旧事实被多次确认，而新表述可能是一时口误，不应该覆盖
3. 不要"无脑去旧留新"——如果用户明确表示情况发生了变化，应该用新事实替代旧事实
4. 在reason字段中说明判断依据

输出格式：JSON数组
[
  {
    "category": "学习",
    "key": "编程语言",
    "value": "Go语言",
    "evidence": "用户说'我正在学go'",
    "source": "explicit",
    "confidence": "high",
    "action": "create",
    "target_id": "",
    "reason": ""
  }
]

注意：
- 只提取用户的信息，不提取AI的回复内容
- 不要提取临时性的、无意义的对话内容
- 如果用户只是打招呼或闲聊，返回空数组 []
- 如果新信息与已有信息完全相同，用"confirm"动作即可，不需要重复创建
- category用中文，key和value用中文简洁描述
- key应该是一个简短的属性名（如"正在学习的语言"），value是具体值（如"Go语言"）`

	userPrompt := fmt.Sprintf(
		"已有用户画像: %s\n\n最近对话:\n%s\n用户: %s\n助手: %s\n\n请从以上对话中提取关于用户的新信息或更新已有信息。",
		existingJSON,
		input.History,
		strings.TrimSpace(input.UserMessage),
		truncateString(input.AssistantReply, 500),
	)

	return []*schema.Message{
		schema.SystemMessage(systemPrompt),
		schema.UserMessage(userPrompt),
	}
}

type answerPromptInput struct {
	Query         string
	History       string
	UserFacts     string
	HasRAGContext bool
	RAGContext    string
}

func buildMessages(input answerPromptInput) []*schema.Message {
	systemPrompt := buildSystemPrompt(input.HasRAGContext, input.UserFacts)
	userPrompt := buildUserPrompt(input)
	return []*schema.Message{
		schema.SystemMessage(systemPrompt),
		schema.UserMessage(userPrompt),
	}
}

func buildUserPrompt(input answerPromptInput) string {
	if input.HasRAGContext {
		userPrompt := fmt.Sprintf("视频转录文本（来自知识库，请务必根据这些内容回答，并标注片段编号）：\n%s\n\n用户问题：%s", input.RAGContext, input.Query)
		if strings.TrimSpace(input.History) != "" {
			userPrompt = fmt.Sprintf("对话历史：\n%s\n\n%s", input.History, userPrompt)
		}
		return userPrompt + "\n\n请根据视频知识库内容和对话历史回答用户的问题。如果引用了知识库中的具体信息，请标注对应的片段编号。"
	}

	userPrompt := fmt.Sprintf("用户问题：%s", input.Query)
	if strings.TrimSpace(input.History) != "" {
		userPrompt = fmt.Sprintf("对话历史：\n%s\n\n用户最新问题：%s", input.History, input.Query)
	}
	if strings.TrimSpace(input.UserFacts) != "" {
		userPrompt += "\n\n请结合用户画像信息，自然地个性化你的回复。"
	}
	return userPrompt
}

func buildSystemPrompt(hasRAG bool, userFacts string) string {
	var sb strings.Builder

	sb.WriteString("你是视频知识库问答助手。")

	if strings.TrimSpace(userFacts) != "" {
		sb.WriteString("\n\n")
		sb.WriteString(userFacts)
		sb.WriteString("\n请自然地利用这些用户信息来个性化你的回复，但不要直接复述它们。")
	}

	if hasRAG {
		sb.WriteString(`

要求：
1. 仅根据提供的上下文回答，不要编造信息。
2. 提取上下文中与问题最相关的内容，用自己的话重新组织。
3. 回答要具体、有细节，不要泛泛而谈。
4. 如果上下文信息不足，明确告知用户，不要猜测。
5. 结合对话历史理解用户的意图和上下文。
6. 用中文回答，语言清晰易懂。
7. 当引用了知识库中某段内容时，注明"根据知识库中的内容..."以便用户追溯来源。`)
	} else {
		sb.WriteString(`

要求：
1. 准确回答用户问题，不要编造信息。
2. 如果问题超出你的知识范围，明确告知用户。
3. 结合对话历史理解用户的意图和上下文。
4. 用中文回答，语言清晰易懂。`)
	}

	return sb.String()
}

func fallbackAnswer(packedContext rag.PackedContext) string {
	if packedContext.HasContext() {
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("基于 %d 个相关文本片段：\n\n", packedContext.UsedChunks))
		sb.WriteString(packedContext.Text)
		if packedContext.Truncated {
			sb.WriteString("\n\n部分检索上下文已按长度预算截断。")
		}
		return sb.String()
	}
	return "抱歉，我暂时无法处理你的请求。请稍后再试。"
}

// extractJSON finds and parses the first balanced JSON object or array in an
// LLM response, allowing providers to wrap JSON in explanatory text.
func extractJSON(content string, v interface{}) error {
	content = strings.TrimSpace(content)
	if err := json.Unmarshal([]byte(content), v); err == nil {
		return nil
	}

	start := -1
	depth := 0
	var openBracket, closeBracket rune

	for i, ch := range content {
		if start == -1 {
			if ch == '[' || ch == '{' {
				start = i
				depth = 1
				if ch == '[' {
					openBracket, closeBracket = '[', ']'
				} else {
					openBracket, closeBracket = '{', '}'
				}
			}
			continue
		}

		if ch == openBracket {
			depth++
		} else if ch == closeBracket {
			depth--
			if depth == 0 {
				return json.Unmarshal([]byte(content[start:i+1]), v)
			}
		}
		if openBracket == '[' && ch == '{' {
			depth++
		} else if openBracket == '[' && ch == '}' {
			depth--
		} else if openBracket == '{' && ch == '[' {
			depth++
		} else if openBracket == '{' && ch == ']' {
			depth--
		}
	}

	if start >= 0 && start < len(content) {
		return json.Unmarshal([]byte(content[start:]), v)
	}
	return fmt.Errorf("no JSON found in: %s", truncateString(content, 200))
}

func truncateString(s string, maxRunes int) string {
	runes := []rune(s)
	if len(runes) <= maxRunes {
		return s
	}
	return string(runes[:maxRunes]) + "..."
}
