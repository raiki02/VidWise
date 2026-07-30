package knowledgeagent

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/google/uuid"
	"github.com/raiki02/vidwise/internal/chat"
	"github.com/raiki02/vidwise/internal/chatagent"
	"github.com/raiki02/vidwise/internal/rag"
	taskpkg "github.com/raiki02/vidwise/internal/task"
	"github.com/raiki02/vidwise/internal/videoinput"
)

type Service struct {
	sessions   SessionStore
	answerer   AnswerRunner
	sources    SourceService
	videos     VideoProcessor
	indexer    TranscriptIndexer
	formatter  TextFormatter
	tasks      TaskReader
	actions    *ActionStore
	classifier IntentClassifier
}

type ServiceConfig struct {
	Sessions         SessionStore
	Answerer         AnswerRunner
	Sources          SourceService
	Videos           VideoProcessor
	Indexer          TranscriptIndexer
	Formatter        TextFormatter
	Tasks            TaskReader
	Actions          *ActionStore
	IntentClassifier IntentClassifier
}

func NewService(cfg ServiceConfig) *Service {
	if cfg.Actions == nil {
		cfg.Actions = NewActionStore(ActionStoreOptions{})
	}
	return &Service{
		sessions:   normalizeSessionStore(cfg.Sessions),
		answerer:   cfg.Answerer,
		sources:    cfg.Sources,
		videos:     cfg.Videos,
		indexer:    cfg.Indexer,
		formatter:  cfg.Formatter,
		tasks:      cfg.Tasks,
		actions:    cfg.Actions,
		classifier: cfg.IntentClassifier,
	}
}

func (s *Service) Turn(ctx context.Context, req TurnRequest) (TurnResponse, error) {
	req.UserID = strings.TrimSpace(req.UserID)
	req.SessionID = strings.TrimSpace(req.SessionID)
	req.Message = strings.TrimSpace(req.Message)
	if req.UserID == "" {
		return TurnResponse{}, errors.New("user_id is required")
	}
	if req.Message == "" {
		return TurnResponse{}, errors.New("message is required")
	}

	sessionID, history, err := s.prepareSession(ctx, req)
	if err != nil {
		return TurnResponse{}, err
	}
	req.SessionID = sessionID
	s.saveMessage(ctx, sessionID, "user", req.Message)

	var resp TurnResponse
	switch intent := s.route(ctx, req); intent.kind {
	case ActionProcessVideo:
		action := s.storePending(newProcessVideoAction(intent.videoURL, intent.videoName, req), req.UserID, sessionID)
		resp = TurnResponse{
			TraceID:        req.TraceID,
			SessionID:      sessionID,
			Answer:         fmt.Sprintf("我识别到了视频链接：%s。确认后我会开始转写和整理，完成后你可以再确认存入知识库。", displayName(intent.videoTitle, intent.videoName, "这个视频")),
			PendingActions: []AgentAction{action},
		}
	case ActionIndexTaskTranscript:
		action := s.storePending(newIndexTaskAction(intent.taskID), req.UserID, sessionID)
		resp = TurnResponse{
			TraceID:        req.TraceID,
			SessionID:      sessionID,
			Answer:         "我找到了可入库的任务。确认后会把该任务的格式化转写存入你的知识库。",
			PendingActions: []AgentAction{action},
		}
	case ActionDeleteSource:
		action := s.storePending(newDeleteSourceAction(intent.sourceIDs), req.UserID, sessionID)
		resp = TurnResponse{
			TraceID:        req.TraceID,
			SessionID:      sessionID,
			Answer:         "删除知识库 source 会移除对应检索内容，需要你先确认。",
			PendingActions: []AgentAction{action},
		}
	case ActionListSources:
		resp, err = s.listSources(ctx, req)
		if err != nil {
			return TurnResponse{}, err
		}
	case ActionFormatText:
		action := s.storePending(newFormatTextAction(intent.text), req.UserID, sessionID)
		resp = TurnResponse{
			TraceID:        req.TraceID,
			SessionID:      sessionID,
			Answer:         "我可以帮你整理这段文本。确认后会调用格式化流程处理。",
			PendingActions: []AgentAction{action},
		}
	default:
		resp, err = s.answerFromKnowledge(ctx, req, history)
		if err != nil {
			return TurnResponse{}, err
		}
	}

	s.saveMessage(ctx, sessionID, "assistant", resp.Answer)
	return resp, nil
}

func (s *Service) Confirm(ctx context.Context, actionID string, req ConfirmRequest) (TurnResponse, error) {
	actionID = strings.TrimSpace(actionID)
	req.UserID = strings.TrimSpace(req.UserID)
	if actionID == "" {
		return TurnResponse{}, errors.New("action id is required")
	}
	if req.UserID == "" {
		return TurnResponse{}, errors.New("user_id is required")
	}
	action, err := s.actions.Reserve(actionID, req.UserID)
	if err != nil {
		return TurnResponse{}, err
	}

	executed := ExecutedAction{
		AgentAction: action,
		Status:      actionStatusExecuting,
	}
	resp := TurnResponse{
		TraceID: req.TraceID,
	}

	switch action.Type {
	case ActionProcessVideo:
		result, execErr := s.executeProcessVideo(ctx, action, req.TraceID)
		if execErr != nil {
			executed.Error = execErr.Error()
			s.actions.Fail(action.ID, executed)
			return TurnResponse{}, execErr
		}
		executed.Output = map[string]any{
			"task_id":    result.TaskID,
			"status":     result.Status,
			"trace_id":   result.TraceID,
			"session_id": result.SessionID,
		}
		resp.SessionID = result.SessionID
		resp.TaskIDs = []string{result.TaskID}
		resp.Answer = fmt.Sprintf("已开始处理视频，任务 ID：%s。完成后我会提示你把格式化转写存入知识库。", result.TaskID)
	case ActionIndexTaskTranscript:
		result, execErr := s.executeIndexTask(ctx, action)
		if execErr != nil {
			executed.Error = execErr.Error()
			s.actions.Fail(action.ID, executed)
			return TurnResponse{}, execErr
		}
		executed.Output = map[string]any{
			"status":       result.Status,
			"task_id":      result.TaskID,
			"chunk_count":  result.ChunkCount,
			"content_type": result.ContentType,
			"source_ids":   result.SourceIDs,
		}
		resp.TaskIDs = []string{result.TaskID}
		resp.Answer = fmt.Sprintf("已存入知识库，生成 %d 个知识片段。", result.ChunkCount)
	case ActionDeleteSource:
		result, execErr := s.executeDeleteSource(ctx, action, req.UserID)
		if execErr != nil {
			executed.Error = execErr.Error()
			s.actions.Fail(action.ID, executed)
			return TurnResponse{}, execErr
		}
		executed.Output = map[string]any{"source_ids": result.SourceIDs}
		resp.Answer = fmt.Sprintf("已删除 %d 个 source。", len(result.SourceIDs))
	case ActionFormatText:
		result, execErr := s.executeFormatText(ctx, action)
		if execErr != nil {
			executed.Error = execErr.Error()
			s.actions.Fail(action.ID, executed)
			return TurnResponse{}, execErr
		}
		executed.Output = map[string]any{"text": result.Text, "text_length": len(result.Text)}
		resp.Answer = result.Text
	default:
		execErr := fmt.Errorf("unsupported action type: %s", action.Type)
		executed.Error = execErr.Error()
		s.actions.Fail(action.ID, executed)
		return TurnResponse{}, execErr
	}

	executed.Status = actionStatusExecuted
	resp.ExecutedActions = []ExecutedAction{executed}
	if resp.SessionID == "" {
		resp.SessionID = stringInput(action.Input, "session_id")
	}
	s.saveMessage(ctx, resp.SessionID, "assistant", resp.Answer)
	s.actions.Complete(action.ID, executed)
	return resp, nil
}

func (s *Service) prepareSession(ctx context.Context, req TurnRequest) (string, string, error) {
	sessionID := strings.TrimSpace(req.SessionID)
	if sessionID == "" && s.sessions != nil {
		session, err := s.sessions.CreateSessionForUser(ctx, req.UserID, "新对话")
		if err != nil {
			return "", "", fmt.Errorf("create session: %w", err)
		}
		sessionID = session.ID
	}
	if sessionID == "" || s.sessions == nil {
		return sessionID, "", nil
	}
	messages, err := s.sessions.GetRecentMessages(ctx, sessionID, 20)
	if err != nil {
		return sessionID, "", nil
	}
	return sessionID, buildHistoryText(messages), nil
}

func (s *Service) saveMessage(ctx context.Context, sessionID, role, content string) {
	if s.sessions == nil || sessionID == "" || strings.TrimSpace(content) == "" {
		return
	}
	_, _ = s.sessions.AddMessage(ctx, sessionID, role, content)
}

func (s *Service) storePending(action AgentAction, userID, sessionID string) AgentAction {
	if sessionID != "" {
		if action.Input == nil {
			action.Input = map[string]any{}
		}
		if stringInput(action.Input, "session_id") == "" {
			action.Input["session_id"] = sessionID
		}
	}
	if action.ID == "" && s.actions == nil {
		action.ID = uuid.New().String()
		return action
	}
	return s.actions.Add(action, userID, sessionID)
}

func (s *Service) answerFromKnowledge(ctx context.Context, req TurnRequest, history string) (TurnResponse, error) {
	if s.answerer == nil {
		return TurnResponse{
			TraceID:   req.TraceID,
			SessionID: req.SessionID,
			Answer:    "当前问答能力不可用，请稍后再试。",
		}, nil
	}
	turn, err := s.answerer.RunTurn(ctx, chatagent.TurnRequest{
		Query:       req.Message,
		History:     history,
		UserID:      req.UserID,
		SessionID:   req.SessionID,
		SourceIDs:   req.SourceIDs,
		DocumentIDs: req.DocumentIDs,
	})
	if err != nil {
		return TurnResponse{}, err
	}
	action := AgentAction{
		ID:                   uuid.New().String(),
		Type:                 ActionAnswerFromKnowledge,
		Title:                "查询知识库",
		Summary:              "基于当前个人知识库回答问题。",
		RequiresConfirmation: false,
		Input:                map[string]any{"query": req.Message},
		RiskLevel:            RiskReadOnly,
	}
	return TurnResponse{
		TraceID:   req.TraceID,
		SessionID: req.SessionID,
		Answer:    turn.Answer,
		ExecutedActions: []ExecutedAction{{
			AgentAction: action,
			Status:      actionStatusExecuted,
			Output:      map[string]any{"rag_status": string(turn.Retrieval.Status), "chunk_count": turn.Retrieval.ChunkCount},
		}},
		RAGTrace: ragTraceFromTurn(turn, req.SourceIDs, req.DocumentIDs),
		Chunks:   traceChunksFromRelevant(turn.Chunks),
	}, nil
}

func (s *Service) listSources(ctx context.Context, req TurnRequest) (TurnResponse, error) {
	action := AgentAction{
		ID:                   uuid.New().String(),
		Type:                 ActionListSources,
		Title:                "列出知识库 sources",
		Summary:              "读取当前用户范围内的知识库 source 列表。",
		RequiresConfirmation: false,
		Input:                map[string]any{"source_ids": normalizedIDs(req.SourceIDs), "document_ids": normalizedIDs(req.DocumentIDs)},
		RiskLevel:            RiskReadOnly,
	}
	if s.sources == nil {
		return TurnResponse{}, errors.New("RAG source catalog is not available")
	}
	sources, err := s.sources.ListSources(ctx, rag.SourceListRequest{
		Filter: retrieveFilterForRequest(req),
		Limit:  50,
	})
	if err != nil {
		return TurnResponse{}, err
	}
	return TurnResponse{
		TraceID:   req.TraceID,
		SessionID: req.SessionID,
		Answer:    sourceListAnswer(sources),
		ExecutedActions: []ExecutedAction{{
			AgentAction: action,
			Status:      actionStatusExecuted,
			Output:      map[string]any{"sources": sources, "count": len(sources)},
		}},
	}, nil
}

func (s *Service) executeProcessVideo(ctx context.Context, action AgentAction, traceID string) (VideoProcessResult, error) {
	if s.videos == nil {
		return VideoProcessResult{}, errors.New("video processing is not available")
	}
	input := action.Input
	return s.videos.StartVideoProcess(ctx, VideoProcessRequest{
		URL:       stringInput(input, "url"),
		Name:      stringInput(input, "name"),
		UserID:    stringInput(input, "user_id"),
		SessionID: stringInput(input, "session_id"),
		Language:  firstNonEmpty(stringInput(input, "language"), "zh"),
	}, traceID)
}

func (s *Service) executeIndexTask(ctx context.Context, action AgentAction) (TranscriptIndexResult, error) {
	if s.indexer == nil {
		return TranscriptIndexResult{}, errors.New("transcript indexing is not available")
	}
	return s.indexer.IndexTranscriptTask(ctx, stringInput(action.Input, "task_id"))
}

func (s *Service) executeDeleteSource(ctx context.Context, action AgentAction, userID string) (rag.DeleteResult, error) {
	if s.sources == nil {
		return rag.DeleteResult{}, errors.New("RAG indexing is not available")
	}
	sourceIDs := stringSliceInput(action.Input, "source_ids")
	sessionID := stringInput(action.Input, "session_id")
	return s.sources.DeleteSourcesWithOptions(ctx, rag.DeleteRequest{
		SourceIDs: sourceIDs,
		Filter: &rag.RetrieveFilter{
			UserID:    userID,
			SessionID: sessionID,
		},
	})
}

func (s *Service) executeFormatText(ctx context.Context, action AgentAction) (TextFormatResult, error) {
	if s.formatter == nil {
		return TextFormatResult{}, errors.New("text formatting is not available")
	}
	return s.formatter.FormatText(ctx, stringInput(action.Input, "text"))
}

func buildHistoryText(msgs []chat.Message) string {
	var b strings.Builder
	for _, msg := range msgs {
		if msg.Role == "user" {
			b.WriteString(fmt.Sprintf("用户: %s\n", msg.Content))
			continue
		}
		b.WriteString(fmt.Sprintf("助手: %s\n", msg.Content))
	}
	return b.String()
}

func retrieveFilterForRequest(req TurnRequest) *rag.RetrieveFilter {
	return &rag.RetrieveFilter{
		UserID:      req.UserID,
		SessionID:   req.SessionID,
		SourceIDs:   normalizedIDs(req.SourceIDs),
		DocumentIDs: normalizedIDs(req.DocumentIDs),
	}
}

func ragTraceFromTurn(turn chatagent.TurnResult, sourceIDs, documentIDs []string) *RAGTrace {
	return &RAGTrace{
		Triggered:               turn.Evaluation.ShouldRetrieve,
		Reason:                  turn.Evaluation.Reason,
		Status:                  string(turn.Retrieval.Status),
		Query:                   turn.Retrieval.Query,
		Queries:                 turn.Retrieval.Queries,
		SourceIDs:               normalizedIDs(sourceIDs),
		DocumentIDs:             normalizedIDs(documentIDs),
		ChunkCount:              turn.Retrieval.ChunkCount,
		ContextUsedChunks:       turn.RAGContext.UsedChunks,
		ContextSkippedDuplicate: turn.RAGContext.SkippedDuplicates,
		ContextTruncated:        turn.RAGContext.Truncated,
		ContextChunks:           traceChunksFromCitations(turn.RAGContext.Citations),
		AnswerStatus:            string(turn.Grounding.Status),
		CitationRequired:        turn.Grounding.CitationRequired,
		HasCitations:            turn.Grounding.HasCitations,
		CitedSnippets:           turn.Grounding.CitedSnippets,
		InvalidSnippets:         turn.Grounding.InvalidSnippets,
	}
}

func traceChunksFromRelevant(chunks []rag.RelevantChunk) []TraceChunk {
	out := make([]TraceChunk, 0, len(chunks))
	for _, chunk := range chunks {
		out = append(out, traceChunkFromRelevant(chunk))
	}
	return out
}

func traceChunksFromCitations(citations []rag.ContextCitation) []TraceChunk {
	out := make([]TraceChunk, 0, len(citations))
	for _, citation := range citations {
		chunk := traceChunkFromRelevant(citation.Chunk)
		chunk.SnippetNumber = citation.SnippetNumber
		out = append(out, chunk)
	}
	return out
}

func traceChunkFromRelevant(chunk rag.RelevantChunk) TraceChunk {
	return TraceChunk{
		Text:          chunk.Text,
		Score:         chunk.Score,
		SourceID:      chunk.SourceID,
		DocumentID:    chunk.DocumentID,
		ChunkID:       chunk.ChunkID,
		ContentHash:   chunk.ContentHash,
		TaskID:        chunk.TaskID,
		SessionID:     chunk.SessionID,
		ChunkIdx:      chunk.ChunkIdx,
		SourceName:    chunk.SourceName,
		SourceURL:     chunk.SourceURL,
		ContentType:   chunk.ContentType,
		DocumentTitle: chunk.DocumentTitle,
		HeadingPath:   chunk.HeadingPath,
	}
}

type routedIntent struct {
	kind       ActionType
	videoURL   string
	videoName  string
	videoTitle string
	taskID     string
	sourceIDs  []string
	text       string
}

func (s *Service) route(ctx context.Context, req TurnRequest) routedIntent {
	intent := s.routeDeterministic(req)
	if intent.kind != ActionAnswerFromKnowledge {
		return intent
	}
	return s.routeWithClassifier(ctx, req)
}

func (s *Service) routeDeterministic(req TurnRequest) routedIntent {
	message := req.Message
	if normalized := videoinput.NormalizeShareInput(message, ""); looksLikeURL(normalized.URL) {
		name := firstNonEmpty(normalized.Name, fallbackVideoName(normalized.URL))
		return routedIntent{
			kind:       ActionProcessVideo,
			videoURL:   normalized.URL,
			videoName:  name,
			videoTitle: normalized.Title,
		}
	}
	if isIndexTaskIntent(message) {
		if taskID := firstNonEmpty(parseTaskID(message), indexableTaskIDFromRequest(s.tasks, req)); taskID != "" {
			return routedIntent{kind: ActionIndexTaskTranscript, taskID: taskID}
		}
	}
	if isDeleteSourceIntent(message) {
		if sourceIDs := firstNonEmptyStrings(normalizedIDs(req.SourceIDs), parseSourceIDs(message)); len(sourceIDs) > 0 {
			return routedIntent{kind: ActionDeleteSource, sourceIDs: sourceIDs}
		}
	}
	if isListSourcesIntent(message) {
		return routedIntent{kind: ActionListSources}
	}
	if isFormatTextIntent(message) {
		return routedIntent{kind: ActionFormatText, text: textForFormatAction(message)}
	}
	return routedIntent{kind: ActionAnswerFromKnowledge}
}

func (s *Service) routeWithClassifier(ctx context.Context, req TurnRequest) routedIntent {
	if s.classifier == nil {
		return routedIntent{kind: ActionAnswerFromKnowledge}
	}
	result, err := s.classifier.ClassifyIntent(ctx, IntentClassifyRequest{
		UserID:      req.UserID,
		SessionID:   req.SessionID,
		Message:     req.Message,
		SourceIDs:   normalizedIDs(req.SourceIDs),
		DocumentIDs: normalizedIDs(req.DocumentIDs),
	})
	if err != nil {
		return routedIntent{kind: ActionAnswerFromKnowledge}
	}
	return routedIntentFromClassification(result, req)
}

func routedIntentFromClassification(result IntentClassifyResult, req TurnRequest) routedIntent {
	switch result.Type {
	case ActionProcessVideo:
		normalized := videoinput.NormalizeShareInput(firstNonEmpty(result.VideoURL, req.Message), result.VideoName)
		if !looksLikeURL(normalized.URL) {
			return routedIntent{kind: ActionAnswerFromKnowledge}
		}
		return routedIntent{
			kind:       ActionProcessVideo,
			videoURL:   normalized.URL,
			videoName:  firstNonEmpty(normalized.Name, fallbackVideoName(normalized.URL)),
			videoTitle: normalized.Title,
		}
	case ActionIndexTaskTranscript:
		taskID := firstNonEmpty(result.TaskID, parseTaskID(req.Message))
		if taskID == "" {
			return routedIntent{kind: ActionAnswerFromKnowledge}
		}
		return routedIntent{kind: ActionIndexTaskTranscript, taskID: taskID}
	case ActionDeleteSource:
		sourceIDs := firstNonEmptyStrings(normalizedIDs(result.SourceIDs), normalizedIDs(req.SourceIDs), parseSourceIDs(req.Message))
		if len(sourceIDs) == 0 {
			return routedIntent{kind: ActionAnswerFromKnowledge}
		}
		return routedIntent{kind: ActionDeleteSource, sourceIDs: sourceIDs}
	case ActionListSources:
		return routedIntent{kind: ActionListSources}
	case ActionFormatText:
		text := firstNonEmpty(result.Text, textForFormatAction(req.Message))
		if text == "" {
			return routedIntent{kind: ActionAnswerFromKnowledge}
		}
		return routedIntent{kind: ActionFormatText, text: text}
	default:
		return routedIntent{kind: ActionAnswerFromKnowledge}
	}
}

func newProcessVideoAction(videoURL, name string, req TurnRequest) AgentAction {
	return AgentAction{
		Type:                 ActionProcessVideo,
		Title:                "处理视频",
		Summary:              fmt.Sprintf("下载音频、转写并整理：%s", videoURL),
		RequiresConfirmation: true,
		Input: map[string]any{
			"url":        videoURL,
			"name":       name,
			"user_id":    req.UserID,
			"session_id": req.SessionID,
			"language":   "zh",
		},
		RiskLevel: RiskExpensive,
	}
}

func newIndexTaskAction(taskID string) AgentAction {
	return AgentAction{
		Type:                 ActionIndexTaskTranscript,
		Title:                "存入知识库",
		Summary:              fmt.Sprintf("把任务 %s 的格式化转写写入知识库。", taskID),
		RequiresConfirmation: true,
		Input:                map[string]any{"task_id": taskID},
		RiskLevel:            RiskExpensive,
	}
}

func newDeleteSourceAction(sourceIDs []string) AgentAction {
	return AgentAction{
		Type:                 ActionDeleteSource,
		Title:                "删除 source",
		Summary:              fmt.Sprintf("删除 %d 个知识库 source。", len(sourceIDs)),
		RequiresConfirmation: true,
		Input:                map[string]any{"source_ids": sourceIDs},
		RiskLevel:            RiskDestructive,
	}
}

func newFormatTextAction(text string) AgentAction {
	return AgentAction{
		Type:                 ActionFormatText,
		Title:                "整理文本",
		Summary:              "调用文本格式化流程整理用户提供的内容。",
		RequiresConfirmation: true,
		Input:                map[string]any{"text": text},
		RiskLevel:            RiskExpensive,
	}
}

func sourceListAnswer(sources []rag.SourceSummary) string {
	if len(sources) == 0 {
		return "当前范围下还没有知识库 source。"
	}
	lines := []string{fmt.Sprintf("当前范围下有 %d 个知识库 source：", len(sources))}
	limit := len(sources)
	if limit > 8 {
		limit = 8
	}
	for i := 0; i < limit; i++ {
		source := sources[i]
		name := displayName(source.SourceName, source.DocumentTitle, source.SourceID)
		lines = append(lines, fmt.Sprintf("%d. %s（%s，%d chunks）", i+1, name, source.SourceID, source.ChunkCount))
	}
	if len(sources) > limit {
		lines = append(lines, fmt.Sprintf("还有 %d 个 source 未展示。", len(sources)-limit))
	}
	return strings.Join(lines, "\n")
}

func isIndexTaskIntent(message string) bool {
	return containsAny(message, "存入知识库", "入库", "索引", "index")
}

func isDeleteSourceIntent(message string) bool {
	lower := strings.ToLower(message)
	return strings.Contains(message, "删除") || strings.Contains(message, "移除") || strings.Contains(lower, "delete") || strings.Contains(lower, "remove")
}

func isListSourcesIntent(message string) bool {
	lower := strings.ToLower(message)
	if !(strings.Contains(lower, "source") || strings.Contains(message, "知识库") || strings.Contains(message, "资料") || strings.Contains(message, "文档")) {
		return false
	}
	return containsAny(message, "列出", "有哪些", "显示", "查看", "list", "show")
}

func isFormatTextIntent(message string) bool {
	if looksLikeURL(videoinput.NormalizeShareInput(message, "").URL) {
		return false
	}
	return containsAny(message, "格式化", "整理这段", "整理文本", "润色", "format")
}

func textForFormatAction(message string) string {
	if idx := strings.Index(message, "\n"); idx >= 0 {
		return strings.TrimSpace(message[idx+1:])
	}
	return message
}

var (
	taskIDRE           = regexp.MustCompile(`(?i)(?:task[_:\s-]*)?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|task[-_][A-Za-z0-9._-]+)`)
	sourcePrefixedIDRE = regexp.MustCompile(`(?i)\b(source[-_][A-Za-z0-9._-]+)\b`)
	sourceLabeledIDRE  = regexp.MustCompile(`(?i)\bsource(?:[_ -]?id)?\s*[:：=]?\s*([A-Za-z0-9][A-Za-z0-9._-]{5,})\b`)
	stableHashIDRE     = regexp.MustCompile(`(?i)\b([0-9a-f]{64}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b`)
)

func parseTaskID(message string) string {
	match := taskIDRE.FindStringSubmatch(message)
	if len(match) < 2 {
		return ""
	}
	return strings.TrimSpace(match[1])
}

func parseSourceIDs(message string) []string {
	out := make([]string, 0, 2)
	seen := map[string]struct{}{}
	add := func(id string) {
		id = strings.TrimSpace(id)
		if id == "" || !looksLikeSourceID(id) {
			return
		}
		if _, ok := seen[id]; ok {
			return
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	for _, match := range sourcePrefixedIDRE.FindAllStringSubmatch(message, -1) {
		if len(match) >= 2 {
			add(match[1])
		}
	}
	for _, match := range sourceLabeledIDRE.FindAllStringSubmatch(message, -1) {
		if len(match) >= 2 {
			add(match[1])
		}
	}
	if containsAny(message, "source", "source_id", "知识库", "资料", "文档") {
		for _, match := range stableHashIDRE.FindAllStringSubmatch(message, -1) {
			if len(match) >= 2 {
				add(match[1])
			}
		}
	}
	return out
}

func looksLikeSourceID(id string) bool {
	id = strings.TrimSpace(id)
	if id == "" {
		return false
	}
	lower := strings.ToLower(id)
	if strings.HasPrefix(lower, "source-") || strings.HasPrefix(lower, "source_") {
		suffix := strings.TrimPrefix(strings.TrimPrefix(lower, "source-"), "source_")
		return suffix != "" && suffix != "id"
	}
	return stableHashIDRE.MatchString(id)
}

func indexableTaskIDFromRequest(tasks TaskReader, req TurnRequest) string {
	if tasks == nil {
		return ""
	}
	for _, id := range parseCandidateTaskIDs(req.Message) {
		task, ok := tasks.Get(id)
		if !ok {
			continue
		}
		if task.Type == "video_process" && task.Status == string(taskpkg.StatusDone) {
			return task.ID
		}
	}
	return ""
}

func parseCandidateTaskIDs(message string) []string {
	id := parseTaskID(message)
	if id == "" {
		return nil
	}
	return []string{id}
}

func normalizedIDs(ids []string) []string {
	if len(ids) == 0 {
		return nil
	}
	out := make([]string, 0, len(ids))
	seen := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	return out
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			return value
		}
	}
	return ""
}

func firstNonEmptyStrings(values ...[]string) []string {
	for _, value := range values {
		if len(value) > 0 {
			return value
		}
	}
	return nil
}

func containsAny(value string, needles ...string) bool {
	lower := strings.ToLower(value)
	for _, needle := range needles {
		if strings.Contains(lower, strings.ToLower(needle)) {
			return true
		}
	}
	return false
}

func looksLikeURL(value string) bool {
	value = strings.TrimSpace(value)
	return strings.HasPrefix(value, "http://") || strings.HasPrefix(value, "https://")
}

func fallbackVideoName(videoURL string) string {
	name := videoinput.SanitizeName(strings.TrimPrefix(strings.TrimPrefix(videoURL, "https://"), "http://"))
	if name == "" {
		return "video"
	}
	if len([]rune(name)) > 80 {
		runes := []rune(name)
		name = string(runes[:80])
	}
	return name
}

func displayName(values ...string) string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			return value
		}
	}
	return ""
}

func stringInput(input map[string]any, key string) string {
	value, ok := input[key]
	if !ok {
		return ""
	}
	text, ok := value.(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(text)
}

func stringSliceInput(input map[string]any, key string) []string {
	value, ok := input[key]
	if !ok {
		return nil
	}
	switch got := value.(type) {
	case []string:
		return normalizedIDs(got)
	case []any:
		out := make([]string, 0, len(got))
		for _, item := range got {
			if text, ok := item.(string); ok {
				out = append(out, text)
			}
		}
		return normalizedIDs(out)
	default:
		return nil
	}
}

func normalizeSessionStore(store SessionStore) SessionStore {
	if store == nil {
		return nil
	}
	return store
}
