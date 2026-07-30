package knowledgeagent

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/cloudwego/eino/schema"
	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/chatagent"
	"github.com/raiki02/vidwise/internal/paragraph"
)

type LLMIntentClassifier struct {
	llmCfg   appconfig.LLMConfig
	newModel chatagent.ModelFactory
}

func NewLLMIntentClassifier(cfg appconfig.LLMConfig) *LLMIntentClassifier {
	return NewLLMIntentClassifierWithModelFactory(cfg, defaultIntentModelFactory)
}

func NewLLMIntentClassifierWithModelFactory(cfg appconfig.LLMConfig, factory chatagent.ModelFactory) *LLMIntentClassifier {
	if factory == nil {
		factory = defaultIntentModelFactory
	}
	return &LLMIntentClassifier{
		llmCfg:   cfg,
		newModel: factory,
	}
}

func defaultIntentModelFactory(ctx context.Context, cfg appconfig.LLMConfig) (chatagent.ChatModel, error) {
	return paragraph.NewChatModel(ctx, cfg)
}

func (c *LLMIntentClassifier) ClassifyIntent(ctx context.Context, req IntentClassifyRequest) (IntentClassifyResult, error) {
	if c == nil || !llmIntentEnabled(c.llmCfg) {
		return IntentClassifyResult{Type: ActionAnswerFromKnowledge}, nil
	}
	cm, err := c.newModel(ctx, c.llmCfg)
	if err != nil {
		slog.Warn("knowledge_agent.intent_llm_unavailable", "err", err)
		return IntentClassifyResult{Type: ActionAnswerFromKnowledge}, nil
	}
	resp, err := cm.Generate(ctx, buildIntentClassificationMessages(req))
	if err != nil || resp == nil || strings.TrimSpace(resp.Content) == "" {
		slog.Warn("knowledge_agent.intent_llm_failed", "err", err)
		return IntentClassifyResult{Type: ActionAnswerFromKnowledge}, nil
	}

	var decision llmIntentDecision
	if err := extractIntentJSON(resp.Content, &decision); err != nil {
		slog.Warn("knowledge_agent.intent_llm_parse_failed", "content", resp.Content, "err", err)
		return IntentClassifyResult{Type: ActionAnswerFromKnowledge}, nil
	}
	return decision.result(), nil
}

type llmIntentDecision struct {
	ActionType string   `json:"action_type"`
	VideoURL   string   `json:"video_url"`
	VideoName  string   `json:"video_name"`
	TaskID     string   `json:"task_id"`
	SourceID   string   `json:"source_id"`
	SourceIDs  []string `json:"source_ids"`
	Text       string   `json:"text"`
	Confidence float64  `json:"confidence"`
	Reason     string   `json:"reason"`
}

func (d llmIntentDecision) result() IntentClassifyResult {
	actionType := ActionType(strings.TrimSpace(d.ActionType))
	if !isSupportedClassifiedAction(actionType) {
		actionType = ActionAnswerFromKnowledge
	}
	if d.Confidence > 0 && d.Confidence < 0.5 {
		actionType = ActionAnswerFromKnowledge
	}
	sourceIDs := append([]string{d.SourceID}, d.SourceIDs...)
	return IntentClassifyResult{
		Type:       actionType,
		VideoURL:   strings.TrimSpace(d.VideoURL),
		VideoName:  strings.TrimSpace(d.VideoName),
		TaskID:     strings.TrimSpace(d.TaskID),
		SourceIDs:  normalizedIDs(sourceIDs),
		Text:       strings.TrimSpace(d.Text),
		Confidence: d.Confidence,
		Reason:     strings.TrimSpace(d.Reason),
	}
}

func isSupportedClassifiedAction(actionType ActionType) bool {
	switch actionType {
	case ActionAnswerFromKnowledge,
		ActionProcessVideo,
		ActionIndexTaskTranscript,
		ActionListSources,
		ActionDeleteSource,
		ActionFormatText:
		return true
	default:
		return false
	}
}

func llmIntentEnabled(cfg appconfig.LLMConfig) bool {
	return cfg.Enabled != nil && *cfg.Enabled && strings.TrimSpace(cfg.Model) != ""
}

func buildIntentClassificationMessages(req IntentClassifyRequest) []*schema.Message {
	systemPrompt := strings.Join([]string{
		"你是 VidWise 视频知识助手的意图分类器，只负责把用户消息路由到固定动作。",
		"只输出一个 JSON object，不要解释。",
		"action_type 只能是 answer_from_knowledge、process_video、index_task_transcript、list_sources、delete_source、format_text。",
		"如果不确定，或动作缺少必要输入，使用 answer_from_knowledge。",
		"process_video 必须有视频 URL；index_task_transcript 必须有 task_id；delete_source 必须是明确删除意图且有 source_id/source_ids；format_text 必须是明确整理文本意图且有待整理文本；list_sources 只用于列出/查看知识库 sources。",
		"不要把普通问题、闲聊、总结问题分类成昂贵或破坏性动作。",
		"返回字段：action_type, video_url, video_name, task_id, source_ids, text, confidence, reason。",
	}, "\n")
	userPrompt := fmt.Sprintf(
		"用户 ID: %s\nSession ID: %s\n当前 source_ids: %s\n当前 document_ids: %s\n用户消息:\n%s",
		strings.TrimSpace(req.UserID),
		strings.TrimSpace(req.SessionID),
		strings.Join(normalizedIDs(req.SourceIDs), ", "),
		strings.Join(normalizedIDs(req.DocumentIDs), ", "),
		strings.TrimSpace(req.Message),
	)
	return []*schema.Message{
		schema.SystemMessage(systemPrompt),
		schema.UserMessage(userPrompt),
	}
}

func extractIntentJSON(content string, v interface{}) error {
	content = strings.TrimSpace(content)
	if err := json.Unmarshal([]byte(content), v); err == nil {
		return nil
	}

	start := -1
	var stack []rune
	inString := false
	escaped := false
	var firstErr error
	for i, ch := range content {
		if start == -1 {
			if ch == '{' || ch == '[' {
				start = i
				if ch == '{' {
					stack = []rune{'}'}
				} else {
					stack = []rune{']'}
				}
			}
			continue
		}
		if inString {
			if escaped {
				escaped = false
				continue
			}
			switch ch {
			case '\\':
				escaped = true
			case '"':
				inString = false
			}
			continue
		}
		switch ch {
		case '"':
			inString = true
		case '{':
			stack = append(stack, '}')
		case '[':
			stack = append(stack, ']')
		case '}', ']':
			if len(stack) == 0 || ch != stack[len(stack)-1] {
				start = -1
				stack = nil
				continue
			}
			stack = stack[:len(stack)-1]
			if len(stack) == 0 {
				if err := json.Unmarshal([]byte(content[start:i+1]), v); err == nil {
					return nil
				} else if firstErr == nil {
					firstErr = err
				}
				start = -1
			}
		}
	}
	if firstErr != nil {
		return firstErr
	}
	return fmt.Errorf("no JSON found in LLM intent response")
}
