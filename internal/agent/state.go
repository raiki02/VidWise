package agent

import (
	"github.com/cloudwego/eino/schema"
)

// AgentState holds the conversational and task state for an agent run.
type AgentState struct {
	UserID    string           `json:"user_id"`
	SessionID string           `json:"session_id"`
	TaskID    string           `json:"task_id"`
	TraceID   string           `json:"trace_id"`
	Memory    []schema.Message `json:"memory"`
	Steps     []StepResult     `json:"steps"`
}

// StepResult captures the output of a tool execution step.
type StepResult struct {
	StepName string `json:"step_name"`
	Output   string `json:"output"`
	Error    string `json:"error,omitempty"`
}

// NewAgentState creates an initial agent state with required IDs.
func NewAgentState(userID, sessionID, taskID, traceID string) *AgentState {
	return &AgentState{
		UserID:    userID,
		SessionID: sessionID,
		TaskID:    taskID,
		TraceID:   traceID,
		Memory:    make([]schema.Message, 0),
		Steps:     make([]StepResult, 0),
	}
}

// AddMessage appends a message to the agent's memory.
func (s *AgentState) AddMessage(msg schema.Message) {
	s.Memory = append(s.Memory, msg)
}

// AddStep records a completed step.
func (s *AgentState) AddStep(result StepResult) {
	s.Steps = append(s.Steps, result)
}
