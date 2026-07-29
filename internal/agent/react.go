package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"

	einomodel "github.com/cloudwego/eino/components/model"
	einotool "github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/flow/agent/react"
	"github.com/cloudwego/eino/schema"
	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/paragraph"
	"github.com/raiki02/vidwise/internal/search"
	webtools "github.com/raiki02/vidwise/internal/tools"
)

const WebSearchAgentInstruction = `你是一个支持联网搜索的视频知识库助手。

当用户的问题涉及最近、最新、新闻、发布、外部资料、官方文档或当前信息时，可以调用 web_search。
调用 web_search 时只传 query 和 freshness，其他搜索、抓取、排序和压缩策略由系统内部决定。
回答时必须基于工具返回的 sources，并尽量引用来源 URL。`

func NewWebSearchReActAgent(ctx context.Context, cfg appconfig.LLMConfig, service search.SearchService) (*react.Agent, error) {
	base, err := paragraph.NewChatModel(ctx, cfg)
	if err != nil {
		return nil, fmtAgentError("create chat model", err)
	}
	toolCallingModel, ok := base.(einomodel.ToolCallingChatModel)
	if !ok {
		return nil, errors.New("configured chat model does not support tool calling")
	}
	searchTool, err := webtools.NewWebSearchTool(service)
	if err != nil {
		return nil, fmtAgentError("create web search tool", err)
	}
	return NewReActAgent(ctx, toolCallingModel, []einotool.BaseTool{searchTool}, WebSearchAgentInstruction)
}

func NewReActAgent(ctx context.Context, model einomodel.ToolCallingChatModel, tools []einotool.BaseTool, instruction string) (*react.Agent, error) {
	if model == nil {
		return nil, errors.New("tool-calling chat model is required")
	}
	if len(tools) == 0 {
		return nil, errors.New("at least one tool is required")
	}
	agent, err := react.NewAgent(ctx, &react.AgentConfig{
		ToolCallingModel: model,
		ToolsConfig: compose.ToolsNodeConfig{
			Tools:               tools,
			ExecuteSequentially: false,
		},
		MessageModifier: prependSystemInstruction(instruction),
		MaxStep:         8,
	})
	if err != nil {
		return nil, fmtAgentError("create react agent", err)
	}
	return agent, nil
}

func prependSystemInstruction(instruction string) react.MessageModifier {
	instruction = strings.TrimSpace(instruction)
	return func(ctx context.Context, input []*schema.Message) []*schema.Message {
		if instruction == "" {
			return input
		}
		out := make([]*schema.Message, 0, len(input)+1)
		out = append(out, schema.SystemMessage(instruction))
		out = append(out, input...)
		return out
	}
}

func fmtAgentError(action string, err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("%s: %w", action, err)
}
