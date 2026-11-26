package agent

import (
	"context"
	"errors"

	einomodel "github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/flow/agent/react"
	"github.com/cloudwego/eino/schema"
)

// SystemPrompt instructs the model on the video processing pipeline and how
// to use the tools.
const SystemPrompt = `你是一个视频知识助手，可以处理视频并回答关于视频内容的问题。

可用工具：
- download_audio: 从视频 URL 下载音频。
- transcribe_audio: 将本地音频转写为文本（返回转写文件路径和预览）。
- index_transcript: 将转写文本切分、向量化并存入知识库。
- search_knowledge: 在知识库中检索与问题相关的片段。

工作流程：
1. 当用户给出视频链接（可能是分享文案）时，依次调用 download_audio → transcribe_audio → index_transcript，然后根据转写预览简要总结视频内容，并告诉用户视频已入库，可以继续提问。
2. 当用户询问视频相关内容时，先调用 search_knowledge 检索相关片段，只依据检索结果回答，并标注来源名称。检索结果为空时如实告知，不要编造内容。
3. 文件路径只能来自工具返回值，绝不能自己猜测或编造路径。
4. 回答使用中文，保持简洁。`

// New builds a ReAct agent with the given tool-calling model and tools.
func New(ctx context.Context, cm einomodel.ToolCallingChatModel, tools []tool.BaseTool, maxStep int) (*react.Agent, error) {
	if cm == nil {
		return nil, errors.New("tool-calling chat model is required")
	}
	if len(tools) == 0 {
		return nil, errors.New("at least one tool is required")
	}
	if maxStep <= 0 {
		maxStep = 20
	}
	return react.NewAgent(ctx, &react.AgentConfig{
		ToolCallingModel: cm,
		ToolsConfig: compose.ToolsNodeConfig{
			Tools:               tools,
			ExecuteSequentially: false,
		},
		MessageModifier: func(_ context.Context, in []*schema.Message) []*schema.Message {
			messages := make([]*schema.Message, 0, len(in)+1)
			messages = append(messages, schema.SystemMessage(SystemPrompt))
			return append(messages, in...)
		},
		MaxStep: maxStep,
	})
}
