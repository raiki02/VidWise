package tool

import (
	"context"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	video_summary "github.com/raiki02/vidwise/internal/video_summary"
)

func NewVideoSummaryTool(client *video_summary.Client) (tool.InvokableTool, *Wrapper, error) {
	inner, err := utils.InferTool(
		"summarize_video",
		"Summarize a local video by calling the video understanding service. Returns caption, scene description, and timed events.",
		func(ctx context.Context, input video_summary.SummarizeInput) (video_summary.CaptionResponse, error) {
			return client.Caption(ctx, input.VideoPath, video_summary.SummarizeOptions{
				Prompt:       input.Prompt,
				MaxNewTokens: input.MaxNewTokens,
			})
		},
	)
	if err != nil {
		return nil, nil, err
	}
	wrapper := NewWrapper(inner, WrapperConfig{Name: "summarize_video", Timeout: 0})
	return inner, wrapper, nil
}
