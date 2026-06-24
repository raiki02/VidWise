package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/cloudwego/eino/components/tool"
	"github.com/raiki02/vidwise/internal/appconfig"
	video_summary "github.com/raiki02/vidwise/internal/video_summary"
)

type VideoSummaryAgent struct {
	cfg           appconfig.Config
	summarizeTool tool.InvokableTool
}

func NewVideoSummaryAgent(cfg appconfig.Config) (*VideoSummaryAgent, error) {
	timeout, err := cfg.VideoSummary.TimeoutDuration()
	if err != nil {
		return nil, fmt.Errorf("invalid video_summary.timeout: %w", err)
	}

	client, err := video_summary.NewClient(cfg.VideoSummary.BaseURL, timeout, video_summary.SummarizeOptions{
		MaxNewTokens: cfg.VideoSummary.Summarize.MaxNewTokens,
		Prompt:       cfg.VideoSummary.Summarize.Prompt,
		DoSample:     cfg.VideoSummary.Summarize.DoSample,
		Temperature:  cfg.VideoSummary.Summarize.Temperature,
		TopP:         cfg.VideoSummary.Summarize.TopP,
	})
	if err != nil {
		return nil, err
	}

	summarizeTool, err := video_summary.NewSummarizeTool(client)
	if err != nil {
		return nil, fmt.Errorf("create video_summary tool failed: %w", err)
	}

	return &VideoSummaryAgent{
		cfg:           cfg,
		summarizeTool: summarizeTool,
	}, nil
}

func (a *VideoSummaryAgent) Run(ctx context.Context, videoPath string) (string, error) {
	start := time.Now()
	args, err := json.Marshal(video_summary.SummarizeInput{
		VideoPath:    videoPath,
		Prompt:       a.cfg.VideoSummary.Summarize.Prompt,
		MaxNewTokens: a.cfg.VideoSummary.Summarize.MaxNewTokens,
	})
	if err != nil {
		return "", fmt.Errorf("encode video_summary tool input failed: %w", err)
	}

	outputJSON, err := a.summarizeTool.InvokableRun(ctx, string(args))
	if err != nil {
		return "", fmt.Errorf("summarize video failed: %w", err)
	}

	var output video_summary.CaptionResponse
	if err := json.Unmarshal([]byte(outputJSON), &output); err != nil {
		return "", fmt.Errorf("decode video_summary tool output failed: %w", err)
	}
	slog.Info("video_summary.done", "elapsed", time.Since(start))
	return formatVideoSummary(output), nil
}

func formatVideoSummary(output video_summary.CaptionResponse) string {
	var b strings.Builder
	if strings.TrimSpace(output.Scene) != "" {
		b.WriteString("Scene:\n")
		b.WriteString(strings.TrimSpace(output.Scene))
		b.WriteString("\n\n")
	}
	if len(output.Events) > 0 {
		b.WriteString("Events:\n")
		for _, event := range output.Events {
			b.WriteString(fmt.Sprintf("<%.1f - %.1f> %s\n", event.Start, event.End, strings.TrimSpace(event.Description)))
		}
	}
	if b.Len() == 0 {
		return strings.TrimSpace(output.Caption)
	}
	return strings.TrimSpace(b.String())
}
