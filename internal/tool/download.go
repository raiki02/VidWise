package tool

import (
	"context"
	"fmt"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	downloadcmd "github.com/raiki02/video-extractor/cmd/download"
)

// DownloadInput is the input for the video download tool.
type DownloadInput struct {
	URL         string `json:"url" jsonschema:"required" jsonschema_description:"The video URL to download."`
	OutputPath  string `json:"output_path" jsonschema:"required" jsonschema_description:"The local file path for the downloaded video."`
	CookiesPath string `json:"cookies_path,omitempty" jsonschema_description:"Optional path to a cookies.txt file for authenticated downloads."`
}

type DownloadOutput struct {
	OutputPath string `json:"output_path"`
	Stdout     string `json:"stdout"`
}

func NewDownloadTool() (tool.InvokableTool, *Wrapper, error) {
	inner, err := utils.InferTool(
		"download_video",
		"Download a video from a URL using yt-dlp. Returns the output file path.",
		func(ctx context.Context, input DownloadInput) (DownloadOutput, error) {
			stdout, err := downloadcmd.Video(input.URL, input.OutputPath, input.CookiesPath)
			if err != nil {
				return DownloadOutput{}, fmt.Errorf("download video: %w", err)
			}
			return DownloadOutput{OutputPath: input.OutputPath, Stdout: string(stdout)}, nil
		},
	)
	if err != nil {
		return nil, nil, err
	}
	wrapper := NewWrapper(inner, WrapperConfig{Name: "download_video"})
	return inner, wrapper, nil
}
