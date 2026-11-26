package tool

import (
	"context"
	"fmt"
	"log/slog"
	"path/filepath"
	"time"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/google/uuid"
	downloadcmd "github.com/raiki02/vidwise/cmd/download"
)

// DownloadAudioInput is the input for the download_audio tool.
type DownloadAudioInput struct {
	URL string `json:"url" jsonschema:"required" jsonschema_description:"Video URL (http/https) to download audio from."`
}

type DownloadAudioOutput struct {
	AudioPath string `json:"audio_path" jsonschema_description:"Absolute path of the downloaded mp3 file."`
}

// NewDownloadAudioTool returns the download_audio eino tool. The output path
// is generated server-side inside workDir; the LLM never controls it.
func NewDownloadAudioTool(workDir, cookiesPath string) (tool.InvokableTool, error) {
	inner, err := utils.InferTool(
		"download_audio",
		"Download audio from a video URL using yt-dlp. Returns the absolute path to the downloaded mp3 file.",
		func(ctx context.Context, input DownloadAudioInput) (DownloadAudioOutput, error) {
			if !IsHTTPURL(input.URL) {
				return DownloadAudioOutput{}, fmt.Errorf("url must be an absolute http(s) URL")
			}
			base := filepath.Join(workDir, uuid.NewString())
			start := time.Now()
			audioPath, stdout, err := downloadcmd.Audio(input.URL, base, cookiesPath)
			if err != nil {
				tail := string(stdout)
				if len(tail) > 500 {
					tail = tail[len(tail)-500:]
				}
				return DownloadAudioOutput{}, fmt.Errorf("yt-dlp failed: %w (output: %s)", err, tail)
			}
			slog.Info("tool.download_audio.done", "url", input.URL, "audio_path", audioPath, "elapsed", time.Since(start))
			return DownloadAudioOutput{AudioPath: audioPath}, nil
		},
	)
	if err != nil {
		return nil, err
	}
	return inner, nil
}
