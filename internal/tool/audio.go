package tool

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	downloadcmd "github.com/raiki02/vidwise/cmd/download"
)

// AudioExtractInput is the input for the audio extraction tool.
type AudioExtractInput struct {
	URL         string `json:"url" jsonschema:"required" jsonschema_description:"The video URL to extract audio from."`
	OutputBase  string `json:"output_base" jsonschema:"required" jsonschema_description:"The base path (without extension) for the output audio file."`
	CookiesPath string `json:"cookies_path,omitempty" jsonschema_description:"Optional path to a cookies.txt file for authenticated downloads. When omitted, the configured download cookies path is used."`
}

type AudioExtractOutput struct {
	AudioPath string `json:"audio_path"`
	Stdout    string `json:"stdout"`
}

func NewAudioExtractTool(cookiesPath ...string) (tool.InvokableTool, *Wrapper, error) {
	return newAudioDownloadTool("extract_audio", "Extract audio from a video URL using yt-dlp. Returns the path to the extracted audio file.", optionalString(cookiesPath))
}

func NewAudioDownloadTool(cookiesPath ...string) (tool.InvokableTool, *Wrapper, error) {
	return newAudioDownloadTool("download_audio", "Download audio from a video URL using yt-dlp. Returns the path to the downloaded audio file.", optionalString(cookiesPath))
}

func newAudioDownloadTool(name, description, defaultCookiesPath string) (tool.InvokableTool, *Wrapper, error) {
	inner, err := utils.InferTool(
		name,
		description,
		func(ctx context.Context, input AudioExtractInput) (AudioExtractOutput, error) {
			outputBase := filepath.Clean(input.OutputBase)
			audioPath, stdout, err := downloadcmd.Audio(input.URL, outputBase, firstNonEmpty(input.CookiesPath, defaultCookiesPath))
			if err != nil {
				return AudioExtractOutput{}, fmt.Errorf("extract audio: %w", err)
			}
			return AudioExtractOutput{AudioPath: audioPath, Stdout: string(stdout)}, nil
		},
	)
	if err != nil {
		return nil, nil, err
	}
	wrapper := NewWrapper(inner, WrapperConfig{Name: name})
	return inner, wrapper, nil
}

func optionalString(values []string) string {
	if len(values) == 0 {
		return ""
	}
	return values[0]
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
