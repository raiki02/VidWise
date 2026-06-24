package tool

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	downloadcmd "github.com/raiki02/vidwise/cmd/download"
)

// AudioExtractInput is the input for the audio extraction tool.
type AudioExtractInput struct {
	URL        string `json:"url" jsonschema:"required" jsonschema_description:"The video URL to extract audio from."`
	OutputBase string `json:"output_base" jsonschema:"required" jsonschema_description:"The base path (without extension) for the output audio file."`
}

type AudioExtractOutput struct {
	AudioPath string `json:"audio_path"`
	Stdout    string `json:"stdout"`
}

func NewAudioExtractTool() (tool.InvokableTool, *Wrapper, error) {
	inner, err := utils.InferTool(
		"extract_audio",
		"Extract audio from a video URL using yt-dlp. Returns the path to the extracted audio file.",
		func(ctx context.Context, input AudioExtractInput) (AudioExtractOutput, error) {
			outputBase := filepath.Clean(input.OutputBase)
			audioPath, stdout, err := downloadcmd.Audio(input.URL, outputBase)
			if err != nil {
				return AudioExtractOutput{}, fmt.Errorf("extract audio: %w", err)
			}
			return AudioExtractOutput{AudioPath: audioPath, Stdout: string(stdout)}, nil
		},
	)
	if err != nil {
		return nil, nil, err
	}
	wrapper := NewWrapper(inner, WrapperConfig{Name: "extract_audio"})
	return inner, wrapper, nil
}
