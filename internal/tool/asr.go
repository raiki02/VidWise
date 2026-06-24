package tool

import (
	"context"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/raiki02/video-extractor/internal/asr"
)

func NewASRTool(client *asr.Client) (tool.InvokableTool, *Wrapper, error) {
	inner, err := utils.InferTool(
		"transcribe_audio",
		"Transcribe a local audio file to text by calling the ASR service. Returns the full transcript with segments.",
		func(ctx context.Context, input asr.TranscribeInput) (asr.TranscribeResponse, error) {
			return client.Transcribe(ctx, input.AudioPath, input.Language)
		},
	)
	if err != nil {
		return nil, nil, err
	}
	wrapper := NewWrapper(inner, WrapperConfig{Name: "transcribe_audio", Timeout: 0}) // ASR timeout from config
	return inner, wrapper, nil
}
