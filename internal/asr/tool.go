package asr

import (
	"context"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
)

type TranscribeInput struct {
	AudioPath string `json:"audio_path" jsonschema:"required" jsonschema_description:"Local audio file path to transcribe."`
	Language  string `json:"language,omitempty" jsonschema_description:"BCP-47 language code. Defaults to the configured ASR language."`
}

func NewTranscribeTool(client *Client) (tool.InvokableTool, error) {
	return utils.InferTool(
		"transcribe_audio",
		"Transcribe a local audio file to text by calling the ASR service.",
		func(ctx context.Context, input TranscribeInput) (TranscribeResponse, error) {
			return client.Transcribe(ctx, input.AudioPath, input.Language)
		},
	)
}
