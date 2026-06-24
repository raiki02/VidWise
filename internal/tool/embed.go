package tool

import (
	"context"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/raiki02/vidwise/internal/model"
)

// EmbedInput is the input for the embedding tool.
type EmbedInput struct {
	Texts []string `json:"texts" jsonschema:"required" jsonschema_description:"Array of text strings to embed."`
}

type EmbedOutput struct {
	Embeddings [][]float64 `json:"embeddings"`
}

func NewEmbedTool(client *model.EmbedClient) (tool.InvokableTool, *Wrapper, error) {
	inner, err := utils.InferTool(
		"embed_texts",
		"Generate vector embeddings for text strings using the configured embedding model.",
		func(ctx context.Context, input EmbedInput) (EmbedOutput, error) {
			embeddings, err := client.Embed(ctx, input.Texts)
			if err != nil {
				return EmbedOutput{}, err
			}
			return EmbedOutput{Embeddings: embeddings}, nil
		},
	)
	if err != nil {
		return nil, nil, err
	}
	wrapper := NewWrapper(inner, WrapperConfig{Name: "embed_texts", Timeout: 0})
	return inner, wrapper, nil
}
