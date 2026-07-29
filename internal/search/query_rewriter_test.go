package search

import (
	"context"
	"testing"

	einomodel "github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
)

func TestLLMQueryRewriterParsesJSONArrayFromModel(t *testing.T) {
	rewriter, err := NewLLMQueryRewriter(func(context.Context) (QueryRewriteChatModel, error) {
		return fakeRewriteModel{content: `Here: ["OpenAI latest announcement", "OpenAI new model release", "OpenAI latest announcement"]`}, nil
	}, 3)
	if err != nil {
		t.Fatalf("NewLLMQueryRewriter() error = %v", err)
	}

	queries, err := rewriter.Rewrite(context.Background(), "最近OpenAI有什么消息")
	if err != nil {
		t.Fatalf("Rewrite() error = %v", err)
	}
	if len(queries) != 2 {
		t.Fatalf("queries len = %d, want 2: %#v", len(queries), queries)
	}
	if queries[0] != "OpenAI latest announcement" || queries[1] != "OpenAI new model release" {
		t.Fatalf("queries = %#v", queries)
	}
}

type fakeRewriteModel struct {
	content string
}

func (m fakeRewriteModel) Generate(context.Context, []*schema.Message, ...einomodel.Option) (*schema.Message, error) {
	return schema.AssistantMessage(m.content, nil), nil
}
