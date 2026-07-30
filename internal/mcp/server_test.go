package mcp

import (
	"encoding/json"
	"testing"

	einoschema "github.com/cloudwego/eino/schema"
	"github.com/raiki02/vidwise/internal/tool"
)

func TestToolToMCPUsesToolInfoDescriptionAndSchema(t *testing.T) {
	info := &einoschema.ToolInfo{
		Name: "rag_query",
		Desc: "Search the user's knowledge base.",
		ParamsOneOf: einoschema.NewParamsOneOfByParams(map[string]*einoschema.ParameterInfo{
			"query": {
				Type:     einoschema.String,
				Desc:     "Search query.",
				Required: true,
			},
			"top_k": {
				Type: einoschema.Integer,
				Desc: "Maximum number of chunks to return.",
			},
		}),
	}

	got := toolToMCP("rag_query", tool.Entry{Info: info})

	if got.Description != info.Desc {
		t.Fatalf("Description = %q, want %q", got.Description, info.Desc)
	}
	if len(got.RawInputSchema) == 0 {
		t.Fatal("RawInputSchema is empty")
	}
	if got.InputSchema.Type != "" {
		t.Fatalf("InputSchema.Type = %q, want empty when RawInputSchema is set", got.InputSchema.Type)
	}

	var raw map[string]any
	if err := json.Unmarshal(got.RawInputSchema, &raw); err != nil {
		t.Fatalf("RawInputSchema is not valid JSON: %v", err)
	}
	if raw["type"] != "object" {
		t.Fatalf("schema type = %v, want object", raw["type"])
	}
	required, ok := raw["required"].([]any)
	if !ok || len(required) != 1 || required[0] != "query" {
		t.Fatalf("required = %#v, want [query]", raw["required"])
	}
	properties, ok := raw["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties = %#v, want object", raw["properties"])
	}
	querySchema, ok := properties["query"].(map[string]any)
	if !ok {
		t.Fatalf("query property = %#v, want object", properties["query"])
	}
	if querySchema["type"] != "string" {
		t.Fatalf("query type = %v, want string", querySchema["type"])
	}
}

func TestToolToMCPFallsBackToEmptyObjectSchema(t *testing.T) {
	got := toolToMCP("unknown_tool", tool.Entry{})

	if got.Description != "Tool: unknown_tool" {
		t.Fatalf("Description = %q, want fallback", got.Description)
	}
	if len(got.RawInputSchema) != 0 {
		t.Fatalf("RawInputSchema = %s, want empty", string(got.RawInputSchema))
	}
	if got.InputSchema.Type != "object" {
		t.Fatalf("InputSchema.Type = %q, want object", got.InputSchema.Type)
	}
	if len(got.InputSchema.Properties) != 0 {
		t.Fatalf("InputSchema.Properties = %#v, want empty", got.InputSchema.Properties)
	}
}
