package server

import (
	"sort"
	"strings"

	"github.com/raiki02/vidwise/internal/agent"
	"github.com/raiki02/vidwise/internal/capability"
	"github.com/raiki02/vidwise/internal/tool"
)

const frontendManifestVersion = "2026-07-31"

// FrontendManifest is the runtime contract consumed by the embedded web UI.
// Keep user-facing tabs and backend routes here so API changes have one place
// to update before the UI renders them.
type FrontendManifest struct {
	Version           string                           `json:"version"`
	Capabilities      map[string]capability.Capability `json:"capabilities"`
	Features          []FrontendFeature                `json:"features"`
	ExtractTypes      []ExtractTypeSpec                `json:"extract_types"`
	VideoProcessSteps []string                         `json:"video_process_steps"`
	Tools             []string                         `json:"tools"`
}

type FrontendFeature struct {
	ID                   string                  `json:"id"`
	Title                string                  `json:"title"`
	Tab                  string                  `json:"tab,omitempty"`
	Description          string                  `json:"description,omitempty"`
	RequiredCapabilities []capability.Name       `json:"required_capabilities,omitempty"`
	OptionalCapabilities []capability.Name       `json:"optional_capabilities,omitempty"`
	Routes               []FrontendRoute         `json:"routes,omitempty"`
	Available            bool                    `json:"available"`
	Status               capability.Status       `json:"status"`
	Reason               string                  `json:"reason,omitempty"`
	Blocking             []capability.Capability `json:"blocking,omitempty"`
}

type FrontendRoute struct {
	Method      string `json:"method"`
	Path        string `json:"path"`
	Description string `json:"description,omitempty"`
}

type ExtractTypeSpec struct {
	Value                string                  `json:"value"`
	Label                string                  `json:"label"`
	AliasOf              string                  `json:"alias_of,omitempty"`
	FilenameSuffix       string                  `json:"filename_suffix"`
	Description          string                  `json:"description,omitempty"`
	RequiredCapabilities []capability.Name       `json:"required_capabilities,omitempty"`
	Available            bool                    `json:"available"`
	Status               capability.Status       `json:"status"`
	Reason               string                  `json:"reason,omitempty"`
	Blocking             []capability.Capability `json:"blocking,omitempty"`
}

func BuildFrontendManifest(registry *tool.Registry, caps capability.Snapshot) FrontendManifest {
	return FrontendManifest{
		Version:           frontendManifestVersion,
		Capabilities:      caps.Map(),
		Features:          frontendFeatures(caps),
		ExtractTypes:      extractTypeSpecs(caps),
		VideoProcessSteps: agent.VideoProcessStepNames(),
		Tools:             registeredToolNames(registry),
	}
}

func frontendFeatures(caps capability.Snapshot) []FrontendFeature {
	specs := []FrontendFeature{
		{
			ID:                   "chat",
			Title:                "Agent 问答",
			Tab:                  "chat",
			RequiredCapabilities: []capability.Name{capability.LLM},
			OptionalCapabilities: []capability.Name{capability.RAG, capability.ChatSessionStore, capability.MemoryStore},
			Routes: []FrontendRoute{
				{Method: "POST", Path: "/agent/turn"},
				{Method: "POST", Path: "/agent/actions/:id/confirm"},
				{Method: "POST", Path: "/chat/query"},
				{Method: "POST", Path: "/chat/new"},
				{Method: "GET", Path: "/chat/sessions"},
				{Method: "GET", Path: "/chat/session/:id"},
			},
		},
		{
			ID:                   "video",
			Title:                "视频任务",
			Tab:                  "video",
			RequiredCapabilities: []capability.Name{capability.ASR},
			OptionalCapabilities: []capability.Name{capability.LLM, capability.RAG},
			Routes: []FrontendRoute{
				{Method: "POST", Path: "/video/process"},
			},
		},
		{
			ID:                   "extract",
			Title:                "同步提取",
			Tab:                  "extract",
			OptionalCapabilities: []capability.Name{capability.ASR, capability.VideoSummary, capability.LLM, capability.RAG},
			Routes: []FrontendRoute{
				{Method: "GET", Path: "/extract"},
				{Method: "POST", Path: "/extract"},
			},
		},
		{
			ID:                   "sources",
			Title:                "知识库",
			Tab:                  "sources",
			RequiredCapabilities: []capability.Name{capability.RAG},
			Routes: []FrontendRoute{
				{Method: "GET", Path: "/rag/sources"},
				{Method: "DELETE", Path: "/rag/source/:source_id"},
			},
		},
		{
			ID:                   "upload",
			Title:                "上传索引",
			Tab:                  "upload",
			RequiredCapabilities: []capability.Name{capability.RAG},
			Routes: []FrontendRoute{
				{Method: "POST", Path: "/upload"},
			},
		},
		{
			ID:                   "format",
			Title:                "文本格式化",
			Tab:                  "format",
			RequiredCapabilities: []capability.Name{capability.LLM},
			Routes: []FrontendRoute{
				{Method: "POST", Path: "/format"},
			},
		},
		{
			ID:                   "tasks",
			Title:                "任务列表",
			Tab:                  "tasks",
			OptionalCapabilities: []capability.Name{capability.RAG},
			Routes: []FrontendRoute{
				{Method: "GET", Path: "/tasks"},
				{Method: "GET", Path: "/task/:id"},
				{Method: "POST", Path: "/task/:id/index"},
			},
		},
		{
			ID:                   "memory",
			Title:                "用户记忆",
			Tab:                  "memory",
			RequiredCapabilities: []capability.Name{capability.MemoryStore},
			Routes: []FrontendRoute{
				{Method: "GET", Path: "/user/facts"},
				{Method: "GET", Path: "/user/profile"},
			},
		},
	}

	for i := range specs {
		applyFeatureAvailability(&specs[i], caps)
	}
	return specs
}

func extractTypeSpecs(caps capability.Snapshot) []ExtractTypeSpec {
	specs := []ExtractTypeSpec{
		{Value: "video", Label: "视频 MP4", FilenameSuffix: ".mp4"},
		{Value: "audio", Label: "音频 MP3", FilenameSuffix: ".mp3"},
		{Value: "text", Label: "转写文本", FilenameSuffix: ".txt", RequiredCapabilities: []capability.Name{capability.ASR}},
		{Value: "transcript", Label: "转写文本", AliasOf: "text", FilenameSuffix: ".txt", RequiredCapabilities: []capability.Name{capability.ASR}},
		{Value: "summary", Label: "视频摘要", FilenameSuffix: "_summary.txt", RequiredCapabilities: []capability.Name{capability.VideoSummary}},
		{Value: "video_summary", Label: "视频摘要", AliasOf: "summary", FilenameSuffix: "_summary.txt", RequiredCapabilities: []capability.Name{capability.VideoSummary}},
	}
	for i := range specs {
		status, available, reason, blocking := availability(specs[i].RequiredCapabilities, nil, caps)
		specs[i].Status = status
		specs[i].Available = available
		specs[i].Reason = reason
		specs[i].Blocking = blocking
	}
	return specs
}

func applyFeatureAvailability(feature *FrontendFeature, caps capability.Snapshot) {
	status, available, reason, blocking := availability(feature.RequiredCapabilities, feature.OptionalCapabilities, caps)
	feature.Status = status
	feature.Available = available
	feature.Reason = reason
	feature.Blocking = blocking
}

func availability(required, optional []capability.Name, caps capability.Snapshot) (capability.Status, bool, string, []capability.Capability) {
	status := capability.Available
	reasons := make([]string, 0)
	blocking := make([]capability.Capability, 0)

	for _, name := range required {
		c := caps.Get(name)
		switch c.Status {
		case capability.Unavailable:
			status = capability.Unavailable
			blocking = append(blocking, c)
			reasons = append(reasons, capabilityReason(c))
		case capability.Degraded:
			if status != capability.Unavailable {
				status = capability.Degraded
			}
			reasons = append(reasons, capabilityReason(c))
		}
	}

	if status != capability.Unavailable {
		for _, name := range optional {
			c := caps.Get(name)
			if c.Status != capability.Available {
				status = capability.Degraded
				reasons = append(reasons, capabilityReason(c))
			}
		}
	}

	return status, status != capability.Unavailable, strings.Join(uniqueStrings(reasons), "; "), blocking
}

func capabilityReason(c capability.Capability) string {
	if strings.TrimSpace(c.Reason) != "" {
		return string(c.Name) + ": " + c.Reason
	}
	return string(c.Name) + ": " + string(c.Status)
}

func uniqueStrings(values []string) []string {
	out := make([]string, 0, len(values))
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}

func registeredToolNames(registry *tool.Registry) []string {
	if registry == nil {
		return nil
	}
	entries := registry.List()
	names := make([]string, 0, len(entries))
	for name := range entries {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
