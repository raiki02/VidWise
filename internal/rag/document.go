package rag

// Document is the Go equivalent of a LangChain-style document: searchable
// content plus metadata that describes where the content came from.
type Document struct {
	PageContent string            `json:"page_content"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

func cloneMetadata(metadata map[string]string) map[string]string {
	if len(metadata) == 0 {
		return map[string]string{}
	}
	out := make(map[string]string, len(metadata))
	for k, v := range metadata {
		if k == "" || v == "" {
			continue
		}
		out[k] = v
	}
	return out
}

func mergeMetadata(base map[string]string, extra map[string]string) map[string]string {
	out := cloneMetadata(base)
	for k, v := range extra {
		if k == "" || v == "" {
			continue
		}
		out[k] = v
	}
	return out
}
