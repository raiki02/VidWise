package rag

import (
	"testing"

	pb "github.com/qdrant/go-client/qdrant"
	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
)

func TestBuildChunkPayloadProtectsSystemFieldsFromMetadata(t *testing.T) {
	chunk := documentChunk{
		Text: "actual chunk text",
		Metadata: map[string]string{
			qdrantclient.FieldText:         "metadata text",
			qdrantclient.FieldChunkIdx:     "99",
			qdrantclient.FieldSourceID:     "metadata-source",
			qdrantclient.FieldDocumentID:   "metadata-document",
			qdrantclient.FieldContentHash:  "metadata-hash",
			qdrantclient.FieldChunkID:      "metadata-chunk",
			qdrantclient.FieldUserID:       "metadata-user",
			qdrantclient.FieldSessionID:    "metadata-session",
			qdrantclient.FieldSourceName:   " guide.md ",
			qdrantclient.FieldSectionIndex: " 3 ",
		},
	}
	identity := chunkIdentity{
		sourceID:    "computed-source",
		documentID:  "computed-document",
		contentHash: "computed-hash",
		chunkID:     "computed-chunk",
	}

	payload := buildChunkPayload(chunk, identity, 7, " user-1 ", " session-1 ")

	assertPayloadString(t, payload, qdrantclient.FieldText, "actual chunk text")
	assertPayloadInt(t, payload, qdrantclient.FieldChunkIdx, 7)
	assertPayloadString(t, payload, qdrantclient.FieldSourceID, "computed-source")
	assertPayloadString(t, payload, qdrantclient.FieldDocumentID, "computed-document")
	assertPayloadString(t, payload, qdrantclient.FieldContentHash, "computed-hash")
	assertPayloadString(t, payload, qdrantclient.FieldChunkID, "computed-chunk")
	assertPayloadString(t, payload, qdrantclient.FieldUserID, "user-1")
	assertPayloadString(t, payload, qdrantclient.FieldSessionID, "session-1")
	assertPayloadString(t, payload, qdrantclient.FieldSourceName, "guide.md")
	assertPayloadInt(t, payload, qdrantclient.FieldSectionIndex, 3)
}

func assertPayloadString(t *testing.T, payload map[string]*pb.Value, key, want string) {
	t.Helper()
	got := payload[key]
	if got == nil {
		t.Fatalf("payload[%s] missing", key)
	}
	if got.GetStringValue() != want {
		t.Fatalf("payload[%s] = %q, want %q", key, got.GetStringValue(), want)
	}
}

func assertPayloadInt(t *testing.T, payload map[string]*pb.Value, key string, want int64) {
	t.Helper()
	got := payload[key]
	if got == nil {
		t.Fatalf("payload[%s] missing", key)
	}
	if got.GetIntegerValue() != want {
		t.Fatalf("payload[%s] = %d, want %d", key, got.GetIntegerValue(), want)
	}
}
