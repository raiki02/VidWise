package qdrant

import "testing"

func TestPayloadIndexesIncludeTraceableSourceMetadata(t *testing.T) {
	got := map[string]bool{}
	for _, idx := range payloadIndexes() {
		got[idx.field] = true
	}

	for _, field := range []string{
		FieldTaskID,
		FieldSourceID,
		FieldSourceName,
		FieldSourceURL,
		FieldDocumentID,
		FieldContentHash,
		FieldChunkID,
		FieldDocumentTitle,
		FieldHeadingPath,
		FieldHeadingLevel,
		FieldSectionIndex,
		FieldSectionChunkIndex,
		FieldChunkSource,
		FieldHeader1,
		FieldHeader2,
		FieldHeader3,
		FieldHeader4,
		FieldHeader5,
		FieldHeader6,
	} {
		if !got[field] {
			t.Fatalf("expected payload index for %q", field)
		}
	}
}
