package handler

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/background"
	"github.com/raiki02/vidwise/internal/capability"
	"github.com/raiki02/vidwise/internal/extractor"
	"github.com/raiki02/vidwise/internal/paragraph"
	"github.com/raiki02/vidwise/internal/rag"
	"github.com/raiki02/vidwise/internal/tool"
)

// ExtractHandler retains backward compatibility with the legacy /extract and /format endpoints.
type ExtractHandler struct {
	cfg        appconfig.Config
	extractor  *extractor.Service
	sources    *rag.SourceManager
	registry   *tool.Registry
	caps       capability.Snapshot
	background *background.Runner
}

func NewExtractHandler(cfg appconfig.Config, registry *tool.Registry, indexer *rag.Indexer, caps capability.Snapshot) *ExtractHandler {
	return NewExtractHandlerWithBackground(cfg, registry, indexer, caps, nil)
}

func NewExtractHandlerWithBackground(cfg appconfig.Config, registry *tool.Registry, indexer *rag.Indexer, caps capability.Snapshot, runner *background.Runner) *ExtractHandler {
	return NewExtractHandlerWithCatalogAndBackground(cfg, registry, indexer, nil, caps, runner)
}

func NewExtractHandlerWithCatalogAndBackground(cfg appconfig.Config, registry *tool.Registry, indexer *rag.Indexer, catalog *rag.SourceCatalog, caps capability.Snapshot, runner *background.Runner) *ExtractHandler {
	return NewExtractHandlerWithRAGAndBackground(cfg, registry, indexer, catalog, nil, caps, runner)
}

func NewExtractHandlerWithRAGAndBackground(cfg appconfig.Config, registry *tool.Registry, indexer *rag.Indexer, catalog *rag.SourceCatalog, sourceRepo rag.SourceRegistry, caps capability.Snapshot, runner *background.Runner) *ExtractHandler {
	return NewExtractHandlerWithSourceManagerAndBackground(cfg, registry, rag.NewSourceManager(indexer, catalog, sourceRepo), caps, runner)
}

func NewExtractHandlerWithSourceManagerAndBackground(cfg appconfig.Config, registry *tool.Registry, sources *rag.SourceManager, caps capability.Snapshot, runner *background.Runner) *ExtractHandler {
	if runner == nil {
		runner = background.NewRunner(30 * time.Second)
	}
	return &ExtractHandler{
		cfg:        cfg,
		extractor:  extractor.NewService(cfg),
		sources:    sources,
		registry:   registry,
		caps:       caps,
		background: runner,
	}
}

// Extract handles GET/POST /extract (legacy, synchronous).
func (h *ExtractHandler) Extract(c *gin.Context) {
	req, err := bindExtractRequest(c)
	if err != nil {
		errorJSON(c, http.StatusBadRequest, err.Error())
		return
	}

	if cap, ok := h.unavailableExtractCapability(req.Type); ok {
		errorJSONWithFields(c, http.StatusServiceUnavailable, extractCapabilityError(cap.Name), gin.H{"capability": cap})
		return
	}

	result, cleanup, err := h.extractor.Extract(c.Request.Context(), req.URL, req.Name, req.Type)
	if cleanup != nil {
		defer cleanup()
	}
	if err != nil {
		errorJSON(c, statusForExtractError(err), err.Error())
		return
	}

	// For text/transcript extractions, also index into RAG knowledge base.
	if (req.Type == "text" || req.Type == "transcript") && h.ragIndexingAvailable() {
		raw, readErr := readTextFile(result.Path)
		if readErr == nil && raw != "" {
			scope, scopeErr := strictRAGScopeFromRequest(c)
			if scopeErr != nil {
				slog.Info("extract.rag_index_skipped", "name", req.Name, "reason", scopeErr)
			} else {
				filename := result.Filename
				name := req.Name
				h.background.Go("extract.rag_index", func(ctx context.Context) {
					result, idxErr := h.sources.IndexSourceScoped(ctx, rag.Source{
						Text:     raw,
						Filename: filename,
						Format:   rag.ContentFormatPlain,
					}, h.indexOptions(), scope.UserID, scope.SessionID)
					if idxErr != nil {
						slog.Warn("extract.rag_index_failed", "name", name, "err", idxErr)
					} else {
						slog.Info("extract.rag_index_done", "name", name, "content_type", result.ContentType, "chunks", result.ChunkCount)
					}
				})
			}
		}
	}

	c.FileAttachment(result.Path, result.Filename)
}

func (h *ExtractHandler) unavailableExtractCapability(extractType string) (capability.Capability, bool) {
	name, required := requiredExtractCapability(extractType)
	if !required {
		return capability.Capability{}, false
	}
	cap := h.caps.Get(name)
	return cap, !h.caps.Usable(name)
}

func requiredExtractCapability(extractType string) (capability.Name, bool) {
	switch strings.ToLower(strings.TrimSpace(extractType)) {
	case "text", "transcript":
		return capability.ASR, true
	case "summary", "video_summary":
		return capability.VideoSummary, true
	default:
		return "", false
	}
}

func extractCapabilityError(name capability.Name) string {
	switch name {
	case capability.ASR:
		return "ASR service is not available"
	case capability.VideoSummary:
		return "video summary service is not available"
	default:
		return "required service is not available"
	}
}

// FormatText handles POST /format (legacy, synchronous text formatting).
func (h *ExtractHandler) FormatText(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		errorJSON(c, http.StatusBadRequest, "file is required")
		return
	}

	f, err := file.Open()
	if err != nil {
		errorJSON(c, http.StatusInternalServerError, "cannot open uploaded file")
		return
	}
	defer f.Close()

	raw, err := io.ReadAll(f)
	if err != nil {
		errorJSON(c, http.StatusInternalServerError, "cannot read uploaded file")
		return
	}

	text := strings.TrimSpace(string(raw))
	if text == "" {
		errorJSON(c, http.StatusBadRequest, "file is empty")
		return
	}

	sourceFormat := paragraph.TextFormatPlain
	if rag.DetectContentFormat(rag.ContentFormatAuto, file.Filename, file.Header.Get("Content-Type")) == rag.ContentFormatMarkdown {
		sourceFormat = paragraph.TextFormatMarkdown
	}

	formatted, err := paragraph.FormatTextWithFormat(c.Request.Context(), text, sourceFormat, h.cfg.LLM)
	if err != nil {
		errorJSON(c, http.StatusBadGateway, err.Error())
		return
	}

	c.Header("Content-Disposition", "attachment; filename=\"formatted.txt\"")
	c.Data(http.StatusOK, "text/plain; charset=utf-8", []byte(formatted))
}

// UploadText handles POST /upload — accepts a text file, chunks it, and indexes into
// the vector database for later retrieval.
func (h *ExtractHandler) UploadText(c *gin.Context) {
	// Check that RAG indexing is available
	if !h.ragIndexingAvailable() {
		errorJSONWithFields(c, http.StatusServiceUnavailable, "RAG indexing is not available", gin.H{"capability": h.ragIndexingCapability()})
		return
	}

	scope, err := strictRAGScopeFromRequest(c)
	if err != nil {
		errorJSON(c, http.StatusBadRequest, err.Error())
		return
	}

	file, err := c.FormFile("file")
	if err != nil {
		errorJSON(c, http.StatusBadRequest, "file is required")
		return
	}

	// Check file size limit
	if h.cfg.Upload.MaxFileBytes > 0 && file.Size > h.cfg.Upload.MaxFileBytes {
		errorJSON(c, http.StatusRequestEntityTooLarge, fmt.Sprintf("file too large: %d bytes, max %d bytes", file.Size, h.cfg.Upload.MaxFileBytes))
		return
	}

	f, err := file.Open()
	if err != nil {
		errorJSON(c, http.StatusInternalServerError, "cannot open uploaded file")
		return
	}
	defer f.Close()

	raw, err := io.ReadAll(f)
	if err != nil {
		errorJSON(c, http.StatusInternalServerError, "cannot read uploaded file")
		return
	}

	text := strings.TrimSpace(string(raw))
	if text == "" {
		errorJSON(c, http.StatusBadRequest, "file is empty")
		return
	}

	result, err := h.sources.IndexSourceScoped(c.Request.Context(), rag.Source{
		Text:        text,
		Filename:    file.Filename,
		ContentType: file.Header.Get("Content-Type"),
	}, h.indexOptions(), scope.UserID, scope.SessionID)
	if err != nil {
		slog.Error("upload.index_failed", "filename", file.Filename, "err", err)
		errorJSON(c, http.StatusBadGateway, fmt.Sprintf("indexing failed: %v", err))
		return
	}

	slog.Info("upload.done", "filename", file.Filename, "content_type", result.ContentType, "bytes", len(text), "chunks", result.ChunkCount)
	c.JSON(http.StatusOK, gin.H{
		"status":       "ok",
		"filename":     file.Filename,
		"content_type": result.ContentType,
		"size_bytes":   len(text),
		"chunk_count":  result.ChunkCount,
		"source_ids":   result.SourceIDs,
	})
}

func (h *ExtractHandler) DeleteRAGSource(c *gin.Context) {
	if !h.ragIndexingAvailable() {
		errorJSONWithFields(c, http.StatusServiceUnavailable, "RAG indexing is not available", gin.H{"capability": h.ragIndexingCapability()})
		return
	}

	sourceID := strings.TrimSpace(c.Param("source_id"))
	if sourceID == "" {
		errorJSON(c, http.StatusBadRequest, "source_id is required")
		return
	}

	filter, err := deleteFilterFromRequest(c)
	if err != nil {
		errorJSON(c, http.StatusBadRequest, err.Error())
		return
	}

	deleteReq := rag.DeleteRequest{
		SourceIDs: []string{sourceID},
		Filter:    filter,
	}
	result, err := h.sources.DeleteSourcesWithOptions(c.Request.Context(), deleteReq)
	if err != nil {
		slog.Error("rag.delete_source_failed", "source_id", sourceID, "err", err)
		errorJSON(c, http.StatusBadGateway, fmt.Sprintf("delete source failed: %v", err))
		return
	}

	slog.Info("rag.delete_source_done", "source_id", sourceID)
	c.JSON(http.StatusOK, gin.H{
		"status":     "ok",
		"source_ids": result.SourceIDs,
	})
}

func (h *ExtractHandler) ListRAGSources(c *gin.Context) {
	if !h.ragCatalogAvailable() {
		errorJSONWithFields(c, http.StatusServiceUnavailable, "RAG source catalog is not available", gin.H{"capability": h.ragCatalogCapability()})
		return
	}

	filter, err := strictRAGFilterFromRequest(c)
	if err != nil {
		errorJSON(c, http.StatusBadRequest, err.Error())
		return
	}
	limit, err := sourceListLimitFromRequest(c)
	if err != nil {
		errorJSON(c, http.StatusBadRequest, err.Error())
		return
	}

	sources, err := h.sources.ListSources(c.Request.Context(), rag.SourceListRequest{
		Filter: filter,
		Limit:  limit,
	})
	if err != nil {
		slog.Error("rag.list_sources_failed", "err", err)
		errorJSON(c, http.StatusBadGateway, fmt.Sprintf("list sources failed: %v", err))
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"sources": sources,
	})
}

func deleteFilterFromRequest(c *gin.Context) (*rag.RetrieveFilter, error) {
	return strictRAGFilterFromRequest(c)
}

func strictRAGFilterFromRequest(c *gin.Context) (*rag.RetrieveFilter, error) {
	scope, err := strictRAGScopeFromRequest(c)
	if err != nil {
		return nil, err
	}
	return scope.RetrieveFilter(), nil
}

func strictRAGScopeFromRequest(c *gin.Context) (rag.Scope, error) {
	return rag.ResolveScope(ragScopeValueFromRequest(c, "user_id", "X-User-ID"), ragScopeValueFromRequest(c, "session_id", "X-Session-ID"), rag.PersonalKnowledgeScopePolicy())
}

func ragScopeValueFromRequest(c *gin.Context, field, header string) string {
	for _, value := range []string{
		c.GetHeader(header),
		c.Query(field),
		c.PostForm(field),
	} {
		if value = strings.TrimSpace(value); value != "" {
			return value
		}
	}
	return ""
}

func sourceListLimitFromRequest(c *gin.Context) (int, error) {
	raw := strings.TrimSpace(c.Query("limit"))
	if raw == "" {
		return 0, nil
	}
	limit, err := strconv.Atoi(raw)
	if err != nil || limit <= 0 {
		return 0, fmt.Errorf("limit must be a positive integer")
	}
	return limit, nil
}

func (h *ExtractHandler) ragIndexingAvailable() bool {
	return h.sources != nil && h.sources.CanIndex() && h.caps.Usable(capability.RAG)
}

func (h *ExtractHandler) ragIndexingCapability() capability.Capability {
	c := h.caps.Get(capability.RAG)
	if (h.sources == nil || !h.sources.CanIndex()) && c.Status != capability.Unavailable {
		return capability.Capability{
			Name:   capability.RAG,
			Status: capability.Unavailable,
			Reason: "rag indexer unavailable",
		}
	}
	return c
}

func (h *ExtractHandler) ragCatalogAvailable() bool {
	return h.sources != nil && h.sources.CanList() && h.caps.Usable(capability.RAG)
}

func (h *ExtractHandler) ragCatalogCapability() capability.Capability {
	c := h.caps.Get(capability.RAG)
	if (h.sources == nil || !h.sources.CanList()) && c.Status != capability.Unavailable {
		return capability.Capability{
			Name:   capability.RAG,
			Status: capability.Unavailable,
			Reason: "rag source catalog unavailable",
		}
	}
	return c
}

func (h *ExtractHandler) indexOptions() rag.IndexOptions {
	overlapRunes := h.cfg.Upload.OverlapRunes
	return rag.IndexOptions{
		ChunkRunes:   h.cfg.Upload.ChunkRunes,
		OverlapRunes: &overlapRunes,
	}
}

// extractRequest for legacy endpoint.
type extractRequest struct {
	URL  string `form:"url" json:"url" binding:"required"`
	Name string `form:"name" json:"name" binding:"required"`
	Type string `form:"type" json:"type" binding:"required"`
}

func readTextFile(path string) (string, error) {
	if path == "" {
		return "", fmt.Errorf("empty path")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(data)), nil
}

func bindExtractRequest(c *gin.Context) (extractRequest, error) {
	var req extractRequest
	var err error
	if c.Request.Method == http.MethodPost {
		err = c.ShouldBindJSON(&req)
	} else {
		err = c.ShouldBindQuery(&req)
	}
	if err != nil {
		return req, errors.New("url, name and type are required")
	}

	req.URL = strings.TrimSpace(req.URL)
	req.Name = sanitizeName(req.Name)
	req.Type = strings.ToLower(strings.TrimSpace(req.Type))

	if req.URL == "" {
		return req, errors.New("url is required")
	}
	if req.Name == "" {
		return req, errors.New("name is required and must contain letters, numbers, dot, underscore or dash")
	}
	return req, nil
}

var safeNameRE = regexp.MustCompile(`[^\p{L}\p{N}._-]+`)

func sanitizeName(name string) string {
	name = strings.TrimSpace(name)
	name = safeNameRE.ReplaceAllString(name, "_")
	name = strings.Trim(name, "._-")
	return name
}

func statusForExtractError(err error) int {
	if err != nil && strings.Contains(err.Error(), "type must be one of") {
		return http.StatusBadRequest
	}
	return http.StatusBadGateway
}
