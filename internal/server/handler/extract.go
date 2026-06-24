package handler

import (
	"errors"
	"io"
	"net/http"
	"regexp"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/raiki02/video-extractor/internal/appconfig"
	"github.com/raiki02/video-extractor/internal/extractor"
	"github.com/raiki02/video-extractor/internal/paragraph"
)

// ExtractHandler retains backward compatibility with the legacy /extract and /format endpoints.
type ExtractHandler struct {
	cfg       appconfig.Config
	extractor *extractor.Service
}

func NewExtractHandler(cfg appconfig.Config) *ExtractHandler {
	return &ExtractHandler{
		cfg:       cfg,
		extractor: extractor.NewService(cfg),
	}
}

// Extract handles GET/POST /extract (legacy, synchronous).
func (h *ExtractHandler) Extract(c *gin.Context) {
	req, err := bindExtractRequest(c)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	result, cleanup, err := h.extractor.Extract(c.Request.Context(), req.URL, req.Name, req.Type)
	if cleanup != nil {
		defer cleanup()
	}
	if err != nil {
		c.JSON(statusForExtractError(err), gin.H{"error": err.Error()})
		return
	}

	c.FileAttachment(result.Path, result.Filename)
}

// FormatText handles POST /format (legacy, synchronous text formatting).
func (h *ExtractHandler) FormatText(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "file is required"})
		return
	}

	f, err := file.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "cannot open uploaded file"})
		return
	}
	defer f.Close()

	raw, err := io.ReadAll(f)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "cannot read uploaded file"})
		return
	}

	text := strings.TrimSpace(string(raw))
	if text == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "file is empty"})
		return
	}

	formatted, err := paragraph.FormatText(c.Request.Context(), text, h.cfg.LLM)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}

	c.Header("Content-Disposition", "attachment; filename=\"formatted.txt\"")
	c.Data(http.StatusOK, "text/plain; charset=utf-8", []byte(formatted))
}

// extractRequest for legacy endpoint.
type extractRequest struct {
	URL  string `form:"url" json:"url" binding:"required"`
	Name string `form:"name" json:"name" binding:"required"`
	Type string `form:"type" json:"type" binding:"required"`
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
