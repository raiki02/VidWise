package server

import (
	"embed"
	"errors"
	"io"
	"io/fs"
	"net/http"
	"regexp"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/raiki02/video-extractor/internal/appconfig"
	"github.com/raiki02/video-extractor/internal/extractor"
	"github.com/raiki02/video-extractor/internal/paragraph"
)

type Server struct {
	cfg       appconfig.Config
	extractor *extractor.Service
}

type extractRequest struct {
	URL  string `form:"url" json:"url" binding:"required"`
	Name string `form:"name" json:"name" binding:"required"`
	Type string `form:"type" json:"type" binding:"required"`
}

var safeNamePattern = regexp.MustCompile(`[^\p{L}\p{N}._-]+`)

//go:embed web/*
var webFS embed.FS

func New(cfg appconfig.Config) *gin.Engine {
	s := &Server{
		cfg:       cfg,
		extractor: extractor.NewService(cfg),
	}

	e := gin.Default()
	web, err := fs.Sub(webFS, "web")
	if err != nil {
		panic(err)
	}
	e.StaticFS("/static", http.FS(web))
	e.GET("/health", s.health)
	e.GET("/extract", s.extract)
	e.POST("/extract", s.extract)
	e.POST("/format", s.formatText)
	return e
}

func (s *Server) health(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

func (s *Server) extract(c *gin.Context) {
	req, err := bindExtractRequest(c)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	result, cleanup, err := s.extractor.Extract(c.Request.Context(), req.URL, req.Name, req.Type)
	if cleanup != nil {
		defer cleanup()
	}
	if err != nil {
		c.JSON(statusForExtractError(err), gin.H{"error": err.Error()})
		return
	}

	c.FileAttachment(result.Path, result.Filename)
}

func bindExtractRequest(c *gin.Context) (extractRequest, error) {
	var req extractRequest
	var err error
	if c.Request.Method == http.MethodPost {
		err = c.ShouldBind(&req)
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

func sanitizeName(name string) string {
	name = strings.TrimSpace(name)
	name = safeNamePattern.ReplaceAllString(name, "_")
	name = strings.Trim(name, "._-")
	return name
}

func statusForExtractError(err error) int {
	if strings.Contains(err.Error(), "type must be one of") {
		return http.StatusBadRequest
	}
	return http.StatusBadGateway
}

func (s *Server) formatText(c *gin.Context) {
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

	formatted, err := paragraph.FormatText(c.Request.Context(), text, s.cfg.LLM)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}

	c.Header("Content-Disposition", "attachment; filename=\"formatted.txt\"")
	c.Data(http.StatusOK, "text/plain; charset=utf-8", []byte(formatted))
}
