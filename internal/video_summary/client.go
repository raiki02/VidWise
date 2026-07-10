package video_summary

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/raiki02/vidwise/internal/healthcheck"
)

type Client struct {
	baseURL   string
	summarize SummarizeOptions
	http      *http.Client
}

type Event struct {
	Start       float64 `json:"start"`
	End         float64 `json:"end"`
	Description string  `json:"description"`
}

type CaptionResponse struct {
	Caption string  `json:"caption"`
	Scene   string  `json:"scene"`
	Events  []Event `json:"events"`
}

type FindResponse struct {
	Raw      string    `json:"raw"`
	Span     []float64 `json:"span"`
	FormatOK bool      `json:"format_ok"`
}

type SummarizeOptions struct {
	MaxNewTokens int
	Prompt       string
	DoSample     *bool
	Temperature  float32
	TopP         float32
}

func NewClient(baseURL string, timeout time.Duration, options SummarizeOptions) (*Client, error) {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if baseURL == "" {
		return nil, fmt.Errorf("video_summary base_url is required")
	}
	if _, err := url.ParseRequestURI(baseURL); err != nil {
		return nil, fmt.Errorf("invalid video_summary base_url: %w", err)
	}
	if timeout <= 0 {
		timeout = 20 * time.Minute
	}
	if options.MaxNewTokens == 0 {
		options.MaxNewTokens = 2048
	}
	if options.Temperature == 0 {
		options.Temperature = 1.0
	}
	if options.TopP == 0 {
		options.TopP = 1.0
	}
	return &Client{
		baseURL:   baseURL,
		summarize: options,
		http:      &http.Client{Timeout: timeout},
	}, nil
}

func (c *Client) Health(ctx context.Context) error {
	return healthcheck.CheckHTTP(ctx, c.http, c.baseURL, "video_summary service")
}

func (c *Client) Caption(ctx context.Context, videoPath string, options SummarizeOptions) (CaptionResponse, error) {
	var output CaptionResponse
	err := c.callMultipart(ctx, "/caption", videoPath, func(writer *multipart.Writer, file *os.File) error {
		if err := writeFilePart(writer, "file", filepath.Base(videoPath), file); err != nil {
			return err
		}
		return c.writeSummarizeFields(writer, options, false)
	}, &output)
	return output, err
}

func (c *Client) Find(ctx context.Context, videoPath, event string, options SummarizeOptions) (FindResponse, error) {
	if options.MaxNewTokens == 0 {
		options.MaxNewTokens = 64
	}
	var output FindResponse
	err := c.callMultipart(ctx, "/find", videoPath, func(writer *multipart.Writer, file *os.File) error {
		if err := writeFilePart(writer, "file", filepath.Base(videoPath), file); err != nil {
			return err
		}
		if err := writer.WriteField("event", event); err != nil {
			return fmt.Errorf("write event field failed: %w", err)
		}
		return c.writeSummarizeFields(writer, options, true)
	}, &output)
	return output, err
}

func (c *Client) callMultipart(ctx context.Context, path, videoPath string, writeBody func(*multipart.Writer, *os.File) error, output any) error {
	start := time.Now()
	videoPath = filepath.Clean(videoPath)
	file, err := os.Open(videoPath)
	if err != nil {
		return fmt.Errorf("open video file failed: %w", err)
	}
	defer file.Close()
	if fi, statErr := file.Stat(); statErr == nil {
		slog.Info("video_summary.request", "video_path", videoPath, "size_bytes", fi.Size())
	} else {
		slog.Info("video_summary.request", "video_path", videoPath)
	}

	bodyReader, bodyWriter := io.Pipe()
	writer := multipart.NewWriter(bodyWriter)
	go func() {
		err := writeBody(writer, file)
		if closeErr := writer.Close(); err == nil {
			err = closeErr
		}
		if err != nil {
			_ = bodyWriter.CloseWithError(err)
			return
		}
		_ = bodyWriter.Close()
	}()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bodyReader)
	if err != nil {
		return fmt.Errorf("create video_summary request failed: %w", err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	doStart := time.Now()
	resp, err := c.http.Do(req)
	if err != nil {
		slog.Info("video_summary.http_error", "elapsed", time.Since(doStart), "total_elapsed", time.Since(start), "err", err)
		return fmt.Errorf("call video_summary service failed: %w", err)
	}
	defer resp.Body.Close()
	slog.Info("video_summary.http_done", "status", resp.StatusCode, "elapsed", time.Since(doStart))

	readStart := time.Now()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		slog.Info("video_summary.read_error", "elapsed", time.Since(readStart), "total_elapsed", time.Since(start), "err", err)
		return fmt.Errorf("read video_summary response failed: %w", err)
	}
	slog.Info("video_summary.read_done", "bytes", len(respBody), "elapsed", time.Since(readStart))
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		slog.Info("video_summary.response_error", "status", resp.Status, "total_elapsed", time.Since(start))
		return fmt.Errorf("video_summary service returned %s: %s", resp.Status, strings.TrimSpace(string(respBody)))
	}

	decodeStart := time.Now()
	if err := json.Unmarshal(respBody, output); err != nil {
		return fmt.Errorf("decode video_summary response failed: %w", err)
	}
	slog.Info("video_summary.decode_done", "elapsed", time.Since(decodeStart), "total_elapsed", time.Since(start))
	return nil
}

func (c *Client) writeSummarizeFields(writer *multipart.Writer, options SummarizeOptions, skipPrompt bool) error {
	options = c.mergeOptions(options)
	if err := writer.WriteField("max_new_tokens", fmt.Sprintf("%d", options.MaxNewTokens)); err != nil {
		return fmt.Errorf("write max_new_tokens field failed: %w", err)
	}
	if !skipPrompt && options.Prompt != "" {
		if err := writer.WriteField("prompt", options.Prompt); err != nil {
			return fmt.Errorf("write prompt field failed: %w", err)
		}
	}
	if options.DoSample != nil {
		if err := writer.WriteField("do_sample", fmt.Sprintf("%t", *options.DoSample)); err != nil {
			return fmt.Errorf("write do_sample field failed: %w", err)
		}
	}
	if err := writer.WriteField("temperature", fmt.Sprintf("%g", options.Temperature)); err != nil {
		return fmt.Errorf("write temperature field failed: %w", err)
	}
	if err := writer.WriteField("top_p", fmt.Sprintf("%g", options.TopP)); err != nil {
		return fmt.Errorf("write top_p field failed: %w", err)
	}
	return nil
}

func (c *Client) mergeOptions(options SummarizeOptions) SummarizeOptions {
	if options.MaxNewTokens == 0 {
		options.MaxNewTokens = c.summarize.MaxNewTokens
	}
	if options.Prompt == "" {
		options.Prompt = c.summarize.Prompt
	}
	if options.DoSample == nil {
		options.DoSample = c.summarize.DoSample
	}
	if options.Temperature == 0 {
		options.Temperature = c.summarize.Temperature
	}
	if options.TopP == 0 {
		options.TopP = c.summarize.TopP
	}
	return options
}

func writeFilePart(writer *multipart.Writer, fieldName, fileName string, file *os.File) error {
	header := make(textproto.MIMEHeader)
	header.Set("Content-Disposition", fmt.Sprintf(`form-data; name="%s"; filename="%s"`, fieldName, escapeQuotes(fileName)))
	header.Set("Content-Type", "application/octet-stream")

	part, err := writer.CreatePart(header)
	if err != nil {
		return fmt.Errorf("create file part failed: %w", err)
	}
	if _, err := io.Copy(part, file); err != nil {
		return fmt.Errorf("write file part failed: %w", err)
	}
	return nil
}

func escapeQuotes(s string) string {
	return strings.NewReplacer("\\", "\\\\", `"`, "\\\"").Replace(s)
}
