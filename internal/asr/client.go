package asr

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type Client struct {
	baseURL  string
	language string
	http     *http.Client
}

type Segment struct {
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

type TranscribeResponse struct {
	Text                string    `json:"text"`
	Language            string    `json:"language"`
	LanguageProbability float64   `json:"language_probability"`
	Duration            float64   `json:"duration"`
	Segments            []Segment `json:"segments"`
}

func NewClient(baseURL, language string, timeout time.Duration) (*Client, error) {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if baseURL == "" {
		return nil, fmt.Errorf("asr base_url is required")
	}
	if _, err := url.ParseRequestURI(baseURL); err != nil {
		return nil, fmt.Errorf("invalid asr base_url: %w", err)
	}
	if language == "" {
		language = "zh"
	}
	if timeout <= 0 {
		timeout = 10 * time.Minute
	}
	return &Client{
		baseURL:  baseURL,
		language: language,
		http:     &http.Client{Timeout: timeout},
	}, nil
}

func (c *Client) Transcribe(ctx context.Context, audioPath, language string) (TranscribeResponse, error) {
	audioPath = filepath.Clean(audioPath)
	file, err := os.Open(audioPath)
	if err != nil {
		return TranscribeResponse{}, fmt.Errorf("open audio file failed: %w", err)
	}
	defer file.Close()

	if language == "" {
		language = c.language
	}

	bodyReader, bodyWriter := io.Pipe()
	writer := multipart.NewWriter(bodyWriter)
	go func() {
		err := writeMultipartBody(writer, file, audioPath, language)
		if err != nil {
			_ = bodyWriter.CloseWithError(err)
			return
		}
		_ = bodyWriter.Close()
	}()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/transcribe", bodyReader)
	if err != nil {
		return TranscribeResponse{}, fmt.Errorf("create asr request failed: %w", err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := c.http.Do(req)
	if err != nil {
		return TranscribeResponse{}, fmt.Errorf("call asr service failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return TranscribeResponse{}, fmt.Errorf("read asr response failed: %w", err)
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return TranscribeResponse{}, fmt.Errorf("asr service returned %s: %s", resp.Status, strings.TrimSpace(string(respBody)))
	}

	var output TranscribeResponse
	if err := json.Unmarshal(respBody, &output); err != nil {
		return TranscribeResponse{}, fmt.Errorf("decode asr response failed: %w", err)
	}
	return output, nil
}

func writeMultipartBody(writer *multipart.Writer, file *os.File, audioPath, language string) error {
	defer writer.Close()

	if err := writeFilePart(writer, "file", filepath.Base(audioPath), file); err != nil {
		return err
	}
	if err := writer.WriteField("language", language); err != nil {
		return fmt.Errorf("write language field failed: %w", err)
	}
	return nil
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
