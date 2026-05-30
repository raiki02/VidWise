package transcript

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type whisperServerSegment struct {
	Text string `json:"text"`
}

type whisperServerResponse struct {
	Text     string                 `json:"text"`
	Segments []whisperServerSegment `json:"segments"`
}

// Text sends the audio file to a whisper-server and writes the returned transcript
// to <audioPath>.txt. It keeps the historical signature used by the agent fallback.
//
// serverBaseURL: for example "http://127.0.0.1:8080".
func Text(audioPath, serverBaseURL, language, prompt string) (string, []byte, error) {
	if language == "" {
		language = "zh"
	}
	if prompt == "" {
		prompt = "以下是普通话的句子，请使用简体中文输出。"
	}
	serverBaseURL = strings.TrimRight(strings.TrimSpace(serverBaseURL), "/")
	if serverBaseURL == "" {
		serverBaseURL = "http://127.0.0.1:8080"
	}

	outputBase := filepath.Clean(audioPath)
	outputPath := outputBase + ".txt"

	file, err := os.Open(filepath.Clean(audioPath))
	if err != nil {
		return outputPath, nil, fmt.Errorf("open audio file failed: %w", err)
	}
	defer file.Close()

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)

	part, err := writer.CreateFormFile("file", filepath.Base(audioPath))
	if err != nil {
		_ = writer.Close()
		return outputPath, nil, fmt.Errorf("create file form part failed: %w", err)
	}
	if _, err := io.Copy(part, file); err != nil {
		_ = writer.Close()
		return outputPath, nil, fmt.Errorf("write file form part failed: %w", err)
	}

	// Match the curl fields provided by the user.
	_ = writer.WriteField("temperature", "0.0")
	_ = writer.WriteField("temperature_inc", "0.2")
	_ = writer.WriteField("prompt", prompt)
	_ = writer.WriteField("carry_initial_prompt", "true")
	_ = writer.WriteField("response_format", "json")
	// Optional (harmless if the server ignores it).
	_ = writer.WriteField("language", language)

	if err := writer.Close(); err != nil {
		return outputPath, nil, fmt.Errorf("finalize multipart body failed: %w", err)
	}

	url := serverBaseURL + "/inference"
	req, err := http.NewRequest(http.MethodPost, url, &body)
	if err != nil {
		return outputPath, nil, fmt.Errorf("create whisper-server request failed: %w", err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: 30 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return outputPath, nil, fmt.Errorf("call whisper-server failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return outputPath, nil, fmt.Errorf("read whisper-server response failed: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return outputPath, respBody, fmt.Errorf("whisper-server returned %s", resp.Status)
	}

	var parsed whisperServerResponse
	text := ""
	if err := json.Unmarshal(respBody, &parsed); err == nil {
		text = strings.TrimSpace(parsed.Text)
		if text == "" && len(parsed.Segments) > 0 {
			var sb strings.Builder
			for _, seg := range parsed.Segments {
				segText := strings.TrimSpace(seg.Text)
				if segText == "" {
					continue
				}
				if sb.Len() > 0 {
					sb.WriteString("\n")
				}
				sb.WriteString(segText)
			}
			text = sb.String()
		}
	} else {
		// Try a more tolerant decode for different whisper-server variants.
		var anyResp map[string]any
		if err2 := json.Unmarshal(respBody, &anyResp); err2 != nil {
			// Return response body for troubleshooting.
			return outputPath, respBody, fmt.Errorf("decode whisper-server JSON failed: %w", err)
		}
		if v, ok := anyResp["text"].(string); ok {
			text = strings.TrimSpace(v)
		}
		if text == "" {
			if v, ok := anyResp["transcription"].(string); ok {
				text = strings.TrimSpace(v)
			}
		}
		if text == "" {
			if segs, ok := anyResp["segments"].([]any); ok {
				var sb strings.Builder
				for _, s := range segs {
					segMap, ok := s.(map[string]any)
					if !ok {
						continue
					}
					segText, _ := segMap["text"].(string)
					segText = strings.TrimSpace(segText)
					if segText == "" {
						continue
					}
					if sb.Len() > 0 {
						sb.WriteString("\n")
					}
					sb.WriteString(segText)
				}
				text = sb.String()
			}
		}
	}

	text = strings.TrimSpace(text)

	if err := os.WriteFile(outputPath, []byte(text), 0644); err != nil {
		return outputPath, respBody, fmt.Errorf("write transcript file failed: %w", err)
	}

	return outputPath, respBody, nil
}

func CommandError(message string, out []byte, err error) error {
	detail := string(out)
	if detail == "" {
		return fmt.Errorf("%s: %w", message, err)
	}
	return fmt.Errorf("%s: %s: %w", message, detail, err)
}
