package tool

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/raiki02/vidwise/internal/asr"
)

// Transcriber is the ASR service dependency of transcribe_audio.
type Transcriber interface {
	Transcribe(ctx context.Context, audioPath, language string) (asr.TranscribeResponse, error)
}

type TranscribeInput struct {
	AudioPath string `json:"audio_path" jsonschema:"required" jsonschema_description:"Absolute path of the audio file to transcribe, as returned by download_audio."`
	Language  string `json:"language,omitempty" jsonschema_description:"BCP-47 language code; defaults to the configured ASR language."`
}

type TranscribeOutput struct {
	TranscriptPath string  `json:"transcript_path" jsonschema_description:"Absolute path of the file containing the full transcript text."`
	Duration       float64 `json:"duration" jsonschema_description:"Audio duration in seconds."`
	Chars          int     `json:"chars" jsonschema_description:"Number of characters in the full transcript."`
	Preview        string  `json:"preview" jsonschema_description:"First 500 characters of the transcript for quick reference."`
}

// NewTranscribeTool returns the transcribe_audio eino tool. The full transcript
// is written next to the audio file (audio path with .txt extension) so the
// agent never has to pass large text through the tool argument.
func NewTranscribeTool(workDir string, client Transcriber) (tool.InvokableTool, error) {
	inner, err := utils.InferTool(
		"transcribe_audio",
		"Transcribe a local audio file to text via the ASR service. Writes the full transcript to a .txt file next to the audio file and returns its path plus a short preview.",
		func(ctx context.Context, input TranscribeInput) (TranscribeOutput, error) {
			if err := PathInWorkDir(workDir, input.AudioPath); err != nil {
				return TranscribeOutput{}, err
			}
			start := time.Now()
			resp, err := client.Transcribe(ctx, filepath.Clean(input.AudioPath), input.Language)
			if err != nil {
				return TranscribeOutput{}, fmt.Errorf("transcribe: %w", err)
			}
			text := strings.TrimSpace(resp.Text)
			if text == "" {
				return TranscribeOutput{}, fmt.Errorf("transcript is empty")
			}
			transcriptPath := strings.TrimSuffix(filepath.Clean(input.AudioPath), filepath.Ext(input.AudioPath)) + ".txt"
			if err := os.WriteFile(transcriptPath, []byte(text), 0o644); err != nil {
				return TranscribeOutput{}, fmt.Errorf("write transcript: %w", err)
			}
			slog.Info("tool.transcribe_audio.done", "transcript_path", transcriptPath, "chars", utf8.RuneCountInString(text), "elapsed", time.Since(start))
			return TranscribeOutput{
				TranscriptPath: transcriptPath,
				Duration:       resp.Duration,
				Chars:          utf8.RuneCountInString(text),
				Preview:        previewOf(text, 500),
			}, nil
		},
	)
	if err != nil {
		return nil, err
	}
	return inner, nil
}

func previewOf(text string, runes int) string {
	r := []rune(text)
	if len(r) <= runes {
		return text
	}
	return string(r[:runes])
}
