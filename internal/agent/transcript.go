package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/cloudwego/eino/components/tool"
	"github.com/raiki02/video-extractor/cmd/transcript"
	"github.com/raiki02/video-extractor/internal/appconfig"
	"github.com/raiki02/video-extractor/internal/asr"
	"github.com/raiki02/video-extractor/internal/paragraph"
)

type TranscriptAgent struct {
	cfg     appconfig.Config
	asrTool tool.InvokableTool
}

func NewTranscriptAgent(cfg appconfig.Config) (*TranscriptAgent, error) {
	timeout, err := cfg.ASR.TimeoutDuration()
	if err != nil {
		return nil, fmt.Errorf("invalid asr.timeout: %w", err)
	}

	client, err := asr.NewClient(cfg.ASR.BaseURL, cfg.ASR.Language, timeout, asr.TranscribeOptions{
		BeamSize:      cfg.ASR.Transcribe.BeamSize,
		VADFilter:     cfg.ASR.Transcribe.VADFilter,
		InitialPrompt: cfg.ASR.Transcribe.InitialPrompt,
	})
	if err != nil {
		return nil, err
	}

	asrTool, err := asr.NewTranscribeTool(client)
	if err != nil {
		return nil, fmt.Errorf("create asr tool failed: %w", err)
	}

	return &TranscriptAgent{
		cfg:     cfg,
		asrTool: asrTool,
	}, nil
}

func (a *TranscriptAgent) Run(ctx context.Context, audioPath string) (string, error) {
	rawText, err := a.transcribe(ctx, audioPath)
	if err != nil {
		return "", err
	}

	formattedText, err := paragraph.FormatText(ctx, rawText, a.cfg.LLM)
	if err != nil {
		return "", fmt.Errorf("format transcript paragraphs failed: %w", err)
	}
	return formattedText, nil
}

func (a *TranscriptAgent) transcribe(ctx context.Context, audioPath string) (string, error) {
	args, err := json.Marshal(asr.TranscribeInput{
		AudioPath: audioPath,
		Language:  a.cfg.ASR.Language,
	})
	if err != nil {
		return "", fmt.Errorf("encode asr tool input failed: %w", err)
	}

	outputJSON, err := a.asrTool.InvokableRun(ctx, string(args))
	if err != nil {
		return a.transcribeWithWhisperCLI(audioPath, err)
	}

	var output asr.TranscribeResponse
	if err := json.Unmarshal([]byte(outputJSON), &output); err != nil {
		return "", fmt.Errorf("decode asr tool output failed: %w", err)
	}
	return output.Text, nil
}

func (a *TranscriptAgent) transcribeWithWhisperCLI(audioPath string, primaryErr error) (string, error) {
	rawTextPath, out, err := transcript.Text(audioPath, a.cfg.Whisper.ModelPath, a.cfg.Whisper.Language, a.cfg.Whisper.Prompt)
	if err != nil {
		return "", fmt.Errorf(
			"transcribe audio failed: python asr error: %w; whisper-cli fallback error: %w",
			primaryErr,
			transcript.CommandError("whisper-cli transcribe failed", out, err),
		)
	}

	rawText, err := os.ReadFile(rawTextPath)
	if err != nil {
		return "", fmt.Errorf("read whisper-cli transcript failed: %w", err)
	}
	return string(rawText), nil
}
