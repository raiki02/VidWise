package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/asr"
	"github.com/raiki02/vidwise/internal/paragraph"
	"github.com/raiki02/vidwise/internal/tool"
)

type TranscriptAgent struct {
	cfg     appconfig.Config
	asrTool *tool.Wrapper
}

func NewTranscriptAgent(cfg appconfig.Config, registry *tool.Registry) (*TranscriptAgent, error) {
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

	inner, asrWrapper, err := tool.NewASRTool(client)
	if err != nil {
		return nil, fmt.Errorf("create asr tool: %w", err)
	}

	if registry != nil {
		registry.Register("transcribe_audio", inner, nil)
	}
	return &TranscriptAgent{
		cfg:     cfg,
		asrTool: asrWrapper,
	}, nil
}

func (a *TranscriptAgent) Run(ctx context.Context, audioPath string) (string, error) {
	start := time.Now()

	// Run ASR via tool wrapper (with retry and circuit breaker)
	args, err := tool.ToJSON(asr.TranscribeInput{
		AudioPath: audioPath,
		Language:  a.cfg.ASR.Language,
	})
	if err != nil {
		return "", fmt.Errorf("encode asr input: %w", err)
	}

	outputJSON, err := a.asrTool.Run(ctx, args)
	if err != nil {
		return "", fmt.Errorf("transcribe audio: %w", err)
	}

	var output asr.TranscribeResponse
	if err := json.Unmarshal([]byte(outputJSON), &output); err != nil {
		return "", fmt.Errorf("decode asr output: %w", err)
	}
	slog.Info("transcript.stage", "stage", "asr", "elapsed", time.Since(start))

	// Format with LLM
	stage := time.Now()
	formattedText, err := paragraph.FormatTextWithFallback(ctx, output.Text, a.cfg.LLM)
	if err != nil {
		return "", fmt.Errorf("format transcript paragraphs failed: %w", err)
	}
	slog.Info("transcript.stage", "stage", "paragraph_format", "elapsed", time.Since(stage))
	slog.Info("transcript.done", "elapsed", time.Since(start))
	return formattedText, nil
}
