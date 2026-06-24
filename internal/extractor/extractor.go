package extractor

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/raiki02/video-extractor/cmd/audio"
	"github.com/raiki02/video-extractor/cmd/download"
	"github.com/raiki02/video-extractor/cmd/video"
	"github.com/raiki02/video-extractor/internal/agent"
	"github.com/raiki02/video-extractor/internal/appconfig"
)

type Service struct {
	cfg appconfig.Config
}

type Result struct {
	Path     string
	Filename string
}

func NewService(cfg appconfig.Config) *Service {
	return &Service{cfg: cfg}
}

func (s *Service) Extract(ctx context.Context, url, name, extractType string) (Result, func(), error) {
	start := time.Now()
	workDir, err := os.MkdirTemp("", "video-extractor-*")
	if err != nil {
		return Result{}, nil, fmt.Errorf("create temp directory failed: %w", err)
	}
	cleanup := func() {
		_ = os.RemoveAll(workDir)
	}

	switch extractType {
	case "video":
		videoPath := filepath.Join(workDir, name+".mp4")
		stage := time.Now()
		if out, err := download.Video(url, videoPath, s.cfg.Download.CookiesPath); err != nil {
			cleanup()
			return Result{}, nil, commandError("download video failed", out, err)
		}
		slog.Info("extract.stage", "type", extractType, "stage", "download_video", "elapsed", time.Since(stage))

		stage = time.Now()
		compatiblePath, err := s.createCompatibleVideo(videoPath, workDir, name)
		if err != nil {
			cleanup()
			return Result{}, nil, err
		}
		slog.Info("extract.stage", "type", extractType, "stage", "convert_video", "elapsed", time.Since(stage))
		slog.Info("extract.done", "type", extractType, "elapsed", time.Since(start))
		return Result{Path: compatiblePath, Filename: name + ".mp4"}, cleanup, nil
	case "audio":
		stage := time.Now()
		audioPath, out, err := download.Audio(url, filepath.Join(workDir, name))
		if err != nil {
			cleanup()
			return Result{}, nil, commandError("download audio failed", out, err)
		}
		slog.Info("extract.stage", "type", extractType, "stage", "download_audio", "elapsed", time.Since(stage))
		slog.Info("extract.done", "type", extractType, "elapsed", time.Since(start))
		return Result{Path: audioPath, Filename: name + ".mp3"}, cleanup, nil
	case "text", "transcript":
		stage := time.Now()
		audioPath, out, err := download.Audio(url, filepath.Join(workDir, name))
		if err != nil {
			cleanup()
			return Result{}, nil, commandError("download audio failed", out, err)
		}
		slog.Info("extract.stage", "type", extractType, "stage", "download_audio", "elapsed", time.Since(stage))

		stage = time.Now()
		textPath, err := s.createTranscript(ctx, audioPath, workDir, name)
		if err != nil {
			cleanup()
			return Result{}, nil, err
		}
		slog.Info("extract.stage", "type", extractType, "stage", "transcript", "elapsed", time.Since(stage))
		slog.Info("extract.done", "type", extractType, "elapsed", time.Since(start))
		return Result{Path: textPath, Filename: name + ".txt"}, cleanup, nil
	case "summary", "video_summary":
		videoPath := filepath.Join(workDir, name+".mp4")
		stage := time.Now()
		if out, err := download.Video(url, videoPath, s.cfg.Download.CookiesPath); err != nil {
			cleanup()
			return Result{}, nil, commandError("download video failed", out, err)
		}
		slog.Info("extract.stage", "type", extractType, "stage", "download_video", "elapsed", time.Since(stage))

		stage = time.Now()
		summaryPath, err := s.createVideoSummary(ctx, videoPath, workDir, name)
		if err != nil {
			cleanup()
			return Result{}, nil, err
		}
		slog.Info("extract.stage", "type", extractType, "stage", "video_summary", "elapsed", time.Since(stage))
		slog.Info("extract.done", "type", extractType, "elapsed", time.Since(start))
		return Result{Path: summaryPath, Filename: name + "_summary.txt"}, cleanup, nil
	default:
		cleanup()
		return Result{}, nil, fmt.Errorf("type must be one of: video, audio, text, summary")
	}
}

func (s *Service) createCompatibleVideo(videoPath, workDir, name string) (string, error) {
	compatiblePath := filepath.Join(workDir, fmt.Sprintf("%s_compatible.mp4", name))
	if out, err := video.Compatible(videoPath, compatiblePath); err != nil {
		return "", commandError("convert video for playback failed", out, err)
	}
	return compatiblePath, nil
}

func (s *Service) createAudio(videoPath, workDir, name string) (string, error) {
	audioPath := filepath.Join(workDir, fmt.Sprintf("%s.mp3", name))
	if out, err := audio.FromVideo(videoPath, audioPath); err != nil {
		return "", commandError("extract audio failed", out, err)
	}
	return audioPath, nil
}

func (s *Service) createTranscript(ctx context.Context, audioPath, workDir, name string) (string, error) {
	stage := time.Now()
	transcriptAgent, err := agent.NewTranscriptAgent(s.cfg, nil)
	if err != nil {
		return "", err
	}
	slog.Info("extract.stage", "type", "text", "stage", "init_transcript_agent", "elapsed", time.Since(stage))

	stage = time.Now()
	transcriptText, err := transcriptAgent.Run(ctx, audioPath)
	if err != nil {
		return "", err
	}
	slog.Info("extract.stage", "type", "text", "stage", "asr_and_format", "elapsed", time.Since(stage))

	formattedTextPath := filepath.Join(workDir, fmt.Sprintf("%s_formatted.txt", name))
	if err := os.WriteFile(formattedTextPath, []byte(transcriptText), 0644); err != nil {
		return "", fmt.Errorf("write formatted transcript failed: %w", err)
	}
	return formattedTextPath, nil
}

func (s *Service) createVideoSummary(ctx context.Context, videoPath, workDir, name string) (string, error) {
	stage := time.Now()
	summaryAgent, err := agent.NewVideoSummaryAgent(s.cfg)
	if err != nil {
		return "", err
	}
	slog.Info("extract.stage", "type", "summary", "stage", "init_video_summary_agent", "elapsed", time.Since(stage))

	stage = time.Now()
	summaryText, err := summaryAgent.Run(ctx, videoPath)
	if err != nil {
		return "", err
	}
	slog.Info("extract.stage", "type", "summary", "stage", "marlin_caption", "elapsed", time.Since(stage))

	summaryPath := filepath.Join(workDir, fmt.Sprintf("%s_summary.txt", name))
	if err := os.WriteFile(summaryPath, []byte(summaryText), 0644); err != nil {
		return "", fmt.Errorf("write video summary failed: %w", err)
	}
	return summaryPath, nil
}

func commandError(message string, out []byte, err error) error {
	detail := string(out)
	if detail == "" {
		return fmt.Errorf("%s: %w", message, err)
	}
	return fmt.Errorf("%s: %s: %w", message, detail, err)
}
