package extractor

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

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
	workDir, err := os.MkdirTemp("", "video-extractor-*")
	if err != nil {
		return Result{}, nil, fmt.Errorf("create temp directory failed: %w", err)
	}
	cleanup := func() {
		_ = os.RemoveAll(workDir)
	}

	videoPath := filepath.Join(workDir, name+".mp4")
	if out, err := download.Video(url, videoPath); err != nil {
		cleanup()
		return Result{}, nil, commandError("download video failed", out, err)
	}

	switch extractType {
	case "video":
		compatiblePath, err := s.createCompatibleVideo(videoPath, workDir, name)
		if err != nil {
			cleanup()
			return Result{}, nil, err
		}
		return Result{Path: compatiblePath, Filename: name + ".mp4"}, cleanup, nil
	case "audio":
		audioPath, err := s.createAudio(videoPath, workDir, name)
		if err != nil {
			cleanup()
			return Result{}, nil, err
		}
		return Result{Path: audioPath, Filename: name + ".mp3"}, cleanup, nil
	case "text", "transcript":
		textPath, err := s.createTranscript(ctx, videoPath, workDir, name)
		if err != nil {
			cleanup()
			return Result{}, nil, err
		}
		return Result{Path: textPath, Filename: name + ".txt"}, cleanup, nil
	default:
		cleanup()
		return Result{}, nil, fmt.Errorf("type must be one of: video, audio, text")
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

func (s *Service) createTranscript(ctx context.Context, videoPath, workDir, name string) (string, error) {
	audioPath, err := s.createAudio(videoPath, workDir, name)
	if err != nil {
		return "", err
	}

	transcriptAgent, err := agent.NewTranscriptAgent(s.cfg)
	if err != nil {
		return "", err
	}

	transcriptText, err := transcriptAgent.Run(ctx, audioPath)
	if err != nil {
		return "", err
	}

	formattedTextPath := filepath.Join(workDir, fmt.Sprintf("%s_formatted.txt", name))
	if err := os.WriteFile(formattedTextPath, []byte(transcriptText), 0644); err != nil {
		return "", fmt.Errorf("write formatted transcript failed: %w", err)
	}
	return formattedTextPath, nil
}

func commandError(message string, out []byte, err error) error {
	detail := string(out)
	if detail == "" {
		return fmt.Errorf("%s: %w", message, err)
	}
	return fmt.Errorf("%s: %s: %w", message, detail, err)
}
