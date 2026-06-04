package audio

import (
	"os/exec"
	"path/filepath"
)

func FromVideo(videoPath, outputPath string) ([]byte, error) {
	return exec.Command(
		"ffmpeg",
		"-y",
		"-i", filepath.Clean(videoPath),
		"-vn",
		"-acodec", "libmp3lame",
		filepath.Clean(outputPath),
	).CombinedOutput()
}
