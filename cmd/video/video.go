package video

import (
	"os/exec"
	"path/filepath"
)

func Compatible(inputPath, outputPath string) ([]byte, error) {
	return exec.Command(
		"ffmpeg",
		"-y",
		"-i", filepath.Clean(inputPath),
		"-map", "0:v:0",
		"-map", "0:a?",
		"-c:v", "libx264",
		"-preset", "medium",
		"-crf", "23",
		"-pix_fmt", "yuv420p",
		"-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
		"-tag:v", "avc1",
		"-c:a", "aac",
		"-b:a", "128k",
		"-movflags", "+faststart",
		filepath.Clean(outputPath),
	).CombinedOutput()
}
