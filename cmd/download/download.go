package download

import (
	"fmt"
	"os/exec"
	"path/filepath"
)

func Execute(url string, name string) ([]byte, error) {
	return exec.Command("yt-dlp",
		"-o", fmt.Sprintf("%s.%%(ext)s", name),
		url).CombinedOutput()
}

func Video(url, outputPath string) ([]byte, error) {
	return exec.Command(
		"yt-dlp",
		"--no-playlist",
		"-f", "bv*+ba/b",
		"--merge-output-format", "mp4",
		"-o", filepath.Clean(outputPath),
		url,
	).CombinedOutput()
}

// Audio downloads the best available audio stream and extracts it to an mp3 file.
//
// outputBase should be a full path WITHOUT extension; the final output will be
// outputBase + ".mp3".
func Audio(url, outputBase string) (string, []byte, error) {
	outputBase = filepath.Clean(outputBase)
	outputTemplate := outputBase + ".%(ext)s"
	outputPath := outputBase + ".mp3"
	out, err := exec.Command(
		"yt-dlp",
		"--no-playlist",
		"-f", "bestaudio/b",
		"-x",
		"--audio-format", "mp3",
		// 0 is best, 10 is worst; 5 is a reasonable default for ASR.
		"--audio-quality", "5",
		"-o", outputTemplate,
		url,
	).CombinedOutput()
	return outputPath, out, err
}
