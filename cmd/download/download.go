package download

import (
	"os/exec"
	"path/filepath"
	"strings"
)

// Video downloads the best available video+audio stream and merges it to mp4.
//
// cookiesPath is optional; when set, yt-dlp will use that cookies.txt file.
func Video(url, outputPath, cookiesPath string) ([]byte, error) {
	args := []string{
		"yt-dlp",
		"--no-playlist",
		"-f", "bv*+ba/b",
		"--add-header", "Origin:https://www.bilibili.com",
		"--add-header", "Referer:https://www.bilibili.com",
		"--merge-output-format", "mp4",
		"-o", filepath.Clean(outputPath),
	}
	cookiesPath = strings.TrimSpace(cookiesPath)
	if cookiesPath != "" {
		args = append(args, "--cookies", filepath.Clean(cookiesPath))
	}
	args = append(args, url)
	return exec.Command(args[0], args[1:]...).CombinedOutput()
}

// Audio downloads the best available audio stream and extracts it to an mp3 file.
//
// outputBase should be a full path WITHOUT extension; the final output will be
// outputBase + ".mp3".
//
// cookiesPath is optional; when set, yt-dlp will use that cookies.txt file.
func Audio(url, outputBase, cookiesPath string) (string, []byte, error) {
	outputPath, args := audioCommand(url, outputBase, cookiesPath)
	out, err := exec.Command(args[0], args[1:]...).CombinedOutput()
	return outputPath, out, err
}

func audioCommand(url, outputBase, cookiesPath string) (string, []string) {
	outputBase = filepath.Clean(outputBase)
	outputTemplate := outputBase + ".%(ext)s"
	outputPath := outputBase + ".mp3"
	args := []string{
		"yt-dlp",
		"--no-playlist",
		"-f", "bestaudio/b",
		"-x",
		"--audio-format", "mp3",
		// 0 is best, 10 is worst; 5 is a reasonable default for ASR.
		"--audio-quality", "5",
		"-o", outputTemplate,
	}
	cookiesPath = strings.TrimSpace(cookiesPath)
	if cookiesPath != "" {
		args = append(args, "--cookies", filepath.Clean(cookiesPath))
	}
	args = append(args, url)
	return outputPath, args
}
