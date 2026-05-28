package transcript

import (
	"fmt"
	"os/exec"
	"path/filepath"
)

func Text(audioPath, modelPath, language, prompt string) (string, []byte, error) {
	if language == "" {
		language = "zh"
	}
	if prompt == "" {
		prompt = "以下是普通话的句子，请使用简体中文输出。"
	}

	outputBase := filepath.Clean(audioPath)
	outputPath := outputBase + ".txt"
	out, err := exec.Command(
		"whisper-cli",
		"-m", filepath.Clean(modelPath),
		"-f", filepath.Clean(audioPath),
		"--prompt", prompt,
		"-otxt",
		"-l", language,
	).CombinedOutput()
	return outputPath, out, err
}

func CommandError(message string, out []byte, err error) error {
	detail := string(out)
	if detail == "" {
		return fmt.Errorf("%s: %w", message, err)
	}
	return fmt.Errorf("%s: %s: %w", message, detail, err)
}
