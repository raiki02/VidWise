package download

import (
	"reflect"
	"testing"
)

func TestAudioCommandUsesCookiesPathWhenProvided(t *testing.T) {
	outputPath, args := audioCommand("https://example.com/video", "/tmp/demo", "/tmp/cookies.txt")

	if outputPath != "/tmp/demo.mp3" {
		t.Fatalf("output path = %q, want /tmp/demo.mp3", outputPath)
	}
	want := []string{
		"yt-dlp",
		"--no-playlist",
		"-f", "bestaudio/b",
		"-x",
		"--audio-format", "mp3",
		"--audio-quality", "5",
		"-o", "/tmp/demo.%(ext)s",
		"--cookies", "/tmp/cookies.txt",
		"https://example.com/video",
	}
	if !reflect.DeepEqual(args, want) {
		t.Fatalf("args = %#v, want %#v", args, want)
	}
}

func TestAudioCommandOmitsBrowserCookiesWhenCookiesPathMissing(t *testing.T) {
	_, args := audioCommand("https://example.com/video", "/tmp/demo", "")

	for i, arg := range args {
		if arg == "--cookies" || arg == "--cookies-from-browser" {
			t.Fatalf("args[%d] = %q, want no cookie option in %#v", i, arg, args)
		}
	}
}
