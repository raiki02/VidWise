package handler

import (
	"net/url"
	"regexp"
	"strings"
	"unicode"
)

type normalizedVideoShareInput struct {
	URL   string
	Name  string
	Title string
}

var (
	shareURLRE          = regexp.MustCompile(`https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+`)
	shareTitleBracketRE = regexp.MustCompile(`【([^】]+)】`)
)

func normalizeVideoShareInput(rawInput, rawName string) normalizedVideoShareInput {
	videoURL, urlStart := extractVideoURL(rawInput)
	title := extractShareTitle(rawInput, urlStart, videoURL)

	name := sanitizeName(rawName)
	if name == "" {
		name = sanitizeName(title)
	}

	return normalizedVideoShareInput{
		URL:   videoURL,
		Name:  name,
		Title: title,
	}
}

func extractVideoURL(input string) (string, int) {
	match := shareURLRE.FindStringIndex(input)
	if match == nil {
		return strings.TrimSpace(input), -1
	}
	rawURL := input[match[0]:match[1]]
	return trimShareURL(rawURL), match[0]
}

func trimShareURL(videoURL string) string {
	videoURL = strings.TrimSpace(videoURL)
	return strings.TrimRight(videoURL, "，,。；;！!？?)]}】》>\"'`")
}

func extractShareTitle(input string, urlStart int, videoURL string) string {
	if urlStart < 0 {
		return ""
	}
	prefix := strings.TrimSpace(input[:urlStart])
	if prefix == "" {
		return ""
	}
	if title := extractBracketedShareTitle(prefix); title != "" {
		return cleanShareTitle(title)
	}
	if isDouyinURL(videoURL) {
		return extractDouyinShareTitle(prefix)
	}
	return cleanShareTitle(prefix)
}

func extractBracketedShareTitle(prefix string) string {
	matches := shareTitleBracketRE.FindAllStringSubmatch(prefix, -1)
	if len(matches) == 0 {
		return ""
	}
	title := matches[len(matches)-1][1]
	for _, marker := range []string{" | 小红书", "｜小红书"} {
		if idx := strings.Index(title, marker); idx >= 0 {
			title = title[:idx]
		}
	}
	return title
}

func extractDouyinShareTitle(prefix string) string {
	title := prefix
	if idx := strings.Index(title, "#"); idx >= 0 {
		title = title[:idx]
	}
	title = strings.TrimSpace(title)
	title = trimLeadingDouyinMetadata(title)
	return cleanShareTitle(title)
}

func trimLeadingDouyinMetadata(title string) string {
	runes := []rune(title)
	for i, r := range runes {
		if unicode.Is(unicode.Han, r) {
			start := i
			for start > 0 && !unicode.IsSpace(runes[start-1]) {
				start--
			}
			return strings.TrimSpace(string(runes[start:]))
		}
	}
	return title
}

func cleanShareTitle(title string) string {
	title = strings.TrimSpace(title)
	for _, marker := range []string{"复制此链接", "打开Dou音搜索", "打开抖音搜索"} {
		if idx := strings.Index(title, marker); idx >= 0 {
			title = title[:idx]
		}
	}
	return strings.Join(strings.Fields(strings.TrimSpace(title)), " ")
}

func isDouyinURL(videoURL string) bool {
	host := shareInputHost(videoURL)
	return host == "douyin.com" || strings.HasSuffix(host, ".douyin.com")
}

func shareInputHost(videoURL string) string {
	parsed, err := url.Parse(strings.TrimSpace(videoURL))
	if err != nil {
		return ""
	}
	return strings.ToLower(parsed.Hostname())
}
