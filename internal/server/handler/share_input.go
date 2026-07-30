package handler

import "github.com/raiki02/vidwise/internal/videoinput"

type normalizedVideoShareInput = videoinput.NormalizedShareInput

func normalizeVideoShareInput(rawInput, rawName string) normalizedVideoShareInput {
	return videoinput.NormalizeShareInput(rawInput, rawName)
}
