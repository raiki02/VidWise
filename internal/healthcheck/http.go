package healthcheck

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type response struct {
	Status string `json:"status"`
}

// CheckHTTP verifies a FastAPI-style dependency health endpoint.
func CheckHTTP(ctx context.Context, client *http.Client, baseURL, name string) error {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if baseURL == "" {
		return fmt.Errorf("%s base_url is required", name)
	}
	if client == nil {
		client = http.DefaultClient
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/health", nil)
	if err != nil {
		return fmt.Errorf("create %s health request: %w", name, err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("call %s health: %w", name, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read %s health response: %w", name, err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s health returned %s: %s", name, resp.Status, string(body))
	}

	var output response
	if err := json.Unmarshal(body, &output); err != nil {
		return fmt.Errorf("decode %s health response: %w", name, err)
	}
	if strings.ToLower(strings.TrimSpace(output.Status)) != "ok" {
		return fmt.Errorf("%s health status is %q", name, output.Status)
	}
	return nil
}
