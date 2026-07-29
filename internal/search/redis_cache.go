package search

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"strconv"
	"strings"
	"time"
)

type RedisCacheConfig struct {
	Addr      string
	Password  string
	DB        int
	KeyPrefix string
	TTL       time.Duration
	Timeout   time.Duration
}

type RedisCache struct {
	cfg RedisCacheConfig
}

func NewRedisCache(cfg RedisCacheConfig) (*RedisCache, error) {
	if strings.TrimSpace(cfg.Addr) == "" {
		return nil, errors.New("redis addr is required")
	}
	if cfg.TTL <= 0 {
		cfg.TTL = 5 * time.Minute
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = time.Second
	}
	if cfg.KeyPrefix == "" {
		cfg.KeyPrefix = "vidwise:search:"
	}
	return &RedisCache{cfg: cfg}, nil
}

func (c *RedisCache) Get(query string) (*SearchResult, bool) {
	if c == nil {
		return nil, false
	}
	key := c.key(query)
	if key == "" {
		return nil, false
	}
	value, err := c.command("GET", key)
	if err != nil || len(value) == 0 {
		return nil, false
	}
	var result SearchResult
	if err := json.Unmarshal(value, &result); err != nil {
		return nil, false
	}
	result.Cached = true
	return &result, true
}

func (c *RedisCache) Set(query string, result SearchResult) {
	if c == nil {
		return
	}
	key := c.key(query)
	if key == "" {
		return
	}
	body, err := json.Marshal(cloneSearchResult(result))
	if err != nil {
		return
	}
	_, _ = c.command("SETEX", key, strconv.Itoa(int(c.cfg.TTL.Seconds())), string(body))
}

func (c *RedisCache) key(query string) string {
	query = normalizeCacheKey(query)
	if query == "" {
		return ""
	}
	return c.cfg.KeyPrefix + query
}

func (c *RedisCache) command(args ...string) ([]byte, error) {
	conn, err := net.DialTimeout("tcp", c.cfg.Addr, c.cfg.Timeout)
	if err != nil {
		return nil, fmt.Errorf("dial redis: %w", err)
	}
	defer conn.Close()
	deadline := time.Now().Add(c.cfg.Timeout)
	_ = conn.SetDeadline(deadline)

	reader := bufio.NewReader(conn)
	if c.cfg.Password != "" {
		if _, err := writeRedisCommand(conn, "AUTH", c.cfg.Password); err != nil {
			return nil, err
		}
		if _, err := readRedisReply(reader); err != nil {
			return nil, fmt.Errorf("redis auth: %w", err)
		}
	}
	if c.cfg.DB > 0 {
		if _, err := writeRedisCommand(conn, "SELECT", strconv.Itoa(c.cfg.DB)); err != nil {
			return nil, err
		}
		if _, err := readRedisReply(reader); err != nil {
			return nil, fmt.Errorf("redis select: %w", err)
		}
	}
	if _, err := writeRedisCommand(conn, args...); err != nil {
		return nil, err
	}
	return readRedisReply(reader)
}

func writeRedisCommand(w io.Writer, args ...string) (int, error) {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "*%d\r\n", len(args))
	for _, arg := range args {
		fmt.Fprintf(&buf, "$%d\r\n%s\r\n", len(arg), arg)
	}
	return w.Write(buf.Bytes())
}

func readRedisReply(r *bufio.Reader) ([]byte, error) {
	prefix, err := r.ReadByte()
	if err != nil {
		return nil, err
	}
	line, err := r.ReadString('\n')
	if err != nil {
		return nil, err
	}
	line = strings.TrimSuffix(strings.TrimSuffix(line, "\n"), "\r")
	switch prefix {
	case '+':
		return []byte(line), nil
	case '-':
		return nil, errors.New(line)
	case ':':
		return []byte(line), nil
	case '$':
		n, err := strconv.Atoi(line)
		if err != nil {
			return nil, fmt.Errorf("parse redis bulk length: %w", err)
		}
		if n < 0 {
			return nil, nil
		}
		body := make([]byte, n+2)
		if _, err := io.ReadFull(r, body); err != nil {
			return nil, err
		}
		return body[:n], nil
	default:
		return nil, fmt.Errorf("unknown redis reply prefix %q", prefix)
	}
}
