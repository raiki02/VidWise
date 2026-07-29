package search

import (
	"bufio"
	"bytes"
	"strings"
	"testing"
)

func TestWriteRedisCommandUsesRESPArray(t *testing.T) {
	var buf bytes.Buffer
	if _, err := writeRedisCommand(&buf, "SETEX", "key", "60", "value"); err != nil {
		t.Fatalf("writeRedisCommand() error = %v", err)
	}
	want := "*4\r\n$5\r\nSETEX\r\n$3\r\nkey\r\n$2\r\n60\r\n$5\r\nvalue\r\n"
	if buf.String() != want {
		t.Fatalf("command = %q, want %q", buf.String(), want)
	}
}

func TestReadRedisReplyReadsBulkStringAndNil(t *testing.T) {
	got, err := readRedisReply(bufio.NewReader(strings.NewReader("$5\r\nhello\r\n")))
	if err != nil {
		t.Fatalf("readRedisReply() error = %v", err)
	}
	if string(got) != "hello" {
		t.Fatalf("bulk = %q", string(got))
	}

	got, err = readRedisReply(bufio.NewReader(strings.NewReader("$-1\r\n")))
	if err != nil {
		t.Fatalf("readRedisReply(nil) error = %v", err)
	}
	if got != nil {
		t.Fatalf("nil bulk = %#v, want nil", got)
	}
}
