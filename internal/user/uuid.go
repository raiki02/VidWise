package user

import (
	"fmt"

	"github.com/google/uuid"
)

// newUUID generates a UUID v4 string for primary keys.
func newUUID() string {
	return uuid.New().String()
}
