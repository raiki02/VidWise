package task

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

type trackerSnapshot struct {
	Version int           `json:"version"`
	Tasks   []TrackedTask `json:"tasks"`
}

type FileTrackerStore struct {
	mu     sync.Mutex
	path   string
	tasks  map[string]TrackedTask
	loaded bool
}

func NewFileTrackerStore(path string) *FileTrackerStore {
	return &FileTrackerStore{path: strings.TrimSpace(path)}
}

func (s *FileTrackerStore) Load(context.Context) ([]TrackedTask, error) {
	if s == nil || s.path == "" {
		return nil, nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.loadLocked(); err != nil {
		return nil, err
	}

	tasks := make([]TrackedTask, 0, len(s.tasks))
	for _, task := range s.tasks {
		tasks = append(tasks, copyTrackedTask(task))
	}
	return tasks, nil
}

func (s *FileTrackerStore) SaveTask(_ context.Context, task TrackedTask) error {
	if s == nil || s.path == "" || task.ID == "" {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.loadLocked(); err != nil {
		return err
	}
	s.tasks[task.ID] = copyTrackedTask(task)
	return s.saveLocked()
}

func (s *FileTrackerStore) DeleteTasks(_ context.Context, ids []string) error {
	if s == nil || s.path == "" || len(ids) == 0 {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.loadLocked(); err != nil {
		return err
	}
	for _, id := range ids {
		delete(s.tasks, id)
	}
	return s.saveLocked()
}

func (s *FileTrackerStore) loadLocked() error {
	if s.loaded {
		return nil
	}
	s.loaded = true
	s.tasks = make(map[string]TrackedTask)

	data, err := os.ReadFile(s.path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return fmt.Errorf("read task tracker snapshot: %w", err)
	}

	var snapshot trackerSnapshot
	if err := json.Unmarshal(data, &snapshot); err != nil {
		return fmt.Errorf("decode task tracker snapshot: %w", err)
	}
	for _, task := range snapshot.Tasks {
		if strings.TrimSpace(task.ID) == "" {
			continue
		}
		s.tasks[task.ID] = copyTrackedTask(task)
	}
	return nil
}

func (s *FileTrackerStore) saveLocked() error {
	tasks := make([]TrackedTask, 0, len(s.tasks))
	for _, task := range s.tasks {
		tasks = append(tasks, copyTrackedTask(task))
	}
	sort.Slice(tasks, func(i, j int) bool {
		if tasks[i].UpdatedAt.Equal(tasks[j].UpdatedAt) {
			return tasks[i].ID < tasks[j].ID
		}
		return tasks[i].UpdatedAt.After(tasks[j].UpdatedAt)
	})

	data, err := json.Marshal(trackerSnapshot{
		Version: 1,
		Tasks:   tasks,
	})
	if err != nil {
		return fmt.Errorf("encode task tracker snapshot: %w", err)
	}

	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create task tracker storage dir: %w", err)
	}

	tmp, err := os.CreateTemp(dir, "."+filepath.Base(s.path)+".*.tmp")
	if err != nil {
		return fmt.Errorf("create task tracker temp file: %w", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)

	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("write task tracker temp file: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("sync task tracker temp file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close task tracker temp file: %w", err)
	}
	if err := os.Rename(tmpName, s.path); err != nil {
		return fmt.Errorf("replace task tracker snapshot: %w", err)
	}
	return nil
}
