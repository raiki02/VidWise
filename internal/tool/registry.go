package tool

import (
	"fmt"
	"sync"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"
)

// Registry holds all registered tools, serving as the single source of truth
// for both the Eino agent graph and the MCP server.
type Registry struct {
	mu    sync.RWMutex
	tools map[string]Entry
}

type Entry struct {
	Tool tool.InvokableTool
	Info *schema.ToolInfo
}

func NewRegistry() *Registry {
	return &Registry{tools: make(map[string]Entry)}
}

func (r *Registry) Register(name string, t tool.InvokableTool, info *schema.ToolInfo) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tools[name] = Entry{Tool: t, Info: info}
}

func (r *Registry) Get(name string) (tool.InvokableTool, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	e, ok := r.tools[name]
	if !ok {
		return nil, fmt.Errorf("tool not found: %s", name)
	}
	return e.Tool, nil
}

func (r *Registry) GetInfo(name string) (*schema.ToolInfo, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	e, ok := r.tools[name]
	if !ok {
		return nil, fmt.Errorf("tool not found: %s", name)
	}
	return e.Info, nil
}

func (r *Registry) List() map[string]Entry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make(map[string]Entry, len(r.tools))
	for k, v := range r.tools {
		out[k] = v
	}
	return out
}
