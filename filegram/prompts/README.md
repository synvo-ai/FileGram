# Prompts Directory

This directory contains all the prompt templates and tool descriptions used by FileGram, inspired by [OpenCode](https://github.com/anthropics/opencode).

## Directory Structure

```
prompts/
├── __init__.py      # Package exports (load_tool_prompt, load_agent_prompt, etc.)
├── loader.py        # Prompt loader implementation
├── README.md        # This file
│
├── tools/           # Tool descriptions and usage guidelines
│   ├── __init__.py
│   ├── bash.txt     # Shell command execution
│   ├── read.txt     # File reading
│   ├── write.txt    # File writing
│   ├── edit.txt     # File editing (string replacement)
│   ├── glob.txt     # File pattern matching
│   ├── grep.txt     # Content search
│   ├── task.txt     # Sub-agent spawning
│   ├── todoread.txt # Read task list
│   ├── todowrite.txt # Task list management
│   ├── plan_enter.txt # Enter plan mode
│   └── plan_exit.txt  # Exit plan mode
│
├── agents/          # Agent-specific system prompts
│   ├── __init__.py
│   ├── explore.txt  # File search specialist agent
│   └── compaction.txt # Conversation summarizer agent
│
└── session/         # Session-level system prompts
    ├── __init__.py
    ├── system.txt   # Main system prompt
    └── plan_mode.txt # Plan mode restrictions
```

## Usage

The prompts are loaded automatically by each tool/agent using the `PromptLoader`. This mirrors OpenCode's TypeScript static imports.

### Loading Prompts

```python
from filegram.prompts import load_tool_prompt, load_agent_prompt, load_session_prompt

# Load tool description
bash_description = load_tool_prompt("bash")

# Load agent prompt
explore_prompt = load_agent_prompt("explore")

# Load session prompt
system_prompt = load_session_prompt("system")
```

### How It Works

In TypeScript (OpenCode), prompts are loaded via static imports:
```typescript
import DESCRIPTION from "./bash.txt"
```

In Python (FileGram), we use `importlib.resources` to achieve the same effect:
```python
from ..prompts import load_tool_prompt
DESCRIPTION = load_tool_prompt("bash")
```

Each tool class then uses this description in its `description` property:
```python
class BashTool(BaseTool):
    @property
    def description(self) -> str:
        return DESCRIPTION  # Loaded from bash.txt
```

## Source

These prompts are adapted from OpenCode's implementation at:
- `packages/opencode/src/tool/*.txt`
- `packages/opencode/src/agent/prompt/*.txt`
- `packages/opencode/src/session/prompt/*.txt`

## Customization

You can customize these prompts for your specific use case:

1. Edit the `.txt` files directly to modify tool descriptions
2. The changes will be reflected when the agent runs
3. Use clear, concise language that helps the LLM understand when and how to use each tool
