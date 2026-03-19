"""
tools.py
────────
Two responsibilities:
  1. TOOL_DEFINITIONS  — JSON descriptions sent to Claude on every request.
  2. Implementations   — Local Python functions that do the actual work.

Adding a new tool requires three steps, all in this file:
  Step 1 — Add its JSON definition to TOOL_DEFINITIONS.
  Step 2 — Implement the function below.
  Step 3 — Register it in the TOOL_REGISTRY at the bottom.

Nothing else needs to change anywhere else.
"""

import os
from typing import Callable

# ---------------------------------------------------------------------------
# Step 1 — Tool Definitions
# What Claude sees. Descriptions are routing logic written in English.
# The quality of these descriptions directly determines when and how
# Claude decides to invoke each tool.
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a source code file. "
            "Use this to inspect a specific file for bugs, security issues, or code quality problems. "
            "Returns the full file contents with line numbers prepended."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The relative or absolute path to the file."
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "list_files",
        "description": (
            "List all source code files in a directory recursively. "
            "Use this when given a directory path to discover which files exist before reading them. "
            "Returns a list of file paths filtered to common code extensions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "The directory path to scan."
                }
            },
            "required": ["directory"]
        }
    },
    {
        "name": "search_code",
        "description": (
            "Search for a text pattern across all code files in a directory. "
            "Use this to check if a suspicious pattern (e.g. eval(), exec(), hardcoded passwords) "
            "appears in multiple files. Returns matching lines with their file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The text pattern to search for."
                },
                "directory": {
                    "type": "string",
                    "description": "The directory to search in."
                }
            },
            "required": ["pattern", "directory"]
        }
    }
]

# ---------------------------------------------------------------------------
# Constants shared across implementations
# ---------------------------------------------------------------------------

_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go",
    ".rb", ".php", ".c", ".cpp", ".h", ".cs", ".rs", ".swift"
}

_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv",
    "venv", "dist", "build", ".next", ".cache"
}

# ---------------------------------------------------------------------------
# Step 2 — Tool Implementations
# Pure Python. No Claude, no API calls. Each function receives the arguments
# Claude decided to pass and returns a plain string result.
# ---------------------------------------------------------------------------

def read_file(path: str) -> str:
    """Read a single source file and return its contents with line numbers."""
    if not os.path.exists(path):
        return f"Error: File not found — {path}"

    if not os.path.isfile(path):
        return f"Error: Path is a directory, not a file — {path}"

    ext = os.path.splitext(path)[1].lower()
    if ext not in _CODE_EXTENSIONS:
        return f"Error: Unsupported file type '{ext}'. Supported: {', '.join(sorted(_CODE_EXTENSIONS))}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        numbered = "".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
        return f"File: {path} ({len(lines)} lines)\n\n{numbered}"
    except UnicodeDecodeError:
        return f"Error: Could not decode file as UTF-8 — {path}"
    except OSError as e:
        return f"Error reading file: {e}"


def list_files(directory: str) -> str:
    """Recursively list all source code files in a directory."""
    if not os.path.exists(directory):
        return f"Error: Directory not found — {directory}"

    if not os.path.isdir(directory):
        return f"Error: Path is a file, not a directory — {directory}"

    found = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in _CODE_EXTENSIONS:
                rel_path = os.path.relpath(os.path.join(root, fname), directory)
                found.append(rel_path)

    if not found:
        return f"No source code files found in '{directory}'"

    return f"Found {len(found)} source file(s) in '{directory}':\n\n" + \
           "\n".join(f"  {p}" for p in found)


def search_code(pattern: str, directory: str) -> str:
    """Search for a text pattern across all source code files in a directory."""
    if not os.path.exists(directory):
        return f"Error: Directory not found — {directory}"

    matches = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() not in _CODE_EXTENSIONS:
                continue
            full_path = os.path.join(root, fname)
            rel_path  = os.path.relpath(full_path, directory)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        if pattern.lower() in line.lower():
                            matches.append(f"  {rel_path}:{i} | {line.rstrip()}")
            except (UnicodeDecodeError, OSError):
                continue

    if not matches:
        return f"No matches found for pattern '{pattern}' in '{directory}'"

    return f"Found {len(matches)} match(es) for '{pattern}':\n\n" + "\n".join(matches)


# ---------------------------------------------------------------------------
# Step 3 — Tool Registry
# Maps tool names to their implementations.
# The dispatcher (execute_tool) uses this — no if/elif chain needed.
# Adding a new tool means adding one line here.
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "read_file":   lambda inputs: read_file(inputs["path"]),
    "list_files":  lambda inputs: list_files(inputs["directory"]),
    "search_code": lambda inputs: search_code(inputs["pattern"], inputs["directory"]),
}


def execute_tool(name: str, inputs: dict) -> str:
    """Dispatch a tool call from Claude to the correct local implementation."""
    handler = TOOL_REGISTRY.get(name)
    if handler is None:
        return f"Error: Unknown tool '{name}'. Available tools: {', '.join(TOOL_REGISTRY)}"
    return handler(inputs)
