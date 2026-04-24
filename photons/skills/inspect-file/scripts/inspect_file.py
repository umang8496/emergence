#!/usr/bin/env python3

import sys
import json
from pathlib import Path


EXTENSION_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".md": "markdown",
    ".json": "json",
    ".txt": "text",
    ".java": "java",
    ".rs": "rust"
}


def detect_language(path: Path):
    return EXTENSION_MAP.get(path.suffix.lower(), "unknown")


def summarize_content(content: str, max_lines=5):
    lines = content.strip().split("\n")
    return "\n".join(lines[:max_lines])


def inspect_file(file_path: str):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError("File does not exist")

    content = path.read_text(errors="ignore")

    lines = content.split("\n")
    size = path.stat().st_size

    return {
        "status": "success",
        "file": str(path),
        "size_bytes": size,
        "line_count": len(lines),
        "language": detect_language(path),
        "preview": summarize_content(content)
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({
            "status": "error",
            "message": "Usage: python3 inspect-file.py <file>"
        }), file=sys.stderr)
        sys.exit(1)

    try:
        file_path = sys.argv[1]
        result = inspect_file(file_path)
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": str(e)
        }), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()