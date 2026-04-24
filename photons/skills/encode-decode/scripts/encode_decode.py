#!/usr/bin/env python3

import sys
import json
import base64
import urllib.parse


def encode_decode(text: str, operation: str) -> dict:
    if not text:
        raise ValueError("Text cannot be empty")

    if operation == "base64_encode":
        result = base64.b64encode(text.encode()).decode()

    elif operation == "base64_decode":
        result = base64.b64decode(text.encode()).decode()

    elif operation == "url_encode":
        result = urllib.parse.quote(text)

    elif operation == "url_decode":
        result = urllib.parse.unquote(text)

    else:
        raise ValueError(f"Unsupported operation: {operation}")

    return {
        "status": "success",
        "operation": operation,
        "result": result
    }


def main():
    if len(sys.argv) < 3:
        print(json.dumps({
            "status": "error",
            "message": "Usage: python3 encode-decode.py <operation> <text>"
        }), file=sys.stderr)
        sys.exit(1)

    try:
        operation = sys.argv[1]
        text = sys.argv[2]

        output = encode_decode(text, operation)
        print(json.dumps(output))

    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": str(e)
        }), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()