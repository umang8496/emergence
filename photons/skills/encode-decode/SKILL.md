---
name: encode-decode
description: Encode or decode text using base64 or URL encoding.
---

# Encode Decode Skill

This skill performs encoding and decoding operations on text.  
Whenever the prompt asks for encoding or decoding related actions or intent then try using it.

## Supported Operations

- base64_encode
- base64_decode
- url_encode
- url_decode

## Examples

- "Encode this text in base64: hello world"
- "Decode this base64 string: aGVsbG8="
- "URL encode this string: hello world"
- "Decode this URL string: hello%20world"

## Instructions

Run the script:

```bash
python3 encode_decode.py <operation> "<text>"
```
