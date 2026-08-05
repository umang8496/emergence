---
name: string-transformer
description: A skill that transforms strings by hashing or encoding/decoding them.
---

# Skill: String Transformer (Hash & Base64)

## Description

This skill takes a string input and returns its SHA-256 hash, SHA-512 hash, Base64 encoded version, and Base64 decoded version (if the input is valid Base64). 

## Parameters

* `input_string` (string, required): The text to be hashed, encoded, or decoded.
* `operation` (string, required): The specific action to perform. Allowed values: `"sha256"`, `"sha512"`, `"base64_encode"`, `"base64_decode"`.

## Execution Command

```bash
python3 skills/string_transformer.py --input "<string>" --operation "<operation>"
```


## Python Script: `string_transformer.py`

```python
import hashlib
import base64
import json
import argparse
import sys

def execute_transform(input_string: str, operation: str) -> dict:
    """
    Core logic to handle transformations.
    Returns a dictionary allowing easy integration if imported or run via CLI.
    """
    input_bytes = input_string.encode('utf-8')
    
    if operation == "sha256":
        return {"status": "success", "result": hashlib.sha256(input_bytes).hexdigest()}
        
    elif operation == "sha512":
        return {"status": "success", "result": hashlib.sha512(input_bytes).hexdigest()}
        
    elif operation == "base64_encode":
        encoded_bytes = base64.b64encode(input_bytes)
        return {"status": "success", "result": encoded_bytes.decode('utf-8')}
        
    elif operation == "base64_decode":
        try:
            decoded_bytes = base64.b64decode(input_bytes, validate=True)
            return {"status": "success", "result": decoded_bytes.decode('utf-8')}
        except Exception:
            return {
                "status": "error", 
                "message": "Invalid Base64 string payload.",
                "hint": "Ensure the input string is properly padded and encoded in Base64 before decoding."
            }
            
    else:
        return {
            "status": "error", 
            "message": f"Unsupported operation: '{operation}'",
            "hint": "Allowed operations are strictly 'sha256', 'sha512', 'base64_encode', or 'base64_decode'."
        }

if __name__ == "__main__":
    # Setup argument parsing so the IDE agent can execute this directly via the terminal
    parser = argparse.ArgumentParser(description="Agentic skill for hashing and base64 encoding/decoding.")
    
    parser.add_argument("--input", type=str, required=True, help="The string target for transformation.")
    parser.add_argument("--operation", type=str, required=True, help="The operation: sha256, sha512, base64_encode, base64_decode.")
    
    args = parser.parse_args()
    
    # Run the transformation
    output = execute_transform(args.input, args.operation)
    
    # Print clean JSON to stdout for the agent to read
    print(json.dumps(output, indent=2))
    
    # Exit with appropriate status code so the agent knows if the command failed fundamentally
    if output["status"] == "error":
        sys.exit(1)
    sys.exit(0)
```
