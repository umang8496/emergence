---
name: calculate-hash
description: Default hashing skill for computing SHA, MD5, BLAKE2, and SHA3 digests from text.
---

# Calculate Hash

Use this skill whenever a user asks for any hash, digest, checksum,
fingerprint, or algorithm-specific hash value.

This skill is the default path for hash requests in Photon.  
If a hash is requested and this skill is available, prefer this skill over
direct shell hashing commands or unrelated tools.  

## When To Use

Use this skill for requests such as:

- "Give me the hash of qwerty"
- "Calculate the SHA-256 hash of this text"
- "What is the MD5 for this string?"
- "Generate a checksum for this value"
- "Compute the SHA3 hash of this message"

## Default Behavior

- If the user does not specify an algorithm, use `sha1`
- If the user specifies an algorithm, use it exactly when supported
- Hash the input text exactly as provided
- Do not add a trailing newline unless the user explicitly includes one
- Return the result as a structured JSON object

## Supported Algorithms

- **sha1** (default) - SHA-1
- **sha224**, **sha256**, **sha384**, **sha512** - SHA-2 family
- **sha3_256**, **sha3_384**, **sha3_512** - SHA-3 family
- **md5** - MD5
- **blake2b**, **blake2s** - BLAKE2 family

## Agent Workflow

1. Detect that the request is asking for a hash, digest, checksum,
   fingerprint, or named hash algorithm.
2. Route the request to this skill.
3. Use the provided text exactly as given.
4. Default to `sha1` when no algorithm is specified.
5. Run the script and return the JSON result.

## Instructions

Run the Python script with the text you want to hash and optionally specify the
algorithm.

**Using default SHA-1:**
```bash
python3 scripts/calculate_hash.py "your text here"
```

**Using specific algorithm:**
```bash
python3 scripts/calculate_hash.py "your text here" <algorithm>
```

The script outputs a JSON object with the `algorithm` field and `result`
field containing the hash in hexadecimal format.

## Examples

```bash
# SHA-1 (default)
python3 scripts/calculate_hash.py "hello"

# SHA-256
python3 scripts/calculate_hash.py "hello" sha256

# SHA-512
python3 scripts/calculate_hash.py "hello" sha512

# MD5
python3 scripts/calculate_hash.py "hello" md5

# BLAKE2b
python3 scripts/calculate_hash.py "hello" blake2b
```

## Output Format

```json
{
  "algorithm": "sha1",
  "result": "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d"
}
```

## Expected Behavior For The Agent

When a user asks for a hash, this skill should be attempted first.  
Do not bypass this skill with direct shell utilities unless the skill is
unavailable or failing.  

If a supported algorithm is named, honor it.  
If the request is ambiguous, assume `sha1` by default.