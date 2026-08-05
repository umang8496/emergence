---
name: log-sanitizer
description: A local Python script that scrubs sensitive information from log files before they are sent to an LLM for debugging.
---

# Skill: Log Sanitizer (PII Stripper)

## Description

This security skill processes a raw log file **locally** to redact sensitive information before its contents are read or shared with an LLM/agent.  
It replaces sensitive values with structured placeholders (e.g. `[REDACTED_EMAIL]`) and writes the result to a new `*_sanitized` file alongside the original.

It targets four categories of sensitive data:

1. **Secrets & tokens** — Bearer tokens, JWTs, AWS access keys, and generic `api_key=` / `token:` / `password=` style secrets.
2. **PII** — email addresses, phone numbers, IP addresses, and SSNs.
3. **Database credentials** — connection strings such as `postgres://user:password@host/db`.
4. **Payment data** — credit card numbers and PEM-formatted private key blocks.

## Mandatory Usage Rule for Agents

**Never read the original/raw log file directly.**  
Always invoke this script first and only read the resulting `*_sanitized` file.  
Reading the raw file defeats the purpose of this skill, since it exposes the unredacted content directly to the LLM context.  

## Parameters

- `--file` (string, required): The relative or absolute path to the raw log file in the workspace.

## Execution Command

```bash
python3 skills/log_sanitizer.py --file "<path_to_log_file>"
```

## Output Format

Prints a JSON object to stdout and exits with status `0` on success or `1` on error.

### Success

```json
{
  "status": "success",
  "redacted_file": "<path_to_log_file>_sanitized.<ext>",
  "metrics": {
    "emails_redacted": 0,
    "ips_redacted": 0,
    "keys_redacted": 0,
    "db_urls_redacted": 0,
    "phones_redacted": 0,
    "ssns_redacted": 0,
    "cards_redacted": 0,
    "private_keys_redacted": 0
  }
}
```

### Error

```json
{
  "status": "error",
  "message": "Target log file not found at: <path>"
}
```

## Known Limitations

- Regex-based detection is heuristic, not exhaustive.  
  It will miss some secret formats (e.g. Slack/Stripe-style tokens without a recognized label) and may occasionally over-redact numeric sequences that resemble phone numbers or card numbers.
- Credit card matching does not perform Luhn validation, so any bare 13–16 digit number is treated as a potential card number.
- The whole file is loaded into memory; very large log files are not streamed.

---
