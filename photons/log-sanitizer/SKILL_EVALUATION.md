# Agentic Skill Evaluation Protocol: Log Sanitizer

This document defines the automated evaluation suite for the Log Sanitizer CLI skill.  
The executing agent must run each test scenario autonomously in the local workspace terminal and verify the output against the validation criteria.

## System Under Test

* **Script Location:** `skills/log_sanitizer.py`
* **Interface:** Command Line Interface (CLI) via `python3`

---

## Evaluation Suite

### Test Case 1: Mixed PII Redaction

**Agent Objective:**  
  Verify email, IP, phone, SSN, and credit card values are redacted simultaneously.

**Pre-condition:**  
  Create a temporary file named `eval_raw_pii.log` containing exactly:  
  `ERROR: User admin@company.com failed to connect from 192.168.1.50. Card 4111-2222-3333-4444, SSN 000-12-3456, phone +1-555-867-5309`

**Execution Command:**

  ```bash
  python3 skills/log_sanitizer.py --file "eval_raw_pii.log"
  ```

**Evaluation Criteria:**  
  The command must exit with status code `0`. The JSON output must match:

  ```json
  {
    "status": "success",
    "redacted_file": "eval_raw_pii_sanitized.log",
    "metrics": {
      "emails_redacted": 1,
      "ips_redacted": 1,
      "keys_redacted": 0,
      "db_urls_redacted": 0,
      "phones_redacted": 1,
      "ssns_redacted": 1,
      "cards_redacted": 1,
      "private_keys_redacted": 0
    }
  }
```

### Test Case 2: Secret & Token Redaction

**Agent Objective:**  
  Verify Bearer tokens, standalone JWTs, AWS access keys, and generic `key=value` secrets are all redacted.

**Pre-condition:**  
  Create a temporary file named `eval_raw_secrets.log` containing exactly:  
  `Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c and AWS key AKIAIOSFODNN7EXAMPLE and generic api_key=sk_live_51H8xyzABCDEF1234567890 and password: hunter2isbad`

**Execution Command:**

  ```bash
  python3 skills/log_sanitizer.py --file "eval_raw_secrets.log"
  ```

**Evaluation Criteria:**  
  The command must exit with status code `0`.  
  `metrics.keys_redacted` must equal `4`, and the sanitized file must not contain the substrings `eyJ`, `AKIA`, `sk_live_`, or `hunter2isbad`.  

### Test Case 3: Private Key Block Redaction

**Agent Objective:**  
  Ensure multiline PEM-formatted private keys are stripped completely.

**Pre-condition:**  
  Create a temporary file named `eval_raw_key.log` containing:

  ```text
  ERROR: Auth failed for block:
  -----BEGIN PRIVATE KEY-----
  MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC3c
  -----END PRIVATE KEY-----
  Connection closed.
  ```

**Execution Command:**

  ```bash
  python3 skills/log_sanitizer.py --file "eval_raw_key.log"
  ```

**Evaluation Criteria:**  
  The command must exit with status code `0`. `metrics.private_keys_redacted` must equal `1`, and `eval_raw_key_sanitized.log` must read:

  ```text
  ERROR: Auth failed for block:
  [REDACTED_PRIVATE_KEY_BLOCK]
  Connection closed.
  ```

### Test Case 4: Error Handling for Non-Existent Path

**Agent Objective:**  
  Verify the script rejects missing files cleanly without an unhandled stack trace.

**Execution Command:**

  ```bash
  python3 skills/log_sanitizer.py --file "non_existent_directory/missing.log"
  ```

**Evaluation Criteria:**  
  The command must terminate with exit code `1`. The output must be valid JSON:

  ```json
  {
    "status": "error",
    "message": "Target log file not found at: non_existent_directory/missing.log"
  }
  ```

### Test Case 5: DB Connection String vs. Email Pattern Ordering

**Agent Objective:**  
Verify a `user:password@host` connection string is redacted as a whole by `db_url`, and that the generic `email` pattern does not greedily consume the `password@host` portion first (which would leak a password fragment and prevent `db_url` from matching at all).

**Pre-condition:**  
  Create a temporary file named `eval_raw_dburl.log` containing exactly:
  `WARN: Falling back to secondary replica connection string: mysql://svc_reader:R3plica#Secret@db-replica-02.internal.net:3306/billing`

**Execution Command:**

  ```bash
  python3 skills/log_sanitizer.py --file "eval_raw_dburl.log"
  ```

**Evaluation Criteria:**  
  The command must exit with status code `0`.  
  `metrics.db_urls_redacted` must equal `1`, `metrics.emails_redacted` must equal `0`, and the sanitized file must not contain the substrings `R3plica`, `Secret`, or `svc_reader`.

---

## Agent Verification Instructions

Please perform the following steps:

1. Generate the `eval_raw_pii.log`, `eval_raw_secrets.log`, `eval_raw_key.log`, and `eval_raw_dburl.log` files inside the workspace, matching the pre-conditions above.
2. Run the five execution commands above in the integrated workspace terminal.
3. Capture stdout and the shell exit codes, and verify each output meets the evaluation criteria.
4. Delete the temporary input and `*_sanitized` files created for testing.
5. Report a pass/fail summary for each test case.

---
