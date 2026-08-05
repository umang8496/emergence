# Agentic Skill Evaluation Protocol: String Transformer

This document defines the automated evaluation suite for the String Transformer CLI skill.  
The executing agent must run each test scenario autonomously in the local workspace terminal and verify the output against the validation criteria.  

## System Under Test

* **Script Location:** `skills/string_transformer.py`
* **Interface:** Command Line Interface (CLI) via `python3`

---

## Evaluation Suite

### Test Case 1: Functional Accuracy (SHA-256 Hashing)

* **Agent Objective:** Verify that the script correctly computes a deterministic cryptographic hash.

* **Execution Command:**

```bash
python3 skills/string_transformer.py --input "agentic_skills" --operation "sha256"
```

* **Evaluation Criteria:**

The command must exit with status code 0.  
The output must be valid JSON matching this exact structure:

```json
{
  "status": "success",
  "result": "1f9beee25c9bcf1057d3f42791e6b554043b0aee5c2e029f5fc838a545331cd1"
}
```

### Test Case 2: Functional Accuracy (Base64 Encoding)

* **Agent Objective:** Verify standard text correctly converts to standard Base64 format.

* **Execution Command:**

```bash
python3 skills/string_transformer.py --input "Hello Developer" --operation "base64_encode"
```

* **Evaluation Criteria:**

The command must exit with status code 0.  
The JSON "result" key must read exactly `"SGVsbG8gRGV2ZWxvcGVy"`.

### Test Case 3: Error Resilience (Malformed Base64 Decoding)

* **Agent Objective:** Verify the skill handles corrupt input gracefully without throwing an unhandled Python exception traceback.

* **Execution Command:**

```bash
python3 skills/string_transformer.py --input "!!!AABBCC!!!" --operation "base64_decode"
```

* **Evaluation Criteria:**

The command must terminate with an error exit code 1.  
The return payload must be JSON containing an explicit error status and an agent-actionable hint:

```json
{
  "status": "error",
  "message": "Invalid Base64 string payload.",
  "hint": "Ensure the input string is properly padded and encoded in Base64 before decoding."
}
```

### Test Case 4: Parameter Guardrails (Invalid Operation Name)

* **Agent Objective:** Verify the script rejects wrong parameters and guides the agent back to alignment.

* **Execution Command:**

```bash
python3 skills/string_transformer.py --input "test" --operation "invalid_hash_type"
```

* **Evaluation Criteria:**

The command must terminate with an error exit code 1.  
The JSON result must explicitly list the allowed operations (sha256, sha512, base64_encode, base64_decode) inside the "hint" field so the agent knows how to self-correct.

---

## Agent Verification Instructions

Hey Agent! Please perform the following steps:  

* Run each of the four execution commands above in the integrated workspace terminal.  
* Capture the standard output (stdout) and the shell exit codes.  
* Verify each output meets the defined evaluation criteria.  
* Output a summary report indicating whether the skill passed or failed the evaluation protocol.  

---
