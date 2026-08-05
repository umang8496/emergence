import re
import json
import argparse
import sys
import os

# Expanded engineering compliance regex patterns
PATTERNS = {
    # Email regex matching standard formats with optional subdomains
    "email": re.compile(r'[\w\.-]+@[\w\.-]+\.\w+'),
    # IP address regex matching IPv4 formats
    "ip_address": re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
    # Bearer/authorization headers, keeping the "Bearer" prefix for context
    "bearer_token": re.compile(r'(?i)\b(Bearer\s+)[A-Za-z0-9\-_\.]{15,}'),
    # Standalone JSON Web Tokens (header.payload.signature) not preceded by "Bearer"
    "jwt": re.compile(r'\bey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b'),
    # AWS access/session key IDs (AKIA... / ASIA...)
    "aws_key": re.compile(r'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b'),
    # Generic named secrets, e.g. api_key=..., token: "...", password=...
    "generic_secret": re.compile(r'(?i)\b(api[_-]?key|secret|token|password|pwd|access[_-]?key)(\s*[:=]\s*)["\']?[A-Za-z0-9\-_/+=]{8,}["\']?'),
    # Database URL regex matching common connection strings
    "db_url": re.compile(r'[a-zA-Z]+://[^:]+:[^@]+@[^/]+/[^?\s]+'),
    # Phone number regex matching variations with spaces, dashes, or parentheses
    "phone": re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    # Standard 9-digit SSN matching sequence
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    # Luhn algorithm targets: standard credit card numeric lengths
    "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{13,16}\b'),
    # Multiline block identifier for cryptographic secret keys
    "private_key": re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----')
}

def sanitize_log(file_path: str) -> dict:
    if not os.path.exists(file_path):
        return {"status": "error", "message": f"Target log file not found at: {file_path}"}
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        metrics = {
            "emails_redacted": 0, 
            "ips_redacted": 0, 
            "keys_redacted": 0, 
            "db_urls_redacted": 0,
            "phones_redacted": 0,
            "ssns_redacted": 0,
            "cards_redacted": 0,
            "private_keys_redacted": 0
        }
        
        # db_url must run before "email", otherwise "email" greedily consumes the
        # password@host portion of user:password@host connection strings first
        content, metrics["db_urls_redacted"] = PATTERNS["db_url"].subn("[REDACTED_DB_CREDENTIALS]", content)
        content, metrics["emails_redacted"] = PATTERNS["email"].subn("[REDACTED_EMAIL]", content)
        content, metrics["ips_redacted"] = PATTERNS["ip_address"].subn("[REDACTED_IP]", content)
        content, n_bearer = PATTERNS["bearer_token"].subn(r'\1[REDACTED_KEY]', content)
        content, n_jwt = PATTERNS["jwt"].subn("[REDACTED_KEY]", content)
        content, n_aws = PATTERNS["aws_key"].subn("[REDACTED_KEY]", content)
        content, n_generic = PATTERNS["generic_secret"].subn(r'\1\2[REDACTED_KEY]', content)
        metrics["keys_redacted"] = n_bearer + n_jwt + n_aws + n_generic
        content, metrics["phones_redacted"] = PATTERNS["phone"].subn("[REDACTED_PHONE]", content)
        content, metrics["ssns_redacted"] = PATTERNS["ssn"].subn("[REDACTED_SSN]", content)
        content, metrics["cards_redacted"] = PATTERNS["credit_card"].subn("[REDACTED_CREDIT_CARD]", content)
        content, metrics["private_keys_redacted"] = PATTERNS["private_key"].subn("[REDACTED_PRIVATE_KEY_BLOCK]", content)
        
        # Write sanitized output to a twin file path
        base, ext = os.path.splitext(file_path)
        sanitized_path = f"{base}_sanitized{ext}"
        
        with open(sanitized_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return {
            "status": "success",
            "redacted_file": sanitized_path,
            "metrics": metrics
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local PII and Secret Log Sanitizer for Coding Agents.")
    parser.add_argument("--file", type=str, required=True, help="Path to the source log file.")
    args = parser.parse_args()
    
    result = sanitize_log(args.file)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "success" else 1)
