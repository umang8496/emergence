#!/usr/bin/env python3

import hashlib
import json
import sys


def digest_message(message: str, algorithm: str = "sha1") -> dict:
    """
    Calculate hash of the message using the specified algorithm.
    
    Supported algorithms:
    - sha1, sha224, sha256, sha384, sha512 (SHA family)
    - md5
    - blake2b, blake2s
    - sha3_256, sha3_384, sha3_512
    """
    algorithm = algorithm.lower()
    
    # Validate algorithm
    supported_algorithms = hashlib.algorithms_available
    if algorithm not in supported_algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: {', '.join(sorted(supported_algorithms))}")
    
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(message.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()
    
    return {
        "algorithm": algorithm,
        "result": hash_hex
    }


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(json.dumps({
            "error": "Usage: python3 calculate-hash.py <text> [algorithm]",
            "default_algorithm": "sha1",
            "supported_algorithms": list(sorted(hashlib.algorithms_available))
        }), file=sys.stderr)
        sys.exit(1)
    
    try:
        text = sys.argv[1]
        algorithm = sys.argv[2] if len(sys.argv) > 2 else "sha1"
        
        result = digest_message(text, algorithm)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": f"Failed to calculate hash: {str(e)}"}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
