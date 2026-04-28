# Photon

A curated collection of reusable skills for personal use.
This repository contains utilities and tools designed to extend LLM capabilities with specialized functions.

## Table of Contents

- [Calculate Hash](#calculate-hash)
- [Code Review QA](#code-review-qa)
- [Frontend Development](#frontend-development)
- [Tech Jokes](#tech-jokes)
- [Encode Decode](#encode-decode)
- [Historical Facts](#historical-facts)
- [Inspect File](#inspect-file)
- [Lint Markdown](#lint-markdown)
- [How To Invoke Skills In Prompts](#how-to-invoke-skills-in-prompts)
- [Project Structure](#project-structure)

---

## Skills

### Calculate Hash

Computes hashes of text using various algorithms.
Supports SHA-1, SHA-256, SHA-512, MD5, BLAKE2, SHA3, and more. Defaults to SHA-1.

**Location:** `skills/calculate-hash/scripts/`

**Supported Algorithms:**

- `sha1` (default) - SHA-1
- `sha224`, `sha256`, `sha384`, `sha512` - SHA-2 family
- `sha3_256`, `sha3_384`, `sha3_512` - SHA-3 family
- `md5` - MD5
- `blake2b`, `blake2s` - BLAKE2 family

**Usage:**

```bash
cd skills/calculate-hash
# Using default SHA-1
python3 scripts/calculate_hash.py "your text here"

# Using specific algorithm
python3 scripts/calculate_hash.py "your text here" sha256
python3 scripts/calculate_hash.py "your text here" sha512
python3 scripts/calculate_hash.py "your text here" md5
```

**Examples:**

```bash
# SHA-1 (default)
python3 scripts/calculate_hash.py "hello"

# SHA-256
python3 scripts/calculate_hash.py "hello" sha256

# SHA-512
python3 scripts/calculate_hash.py "hello" sha512

# MD5
python3 scripts/calculate_hash.py "hello" md5
```

**Output:**

```json
{
  "algorithm": "sha1",
  "result": "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d"
}
```

**Details:** See [calculate-hash SKILL.md](./skills/calculate-hash/SKILL.md)

---

### Code Review QA

Provides a structured quality assurance checklist for frontend code reviews,
with emphasis on TypeScript/Angular safety, accessibility, performance,
and maintainability.

**Location:** `skills/code-review-qa/`

**Focus Areas:**

- Null/undefined safety and type correctness
- Error handling and resource cleanup
- Accessibility in HTML/CSS templates
- Performance bottlenecks and memory leaks
- Dead code, duplication, and architecture concerns

**How To Use:**

- Ask for a code review on frontend files
- Request a QA-style checklist-based review before merge

**Examples:**

```text
Review this Angular component for null safety and memory leaks.
Perform a QA code review for this TypeScript, HTML, and SCSS change.
Check this frontend PR for accessibility and performance issues.
```

**Details:** See [code-review-qa SKILL.md](./skills/code-review-qa/SKILL.md)

---

### Frontend Development

Guides Angular/CSS implementation work, including design translation,
responsive behavior, and reusable component patterns.

**Location:** `skills/frontend-development/`

**What It Helps With:**

- Building or modifying Angular UI components
- Translating design artifacts into implementation
- Keeping style and structure aligned with project conventions
- Validating responsiveness and accessibility

**How To Use:**

- Ask for implementation help on UI features
- Ask for design-to-code support with consistency checks

**Examples:**

```text
Implement this UI design as an Angular component.
Refactor this page section into a reusable frontend component.
Update this feature and ensure responsive behavior on mobile and desktop.
```

**Details:** See [frontend-development SKILL.md](./skills/frontend-development/SKILL.md)

---

### Tech Jokes

Generates a random tech joke from a built-in collection.

**Location:** `skills/tech-jokes/scripts/`

**Usage:**

```bash
cd skills/tech-jokes/scripts
python3 tech_jokes.py
```

**Output:**

```json
{
  "text": "Why do programmers prefer dark mode? Because light attracts bugs!"
}
```

**Details:** See [tech-jokes SKILL.md](./skills/tech-jokes/SKILL.md)

---

### Encode Decode

Performs encoding and decoding operations on text using base64 or URL encoding.

**Location:** `skills/encode-decode/scripts/`

**Supported Operations:**

- `base64_encode` - Encode text to base64
- `base64_decode` - Decode base64 text
- `url_encode` - URL encode text
- `url_decode` - URL decode text

**Usage:**

```bash
cd skills/encode-decode/scripts
python3 encode_decode.py <operation> "<text>"
```

**Examples:**

```bash
# Base64 encode
python3 encode_decode.py base64_encode "hello"

# Base64 decode
python3 encode_decode.py base64_decode "aGVsbG8="

# URL encode
python3 encode_decode.py url_encode "hello world"

# URL decode
python3 encode_decode.py url_decode "hello%20world"
```

**Output:**

```json
{
  "status": "success",
  "operation": "base64_encode",
  "result": "aGVsbG8="
}
```

**Details:** See [encode-decode SKILL.md](./skills/encode-decode/SKILL.md)

---

### Historical Facts

Provides concise historical facts from a range of eras and categories,
including ancient history, medieval history, modern history, science,
culture, and exploration.

**Location:** `skills/historical-facts/`

**Categories:**

- Ancient History
- Medieval Period
- Renaissance & Early Modern
- Industrial Era
- Modern History
- Science & Technology
- Culture & Society
- Geography & Exploration

**How To Use:**

- Ask for a historical fact directly in the prompt
- Optionally name a time period or topic to narrow the result

**Examples:**

```text
Tell me a historical fact.
Give me a random historical fact about ancient civilizations.
Share a fact about scientific discoveries.
```

**Details:** See [historical-facts SKILL.md](./skills/historical-facts/SKILL.md)

---

### Inspect File

Analyzes a file and returns metadata, structure, and a preview.

**Location:** `skills/inspect-file/scripts/`

**Usage:**

```bash
python3 skills/inspect-file/scripts/inspect_file.py <file>
```

**Examples:**

```bash
# Inspect a Python file
python3 skills/inspect-file/scripts/inspect_file.py app.py

# Inspect a markdown file
python3 skills/inspect-file/scripts/inspect_file.py README.md

# Inspect a JSON file
python3 skills/inspect-file/scripts/inspect_file.py config.json
```

**Output:**

```json
{
  "status": "success",
  "file": "app.py",
  "size_bytes": 2048,
  "line_count": 87,
  "language": "python",
  "preview": "#!/usr/bin/env python3\n\nimport sys\nimport json\nfrom pathlib import Path"
}
```

**Supported File Types:**

- Python (`.py`)
- JavaScript (`.js`)
- TypeScript (`.ts`)
- Markdown (`.md`)
- JSON (`.json`)
- Java (`.java`)
- Rust (`.rs`)
- Plain text (`.txt`)

**Details:** See [inspect-file SKILL.md](./skills/inspect-file/SKILL.md)

---

### Lint Markdown

Formats and normalizes markdown files to match the repository documentation
philosophy.

**Location:** `skills/lint-markdown/`

**What It Checks:**

- Heading hierarchy and heading spacing
- Consistent `-` list markers
- Clean paragraph spacing and section separation
- Sentence-ending hard-break formatting
- Blank line between a heading and the first bullet list item

**How To Use:**

- Ask the agent to format a markdown file
- Ask the agent to clean up a README without changing its meaning
- Ask the agent to align documentation with the Photon style

**Examples:**

```text
Format this markdown file to match the Photon style.
Clean up this README and fix heading levels.
Normalize this markdown without rewriting the content.
```

**Details:** See [lint-markdown SKILL.md](./skills/lint-markdown/SKILL.md)

---

## How To Invoke Skills In Prompts

You can invoke a skill in two reliable ways:

- Mention the skill name directly
- Describe the exact task that maps to the skill behavior

When possible, include input text, file paths, and expected output format.
This improves routing and gives more consistent results.

### Prompt Patterns

Use these patterns:

```text
Use the <skill-name> skill to <task>.
```

```text
Use the skill at skills/<skill-name>/SKILL.md and <task>.
```

### Examples

```text
Use calculate-hash to compute sha256 for: hello world

Use encode-decode to base64_decode this string: aGVsbG8=

Use inspect-file to analyze: consumerLens/main.py

Use lint-markdown to normalize photons/README.md without changing meaning

Use code-review-qa to review this Angular component for null safety and memory leaks

Use frontend-development to implement this UI as a reusable Angular component

Give me a historical fact from the Industrial Era

Tell me a tech joke
```

### Referencing Rules

- Prefer the exact folder-style skill name, such as `calculate-hash` or `code-review-qa`
- If you need strict behavior, reference the skill spec file directly, for example `skills/lint-markdown/SKILL.md`
- For file-focused tasks, include the target path in the same prompt

---

## Project Structure

```text
photons/
├── README.md
└── skills/
    ├── calculate-hash/
    │   ├── SKILL.md
    │   └── scripts/
    │       └── calculate_hash.py
    ├── code-review-qa/
    │   ├── SKILL.md
    │   └── references/
    │       ├── a11y-checklist.md
    │       ├── angular-best-practices.md
    │       ├── eslint-rules.md
    │       ├── memory-leaks.md
    │       ├── performance-guide.md
    │       └── typescript-pitfalls.md
    ├── encode-decode/
    │   ├── SKILL.md
    │   └── scripts/
    │       └── encode_decode.py
    ├── frontend-development/
    │   ├── SKILL.md
    │   └── references/
    │       ├── angular-style-guide.md
    │       ├── component-examples.md
    │       ├── responsive-checklist.md
    │       └── scss-patterns.md
    ├── historical-facts/
    │   └── SKILL.md
    ├── inspect-file/
    │   ├── SKILL.md
    │   └── scripts/
    │       └── inspect_file.py
    ├── lint-markdown/
    │   └── SKILL.md
    └── tech-jokes/
        ├── SKILL.md
        └── scripts/
            └── tech_jokes.py
```
