# Prompt Library

A curated collection of reusable prompts for AI-assisted code analysis,
review, and improvement tasks.

## Contents

### Code Summarizer

Located in `code-summarizer/`:

- **code_summarizer_base.md** — Summarizes code from the perspective of a
  senior backend engineer. Outputs JSON with purpose, components, and data
  flow. Flags visible bugs or risks. Max 120 words.
- **code_summarizer_explained.md** — Mentoring-style explanation aimed at
  junior developers. More detailed and avoids jargon. Outputs JSON with an
  additional notes field.
- **code_summarizer_strict.md** — Strictly factual summary with zero
  inference. Modelled as a static analysis tool. Max 100 words. Returns only
  what is explicitly visible in the code.

### Code Reviewer

Located in `code-reviewer/`:

- **code_reviewer_base.md** — Strict code review from a senior backend
  engineer perspective. Reports bugs, logical errors, performance issues,
  security risks, and poor design choices. Outputs a JSON issues list with
  type, description, impact, and confidence.
- **code_reviewer_security.md** — Security-focused review from a senior
  security engineer perspective. Covers input validation, auth flaws, insecure
  data handling, injection vulnerabilities, and information leakage. Outputs
  a JSON vulnerabilities list with severity and confidence levels.

### Code Refactor Advisor

Located in `code-refactor/`:

- **code_refactor_advisor.md** — Suggests targeted improvements to existing
  code. Focuses on readability, performance, and maintainability. Outputs a
  JSON improvements list with problem, suggestion, and benefit per item.

### Test Case Generator

Located in `test-case-generator/`:

- **test_case_generator.md** — Generates meaningful test cases from a QA
  engineer perspective. Covers edge cases, failure scenarios, and boundary
  conditions. Outputs a JSON test cases list with scenario, input, and
  expected output.

## Demo Outputs

Sample prompt outputs are available in `demo/` organized by prompt and date.
