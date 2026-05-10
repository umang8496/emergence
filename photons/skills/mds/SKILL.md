---
name: mds # markdown stylist
description: The master skill for formatting unstructured Markdown content according to the Emergence repository style guide, ensuring technical compliance, maximum readability, and universal compatibility.
---


# Markdown Stylist (mds) Skill Definition


Use this skill whenever any piece of markdown text needs professional formatting, structural cleanup, normalization, or adherence to strict corporate style guidelines. This is the primary source of truth for all styling decisions in the repository.


## Core Philosophy & Goal
The goal is not merely to write Markdown, but to produce a technically perfect, maximally readable document that maintains 100% fidelity to the original meaning and intent of the content. The resulting markdown must be universally clean and flawless, regardless of the source platform.


## When To Use (Activation Triggers)
Use this skill when the request involves any of the following:

- Formatting an entire markdown file from raw text.
- Cleaning up structural inconsistencies (e.g., inconsistent spacing or list markers).
- Normalizing markdown to adhere to the full suite of defined style rules.
- Fixing syntax issues in mathematical formulas, code blocks, or tables.
- General polishing and cleanup before final review/publishing.


## Primary Style Guides & References
- **[Reference 1]** The overall document structure must be consistent with existing successful examples (e.g., the canonical Photon README).
- **[Reference 2]** All hard rules are dictated by this skill's explicit constraints below.


## Universal Formatting Constraints (Non-Negotiable Rules)

The following nine constraints MUST be followed in every single output:

### I. Structural Logic & Hierarchy
1.  **Heading Flow:** Use the full spectrum of markdown headers (`#` through `######`) dynamically based on content hierarchy. Duplicate titles are permitted and must not be merged or flagged as errors.
2.  **List Separation:** Whenever a list (bulleted or numbered) begins, there **MUST** be an explicit blank newline immediately preceding the first item to separate it from any preceding paragraph text.
3.  **Indentation:** All nested list items must maintain strict indentation using exactly **4 spaces**.

### II. Technical Syntax & Compliance
1.  **Code Blocks:** Code MUST use fenced code blocks and *MUST* include an appropriate language identifier marker immediately after the opening fence (e.g., ` ```python `, ` ```java`).
2.  **Mathematical Expressions:** All formulas, equations, or complex math must be rendered using LaTeX syntax and enclosed by double dollar signs (`$$...$$`) to guarantee correct mathematical rendering across platforms.
3.  **Tables:** Tables require extreme padding and uniform spacing between headers, separators, and all data cells to maximize readability and professionalism.

### III. Aesthetic Polish & Compatibility
1.  **Line Endings (CRITICAL):** Every single line in the final output string **MUST** terminate with exactly two space characters (`  `). This is a non-negotiable technical requirement for compatibility.
2.  **Emphasis:** All intended emphasis must use bold text syntax (**text**) and italics (*text*) where appropriate.
3.  **Quoting & Hidden Sections:** Use the standard markdown blockquote syntax (`>`) for quotes. If content is supplementary, it must be separated by `***` and flagged as optional/internal documentation.


## Supported Issues (Error Handling)

### Errors (Critical Failures to Prevent):
- **Missing Space:** Ensuring there is always a single space after all heading markers (`#`).
- **Hierarchy Skip:** Preventing skips in the logical progression of heading levels (e.g., jumping from `##` to `####`).
- **Code/Syntax Failure:** Never output an unclosed code block, unmatched bracket, or improperly formatted table cell.

### Warnings (Minor Fixes):
- **Spacing:** Correcting excessive multiple consecutive spaces found within a single line.
- **List Marker Consistency:** Ensuring list markers are consistent (preferring `-` over `*`).


## Agent Workflow Logic (How to Think)
When triggered, the agent must follow this thought process:

1.  **Analyze Intent:** Identify the core message and logical structure of the raw text input.
2.  **Map Structure:** Determine the appropriate heading hierarchy (`#`, `##`, etc.) based on the content's flow.
3.  **Enforce Constraints (Pass 1):** Systematically pass the entire text through a filter that enforces all technical syntax rules: adding language tags, wrapping math in `$$...$$`, and ensuring list separation.
4.  **Polish & Format (Pass 2):** Apply aesthetic rules: consistent bolding, maximum table padding, correct indentation, etc.
5.  **Final Check:** Verify the entire resulting output string for the critical two-space line ending rule and global syntactic cleanliness before presenting it to the user.

## Execution Instructions
When asked to format markdown, do not rewrite the content wholesale.  
**Make minimal, targeted edits that fix structural errors, spacing inconsistencies, and syntax violations while preserving 100% of the original meaning and intent.**


## Examples (Activation Phrases)
- "Using the mds skill, please clean up this README."
- "Format the following markdown snippet to adhere to corporate style guidelines."

***
