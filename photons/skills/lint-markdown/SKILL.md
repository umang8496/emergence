---
name: lint-markdown
description: Default markdown formatting skill for cleaning up, normalizing, and aligning markdown files to the repository style.
---

# Markdown Lint Skill

Use this skill whenever a markdown file needs formatting, cleanup,
normalization, or markdown-style correction.

This skill is the default markdown formatting policy for the Emergence
repository.  
When formatting markdown, use the Photon README as the canonical style
example and the repository markdown style guide as the rule reference.  

## When To Use

Use this skill when the request involves any of the following:

- Formatting a markdown file
- Cleaning up markdown structure or spacing
- Fixing heading hierarchy, list consistency, or link formatting
- Normalizing markdown to match the repository documentation philosophy
- Reviewing a markdown file for style issues before publishing

## Primary References

- `photons/README.md` for tone, spacing, section flow, and examples
- `MARKDOWN_STYLE_GUIDE.md` for explicit rules and edge cases

## Markdown Philosophy

- Keep prose clear, direct, and concise
- Use a single top-level title and maintain strict heading progression
- Prefer `-` for unordered lists
- Use blank lines to separate sections and paragraphs
- Keep code blocks fenced and language-tagged
- Use descriptive links rather than bare URLs
- Preserve intentional trailing spaces only when they are required for hard
  line breaks
- Avoid unnecessary decorative spacing, repetition, or noisy markdown constructs

## Repository-Specific Rules

- Use a single `#` title per document
- Never skip heading levels
- Prefer `-` over `*` for unordered lists
- Use fenced code blocks and add a language tag when it is known
- Prefer descriptive markdown links over bare URLs
- Keep visual separators short and intentional rather than using excessive
  horizontal structure
- End a wrapped prose line at the sentence boundary when the sentence ends with
  a full stop and the line uses two trailing spaces for a hard break
- Leave one blank line between a heading and the first bullet that follows it
- Preserve the meaning of the original content while improving presentation

## Supported Rules

### Errors (Critical)

- `heading-spacing` - Missing space after heading marker (`#`)
- `heading-hierarchy` - Heading level skips sequentially
- `missing-image-alt` - Image without alt text
- `unclosed-code-block` - Code block not properly closed
- `unclosed-brackets` - Unmatched brackets or parentheses

### Warnings (Non-Critical)

- `line-length` - Line exceeds 120 characters
- `trailing-whitespace` - Line ends with whitespace
- `multiple-spaces` - Multiple consecutive spaces found
- `unclosed-emphasis` - Unmatched emphasis markers
- `trailing-blank-lines` - Multiple trailing blank lines at end of file
- `duplicate-heading` - Same heading text appears multiple times

### Style Issues

- `list-marker-consistency` - List uses markers other than `-`
- `sentence-break-formatting` - Sentence-ending prose lines should stop at the
  full stop when using two trailing spaces
- `heading-list-spacing` - A heading should have one blank line before the
  first bullet list item beneath it

## Agent Workflow

1. Read the target markdown file.
2. Compare its structure and style against `photons/README.md` and
   `MARKDOWN_STYLE_GUIDE.md`.
3. Make minimal edits that fix structure, spacing, headings, lists, code block
   formatting, sentence-level wrapping, and heading-to-list spacing.
4. Preserve links, code samples, meaning, and any intentional formatting choices.
5. Re-read the changed section and confirm heading flow, sentence breaks,
   spacing, and list consistency.

## Formatting Targets

When asked to format markdown, normalize the file toward these outcomes:

- Consistent heading depth with no skipped levels
- Consistent unordered list markers
- Clean paragraph spacing and section separation
- Properly fenced code blocks with language tags
- Clean links, images, and references
- Sentence-ending hard-break lines stop cleanly at the full stop
- Headings have a blank line before the first bullet list item
- Minimal noise while preserving meaning and intent

## Response Expectations

- Prefer direct file edits over broad rewrites
- Summarize only the meaningful classes of fixes when the user asks for a
  summary
- If the file already matches the repository style, say so instead of inventing
  changes
- If a workspace-local markdown formatter exists, it may be used, but this
  skill remains the source of truth for style decisions

## Examples

- "Format this markdown file to match the Photon style"
- "Clean up this README and fix heading levels"
- "Normalize this markdown without rewriting the content"
- "Make this documentation align with the repository markdown conventions"

## Expected Behavior For The Agent

When a user asks for markdown formatting, this skill should be the one used to
guide the response and the edits.  
Prefer minimal, targeted changes that make the document conform to the
repository style rather than rewriting the file wholesale.  

If the user provides a canonical example document, use it as the primary
reference for tone, spacing, heading flow, and list style.  

This skill is instruction-first.  
It does not depend on a dedicated `lint_markdown.py` script to be useful.
