# Claude Code in Action — Course Notes

## Table of Content

- [What is a Coding Assistant?](#1-what-is-a-coding-assistant)
- [What is Claude Code?](#2-what-is-claude-code)
- [What is `CLAUDE.md` and Why Does It Matter?](#3-what-is-claudemd-and-why-does-it-matter)
- [Tools in Claude Code](#4-tools-in-claude-code)
- [Adding Context to the Claude Code](#5-adding-context-to-claude-code)
- [Controlling Context](#6-controlling-context)
- [Custom Commands](#7-custom-commands)

---

## 1. What is a Coding Assistant?

A **coding assistant** is a software tool that uses Artificial Intelligence (AI) to help developers write, debug, refactor, and optimize code more efficiently and accurately.  
It acts as a **virtual pair programmer** — always available, never tired, and fluent in virtually every programming language.  

### Types of Coding Assistants

| Type            | Description                                                                               | Examples                            |
|-----------------|-------------------------------------------------------------------------------------------|-------------------------------------|
| **Rule-based**  | Follows predefined rules; used for syntax highlighting, formatting, basic error detection | Traditional IDE linters             |
| **AI-based**    | Learns from coding patterns, predicts intent, generates code from natural language        | GitHub Copilot, Cursor, Claude Code |

### Core Capabilities

- **Code Completion & Suggestions** — Predicts and auto-completes code based on context
- **Code Generation** — Generates entire functions or modules from natural language prompts
- **Debugging Assistance** — Detects errors, identifies bugs, and suggests fixes
- **Refactoring** — Suggests cleaner, more efficient implementations
- **Documentation Generation** — Automatically writes inline comments and docs
- **Multi-language Support** — Works across different programming languages and frameworks
- **Security Scanning** — Flags vulnerabilities and suggests patches

### How They Work (Under the Hood)

- Powered by **Large Language Models (LLMs)** trained on massive code repositories (e.g., public GitHub repos)
- Training follows a two-step process:
  1. **Pre-training** — Model learns structure of natural language and code from a vast dataset
  2. **Fine-tuning** — Model is refined on a narrower, specialized dataset for improved accuracy

### Key Benefit

> Coding assistants are **not here to replace developers** — they augment developer capabilities, reduce time on repetitive tasks, and let developers focus on **problem-solving and innovation**.

### Popular Coding Assistants (2026)

- **GitHub Copilot** — IDE-integrated, trained on public GitHub repos
- **Cursor** — AI-native code editor with multi-file editing
- **Claude Code** — Agentic, terminal-native, multi-file autonomous agent *(covered below)*
- **Tabnine**, **Amazon Q**, **Gemini Code Assist**

## 2. What is Claude Code?

**Claude Code** is Anthropic's **agentic coding tool** that lives in your terminal, understands your entire codebase, and autonomously executes complex development tasks through natural language commands.

> *"It's not just another coding assistant — it's a development partner that understands context, maintains state, and can work autonomously on complex tasks."*

### What Makes It "Agentic"?

Unlike traditional coding assistants that passively suggest code, Claude Code **acts as an autonomous agent**. It can:

- Plan and execute **multi-step tasks** end-to-end
- **Read, write, and edit** files across an entire project
- **Run terminal commands**, tests, and scripts
- **Iterate and self-correct** based on results — without constant human input

### Key Differentiators vs. Other Assistants

| Feature         | Traditional Assistants (Copilot, etc.) | Claude Code                      |
|-----------------|----------------------------------------|----------------------------------|
| **Scope**       | Single file / open tab                 | Entire codebase                  |
| **Interaction** | Inline suggestions                     | Natural language in terminal     |
| **Autonomy**    | Passive (suggests)                     | Active (executes)                |
| **Workflow**    | IDE-dependent                          | Editor-agnostic, terminal-native |
| **Integration** | Plugin-based                           | Unix composable, CI/CD pipelines |

### Where You Can Use Claude Code

- **Terminal (CLI)** — Primary interface; full-featured command-line experience
- **VS Code / JetBrains / Cursor / Windsurf** — IDE extensions with visual diffs
- **Web (claude.ai/code)** — Browser-based, no local setup needed
- **Claude Desktop App** — Quick access for parallel or on-the-go work
- **GitHub / GitLab** — Tag `@claude` directly on issues and PRs

### Core Use Cases

- Building new features end-to-end
- Bug fixing across multiple files
- Writing and running tests
- Git workflows (reading issues, writing code, submitting PRs)
- Codebase exploration and documentation
- CI/CD automation (pipe logs, trigger reviews, translate strings)
- Code review and security vulnerability detection

### Architecture (How Claude Code Works Internally)

When given a task, Claude Code doesn't make a single API call — it **orchestrates multiple specialized sub-agents**:

```sh
Your Request
     ↓
Main Agent (Orchestrator)
     ↓
┌──────────────────────────────────────┐
│ Explore   │ Analyze    │ Investigate │
│ codebase  │ modules    │ dependencies│
└──────────────────────────────────────┘
     ↓
Implements changes → Runs tests → Iterates
```

### Key Technical Details

- Built using **Bun** (compilation), **CommanderJS** (CLI structure), **React Ink** (terminal rendering)
- Follows the **Unix philosophy** — composable, pipeable, minimal
- Integrates with **MCP (Model Context Protocol)** for extending capabilities
- Supports **multi-agent parallelism** — spawn multiple agents working simultaneously
- Can chain an average of **21.2 independent tool calls** without human intervention

### Access & Pricing

| Plan                    | Use Case                               |
|-------------------------|----------------------------------------|
| Claude Pro ($20/mo)     | Medium-high coding workload            |
| Claude Max5 ($100/mo)   | Intense workload, Opus access          |
| Claude Max20 ($200/mo)  | Near-autonomous, heavy multi-agent dev |
| Anthropic Console (API) | Pay-as-you-go API access               |

### Important Config File — `CLAUDE.md`

> The better you document your project in `CLAUDE.md` files, the better Claude Code performs. It uses these files to understand architecture, conventions, and constraints before taking action.

---

## 3. What is `CLAUDE.md` and Why Does It Matter?

### The Core Problem It Solves

Claude Code has **no memory between sessions**.  
Every session starts fresh — it doesn't know your code style, your test commands, your branch naming conventions, or your architectural quirks.  
Without `CLAUDE.md`, you'd repeat the same context every single time.  

> *Think of it as a briefing document for a new team member who has amnesia — one file, loaded before every conversation.*

### What is `CLAUDE.md`?

`CLAUDE.md` is a configuration file that Claude automatically incorporates into every conversation, ensuring it always knows your project structure, coding standards, and preferred workflows.  
It becomes part of Claude's **system prompt**, meaning it is loaded *before* any user message in every session.  

### Where Does It Live? (The Hierarchy)

```sh
~/.claude/CLAUDE.md              ← Global: applies to ALL your projects
your-project/
├── CLAUDE.md                    ← Project-level: loaded at session start
├── .claude/
│   ├── settings.json            ← Permissions, model, hooks config
│   ├── rules/                   ← Modular, path-scoped instruction files
│   │   ├── api-rules.md
│   │   └── testing-rules.md
│   └── .mcp.json                ← External tool integrations (MCP)
└── src/
    └── api/
        └── CLAUDE.md            ← Subdirectory-level: loaded only when working in /api
```

**Priority Rule:** More specific (nested) files override broader ones.  
Subdirectory-level `CLAUDE.md` files are NOT loaded at session start — they are only included when Claude is actively working in that part of the codebase.  

### How to Create One

- **Manually** — Write it yourself in Markdown
- **`/init` command** — Run it in your project directory and Claude generates a starter `CLAUDE.md` based on your project structure and detected tech stack. Then prune what isn't needed.

### What to Put In It

This file can document common bash commands, core utilities, code style guidelines, testing instructions, repository conventions, developer environment setup, and project-specific warnings.

A well-structured example:

```markdown
## Commands
- `npm run dev`: Start development server (port 3000)
- `npm run test`: Run Jest tests
- `npm run lint`: ESLint check

## Code Style
- TypeScript strict mode, no `any` types
- Use named exports, not default exports
- CSS: Tailwind utility classes only

## Architecture
- `/app`: Next.js App Router pages
- `/components/ui`: Reusable UI components
- `/lib`: Shared utilities

## Important Rules
- NEVER commit .env files
- Always validate Stripe webhook signatures
```

### Instruction Priority — A Critical Nuance

Claude Code has a strict instruction hierarchy where `CLAUDE.md` content is treated as authoritative system rules, while user prompts are interpreted as flexible requests that must work within those established rules.

However, there is an important caveat:
Claude Code injects the following system reminder with your `CLAUDE.md` file: *"this context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant to your task."* As a result, Claude will ignore the contents of your `CLAUDE.md` if it decides that it is not relevant to its current task.

This means — **keep it lean and universally applicable**.

### Best Practices

| Do                                        | Don't                                     |
|-------------------------------------------|-------------------------------------------|
| Include commands you run repeatedly       | Include sensitive API keys or credentials |
| Document architectural decisions          | Include task-specific instructions        |
| Add team conventions (branching, commits) | Pad with obvious or auto-detectable info  |
| Keep it under ~300 lines                  | Write a novel — context is precious       |
| Use clear Markdown sections               | Mix unrelated rules together              |

### The "Less is More" Principle

As instruction count increases, instruction-following quality decreases uniformly. This means that as you give the LLM more instructions, it doesn't simply ignore newer instructions — it begins to ignore **all of them** uniformly.

> **Rule of thumb:** Every line in your `CLAUDE.md` competes for attention with the actual work. Only include what is *universally applicable* to your project tasks.

### Auto Memory (Beyond `CLAUDE.md`)

Claude also builds auto memory as it works, saving learnings like build commands and debugging insights across sessions without you writing anything.  
This complements `CLAUDE.md` — you don't need to manually document everything you discover.  

---

## 4. Tools in Claude Code

### The Core Problem Tools Solve

Language models are fundamentally **text-in, text-out** machines.  
On their own, they cannot read files, run commands, browse the web, or interact with external systems.  
**Tools** are the bridge that turns a language model into an actual coding agent.  

> *Without tools, Claude Code would just be a chatbot. With tools, it becomes an autonomous developer.*

### What is a Tool?

A **tool** is a defined capability that Claude can invoke to perform a real-world action outside of text generation.  
When Claude needs to do something it can't do with text alone, it requests a tool call — and the system executes it on Claude's behalf.  

### How the Tool Use System Works

```sh
User Request
     ↓
Claude receives task
     ↓
Claude formulates a formatted tool request
(e.g., "read_file: src/index.ts")
     ↓
The assistant/system executes the actual action
     ↓
Result is sent back to Claude as text
     ↓
Claude uses result to continue reasoning / take next action
```

Claude never directly reads your disk or runs a command — it *asks* for it, and the system does it.  
This loop can repeat many times per task (Claude Code averages **21.2 tool calls** per autonomous task).  

### Default Tools in Claude Code

Out of the box, Claude Code ships with tools covering core development operations:

| Tool Category              | What It Does                           |
|----------------------------|----------------------------------------|
| **File Reading**           | Read file contents, list directories   |
| **File Writing / Editing** | Create, modify, delete files           |
| **Command Execution**      | Run shell commands, scripts, tests     |
| **Search (grep)**          | Search codebases for patterns, symbols |
| **Git Operations**         | Stage, commit, read diffs, create PRs  |

### Why Claude's Tool Use is Superior

Not all models use tools equally well. Claude's key advantages:

- **Better at understanding tool function** — knows when and how to use each tool appropriately
- **Better at combining tools** — chains multiple tool calls intelligently to solve complex tasks
- **Security advantage** — Claude Code searches code directly rather than indexing and sending your codebase to external servers
- **Extensibility** — new tools can be added easily without rebuilding anything

### Extending Tools — MCP Servers

Claude Code is designed to be extended.  
When default tools aren't enough, you can plug in **MCP (Model Context Protocol) servers** — external tool packages that run locally or remotely.  

**Example — Playwright MCP Server:**

```sh
claude mcp add playwright npx @playwright/mcp
```

Once added, Claude gains full browser automation capabilities — visiting pages, taking screenshots, interacting with UI elements — all as tool calls.  

**Permission Management:**

- First use of any new MCP tool requires manual approval
- Auto-approve by adding `"MCP__[servername]"` to `settings.local.json` allow array

### Tools and Hooks — Controlling Tool Execution

Since every Claude action is a tool call, you can intercept them using **Hooks**:

| Hook Type         | When It Runs         | Can Block?          |
|-------------------|----------------------|---------------------|
| **Pre-tool use**  | Before tool executes | Yes (exit code 2)   |
| **Post-tool use** | After tool executes  | No                  |

This gives you fine-grained control — e.g., blocking Claude from reading `.env` files, auto-formatting after file edits, or running type checks after changes.

### Tools in the SDK

The Claude Code SDK exposes the same tool system programmatically:

- **Default** — read-only tools (file reading, grep, directory listing)
- **Write tools** — must be explicitly enabled via `options.allowTools` array (e.g., `"edit"`)

This is useful for embedding Claude's tool-powered intelligence into your own scripts, hooks, or pipelines.

---

## 5. Adding Context to Claude Code

### Why Context Matters

Claude Code starts every session with **zero memory** of previous conversations.  
It doesn't know your project structure, your coding conventions, your team's rules, or even how to run your tests — unless you tell it.  

Beyond the amnesia problem, there's a subtler issue: **too much irrelevant context is just as bad as too little.**  
Flooding Claude with unnecessary information actively degrades its performance. The goal is always *just enough* relevant context.  

> *The quality of Claude Code's output is directly proportional to the quality of context you provide.*

### The Three Ways to Add Context

#### 1. `CLAUDE.md` — Persistent, Automatic Context

This is the primary context mechanism. Claude automatically loads `CLAUDE.md` into every session as part of its system prompt — before any user message. You write it once, and Claude always knows it.

**Three levels of `CLAUDE.md`:**

| Level       | Location                         | Scope                            | Committed to Git? |
|-------------|----------------------------------|----------------------------------|-------------------|
| **Machine** | `~/.claude/CLAUDE.md`            | All projects on your machine     | No                |
| **Project** | `your-project/CLAUDE.md`         | Entire project, shared with team | Yes               |
| **Local**   | `your-project/.claude/CLAUDE.md` | Project, personal only           | No                |

**What to put in it:**

```md
## Commands
- `npm run dev` — start dev server on port 3000
- `npm test` — run Jest test suite
- `npm run lint` — ESLint check

## Architecture
- `/app` — Next.js App Router pages
- `/components/ui` — reusable UI components
- `/lib` — shared utilities and helpers

## Code Conventions
- TypeScript strict mode, no `any` types
- Named exports only, no default exports
- Tailwind for all styling — no custom CSS

## Rules
- NEVER commit `.env` files
- Always validate API inputs with Zod
```

**How to create it:**

- **Manually** — write it yourself in Markdown
- **`/init` command** — run it in your project root and Claude auto-generates a starter file based on your codebase. Then prune what isn't needed.

**How to update it mid-session:**

Use the `#` shortcut to add rules to `CLAUDE.md` using natural language without leaving the conversation:

```md
# always use named exports, never default exports
```

Claude intelligently updates the file for you.

#### 2. `@` Symbol — Targeted, On-Demand Context

When you need Claude to focus on a specific file for a specific task, use `@` to mention it directly in your prompt:

```md
@src/db/schema.ts add a new orders table with a foreign key to users
```

This avoids Claude searching broadly through your codebase and pins its attention exactly where needed. Best for precise, file-specific tasks.

#### 3. Reference Critical Files in `CLAUDE.md`

For files Claude should *always* be aware of (like a database schema, API contract, or type definitions), reference them directly inside `CLAUDE.md`:

```md
## Key Files
- Always read `src/db/schema.ts` before making any database changes
- Refer to `src/types/global.d.ts` for shared type definitions
```

This combines the persistence of `CLAUDE.md` with the precision of file referencing.

### The "Less is More" Principle — Critical for Exam

As the number of instructions grows, Claude's ability to follow **all** of them degrades uniformly.  
It doesn't ignore new instructions — it starts partially ignoring everything.  

**Practical rules:**

- Keep `CLAUDE.md` under ~300 lines
- Only include rules that apply to **every task** in the project
- Remove anything Claude could reasonably infer on its own
- Prune regularly as the project evolves

### Context Summary

| Tool                  | When to Use                                   | Persistence                      |
|-----------------------|-----------------------------------------------|----------------------------------|
| `CLAUDE.md` (Project) | Team-wide conventions, architecture, commands | Every session, committed to git  |
| `CLAUDE.md` (Local)   | Personal preferences, local paths             | Every session, not committed     |
| `CLAUDE.md` (Machine) | Global rules across all projects              | Every session, global            |
| `@` file mention      | Task-specific file focus                      | Current prompt only              |
| `#` shortcut          | Quick mid-session rule updates                | Saved to `CLAUDE.md` permanently |

---

## 6. Controlling Context

Context is precious — too much irrelevant information actively hurts Claude's performance.

### Context Control Techniques

| Technique           | How                  | When to Use                                               |
|---------------------|----------------------|-----------------------------------------------------------|
| **Escape**          | Press `Esc` once     | Stop Claude mid-response to redirect                      |
| **Escape + Memory** | `Esc` → `#` shortcut | Stop + save a lesson to prevent repeated mistakes         |
| **Double Escape**   | Press `Esc` twice    | Rewind conversation to an earlier point                   |
| **`/compact`**      | Type `/compact`      | Summarize history but preserve Claude's learned knowledge |
| **`/clear`**        | Type `/clear`        | Full reset — delete entire conversation history           |

### When to Use Each

- **`/compact`** — Long sessions where Claude has built expertise but the conversation is cluttered
- **`/clear`** — Switching to a completely unrelated task
- **Double Escape** — Skip irrelevant debugging back-and-forth without losing context
- **Escape + Memory** — When Claude keeps making the same mistake; stop it and teach it

---

## 7. Custom Commands

### What Are They?

Custom Commands are **user-defined slash commands** that automate repetitive Claude Code tasks — like running a security audit, generating tests, or fixing vulnerabilities.

### How to Create Custom Command

1. Create a file in `.claude/commands/` directory
2. Name the file — the filename becomes the command name
   - `audit.md` → `/audit`
3. Write instructions in Markdown inside the file
4. Restart Claude Code to activate

### Using Arguments

Use the `$arguments` placeholder in your command file to accept dynamic input at runtime:

```md
## Command: /audit
Run a full security audit on the following file: $arguments
Check for: SQL injection, XSS, insecure dependencies
```

```md
/audit src/api/users.ts
```

### Key Details

- Arguments can be any string — file paths, descriptions, identifiers
- Great for: dependency auditing, test generation, vulnerability fixing, code reviews

---

## Summary

|               | Coding Assistant                 | Claude Code                        |
|---------------|----------------------------------|------------------------------------|
| **What**      | AI tool to help write/debug code | Agentic terminal tool by Anthropic |
| **How**       | LLM-powered suggestions in IDE   | Autonomous agent in your terminal  |
| **Scope**     | File/function level              | Full codebase + tools + pipelines  |
| **Key Skill** | Code completion                  | End-to-end task execution          |

---
