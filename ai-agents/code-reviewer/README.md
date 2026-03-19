# Code Reviewer Agent

A CLI-based agentic AI application that autonomously reviews source code for bugs, security vulnerabilities, and code quality issues — powered by Claude.

---

## Table of Contents

- [What Is an Agent?](#what-is-an-agent)
- [Core Concepts](#core-concepts)
  - [Tools](#tools)
  - [Roles](#roles)
  - [The Agentic Loop](#the-agentic-loop)
- [How Tool Calling Works](#how-tool-calling-works)
- [Project Structure](#project-structure)
- [File Responsibilities](#file-responsibilities)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Sample Output](#sample-output)
- [Extending This Agent](#extending-this-agent)

---

## What Is an Agent?

An **agent** is an application that:

- Connects to a local or remote LLM
- Equips the LLM with tools to extend its capabilities beyond text generation
- Uses a system prompt to define its role, output format, and behavior
- Runs an autonomous loop — taking multiple steps to complete a goal without human intervention at each step

A single API call to an LLM is a prompt. An agent is a prompt plus a loop plus tools plus a stopping condition.

```sh
Agent = LLM connection
      + tools (extended capabilities)
      + system prompt (role, format, behavior)
      + agentic loop (autonomy over multiple steps)
      + stopping condition (knows when it is done)
```

Remove any one of these five and you have something less than an agent.

---

## Core Concepts

### Tools

A **tool** is a function that an LLM can call during its reasoning process. Instead of generating text alone, the model can pause, invoke a tool, receive a real result, and continue reasoning with that new information.

Without tools, an LLM is sealed inside its training data and whatever text you give it. Tools are the escape hatch into the real world.

There are three layers to understand:

| Layer | What It Is | How It Works |
|---|---|---|
| **Tool Definition** | A JSON description of the tool | Sent to Claude on every request so it knows what tools exist |
| **Tool Implementation** | A local Python function | Runs on your machine — Claude never executes this directly |
| **Tool Dispatcher** | A routing function | Receives Claude's tool call and invokes the correct implementation |

**Critical point — the LLM never executes tools directly.** Claude reads the tool descriptions, decides which tool to call and with what arguments, and returns a structured request. Your application executes the actual function locally and sends the result back. Claude only ever sees the result.

### Roles

A **role** is the identity given to an agent through its system prompt. It defines:

- What the agent is responsible for
- What it should and should not do
- How it should think and reason
- What its output should look like

Roles fall into natural categories:

- **Planning** — architect, planner, requirements analyst
- **Execution** — coder, refactorer, test writer
- **Review** — code reviewer, security reviewer, coverage analyzer
- **Repair** — build error resolver, debugger, dependency fixer
- **Documentation** — doc updater, changelog writer, codemap generator
- **Domain-specific** — database agent, DevOps agent, ML agent

A good role has three properties: it is clearly scoped, its tools match what it is allowed to do, and it has a well-defined output that signals it is done.

### The Agentic Loop

The agentic loop is the mechanism that gives an agent autonomy. It is a `while` loop that keeps sending messages to the LLM, handling tool calls when requested, and only exits when the LLM signals it is done.

```sh
┌─────────────────────────────────────┐
│         Send messages to Claude     │
└──────────────────┬──────────────────┘
                   │
         ┌─────────▼──────────┐
         │  Tool calls in     │ YES → Execute tools
         │  response?         │      Append results
         └─────────┬──────────┘      Loop again
                   │ NO
         ┌─────────▼──────────┐
         │  stop_reason =     │ YES → Parse output
         │  "end_turn"?       │      Print report
         └────────────────────┘      Exit
```

Each iteration is one round trip to the LLM. The conversation history grows with every turn — Claude's tool requests, your tool results, and Claude's reasoning — because the LLM is stateless and must receive the full context on every call.

---

## How Tool Calling Works

This is the complete mechanics of a single tool call exchange:

### Step 1 — You send a request with tool definitions

```python
response = client.messages.create(
    model    = "claude-sonnet-4-20250514",
    system   = SYSTEM_PROMPT,
    tools    = TOOL_DEFINITIONS,      # descriptions only — not implementations
    messages = messages
)
```

### Step 2 — Claude responds with a tool_use block

```json
{
  "type": "tool_use",
  "id":   "toolu_01XyzAbc",
  "name": "read_file",
  "input": { "path": "./src/auth.py" }
}
```

Claude has decided which tool to call, with what arguments, and why — based entirely on the tool descriptions you provided.

### Step 3 — You execute the tool locally and return the result

```python
result = execute_tool(block.name, block.input)

messages.append({
    "role": "user",
    "content": [{
        "type":        "tool_result",
        "tool_use_id": block.id,      # must match Claude's request id
        "content":     result
    }]
})
```

### Step 4 — Loop continues until Claude stops calling tools

The `tool_use_id` is the matching mechanism — it tells Claude which result belongs to which request, especially when multiple tools are called in a single response.

---

## Project Structure

```sh
code-reviewer/
├── config.py         # Model, max_tokens, and system prompt
├── tools.py          # Tool definitions, implementations, and registry
├── agent.py          # Agentic loop and conversation management
├── reporter.py       # Output parser and terminal printer
├── .env              # ANTHROPIC_API_KEY (never commit this)
├── .env.example      # Safe template to share
└── requirements.txt  # anthropic, python-dotenv
```

---

## File Responsibilities

### `config.py`

The only file that needs to change when building a new agent. Contains the model name, token limit, and system prompt. The system prompt defines the agent's role, workflow, and output format contract.

### `tools.py`

Three concerns in one file:

- `TOOL_DEFINITIONS` — JSON descriptions sent to Claude
- Implementations — the actual Python functions (`read_file`, `list_files`, `search_code`)
- `TOOL_REGISTRY` — maps tool names to implementations; the dispatcher routes through this

Adding a new tool is three steps: add its definition, implement the function, register it in `TOOL_REGISTRY`.

### `agent.py`

The generic agentic loop. Connects to Claude, manages conversation history, dispatches tool calls via `execute_tool()`, and hands the final output to `reporter.py`. This file is intentionally domain-agnostic — it knows nothing about code review specifically.

### `reporter.py`

Two concerns:

- `parse()` — converts Claude's raw text into a structured `ReviewReport` dataclass
- `print_report()` — renders the report to the terminal with ANSI colors and severity grouping

The parser relies on the output format defined in `config.py`'s system prompt. If the prompt's output structure changes, the parser must be updated to match.

---

## Getting Started

### 1. Clone and enter the project

```bash
git clone <repo-url>
cd code-reviewer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API key

```bash
cp .env.example .env
# Edit .env and add your key from https://console.anthropic.com
```

---

## Usage

### Review a single file

```bash
python agent.py ./src/auth.py
```

### Review an entire directory

```bash
python agent.py ./src/
```

The agent will log every tool call it makes as it works, then print the final report.

---

## Sample Output

```sh
umang.mishra@umangs-MacBook-Pro code-reviewer % python3 agents.py src/
✔  API key looks good.

  Reviewing: src/
  ──────────────────────────────────────────────────
  ⚙  list_files(directory='src/')
  ⚙  read_file(path='src/FileProcessor.java')
  ⚙  read_file(path='src/PaymentService.java')
  ⚙  read_file(path='src/UserService.java')
  ⚙  search_code(directory='src/', pattern='printStackTrace')
  ⚙  search_code(directory='src/', pattern='password')

════════════════════════════════════════════════════
  CODE REVIEW REPORT
════════════════════════════════════════════════════

 ✖  CRITICAL
──────────────────────────────────────────────────
  • FileProcessor.java:10
    Resource leak: BufferedReader is never closed, causing file descriptor leak
  • FileProcessor.java:30
    Array index out of bounds vulnerability: no bounds checking before accessing buffer array
  • PaymentService.java:26
    Security vulnerability: logging sensitive credit card numbers in plain text
  • UserService.java:10
    Security vulnerability: storing passwords in plain text without hashing
  • UserService.java:15
    Null pointer exception risk: calling equals() on potentially null value from map lookup
  • UserService.java:21
    23 — Security vulnerability: method exposes raw passwords to callers
  • UserService.java:28
    Security vulnerability: printing all passwords in plain text

 ⚠  WARNING
──────────────────────────────────────────────────
  • FileProcessor.java:14
    Performance issue: inefficient string concatenation in loop (use StringBuilder)
  • FileProcessor.java:18
    19 — Poor error handling: printing stack trace instead of proper logging/recovery
  • PaymentService.java:6
    8 — Logic bug: invalid card check doesn't prevent processing, only prints message
  • PaymentService.java:11
    14 — Logic flaw: negative payment amounts are accepted as valid refunds
  • PaymentService.java:35
    Logic bug: discount calculation gives higher discount (20%) for lower prices than high prices (10%)
  • UserService.java:8
    10 — Missing input validation: no checks for null, empty, or weak passwords

 ℹ  INFO
──────────────────────────────────────────────────
  • PaymentService.java:17
    19 — Hardcoded test card number should be removed or made configurable
  • UserService.java:6
    Consider using thread-safe ConcurrentHashMap if this service is used in multi-threaded environment

────────────────────────────────────────────────────
  Summary: 7 critical, 6 warning(s), 2 info
════════════════════════════════════════════════════

umang.mishra@umangs-MacBook-Pro code-reviewer %
```

---

## Extending This Agent

This project is designed as a reusable template. The agent loop, reporter, and project structure stay the same across agents. Only the domain-specific parts change.

**To build a different agent:**

1. Update `config.py` — change the system prompt to define a new role and output format
2. Update `tools.py` — replace or add tools relevant to the new domain
3. Update `reporter.py` — adjust the parser if the output structure changes

**To add a new tool to this agent:**

In `tools.py`, follow the three-step pattern:

```python
# Step 1 — Add to TOOL_DEFINITIONS
{
    "name": "your_tool",
    "description": "What it does and when Claude should use it.",
    "input_schema": { ... }
}

# Step 2 — Implement the function
def your_tool(arg: str) -> str:
    ...

# Step 3 — Register it
TOOL_REGISTRY = {
    ...
    "your_tool": lambda inputs: your_tool(inputs["arg"]),
}
```

Nothing else needs to change.
