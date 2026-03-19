"""
agent.py
────────
The agentic loop. Connects to Claude, manages conversation history,
dispatches tool calls, and delivers the final report.

This file is intentionally generic — it knows nothing about what the
agent does. All domain-specific logic lives in config.py and tools.py.
"""

import os
import sys
import anthropic

from dotenv import load_dotenv
from config import MODEL, MAX_TOKENS, SYSTEM_PROMPT
from tools  import TOOL_DEFINITIONS, execute_tool
from reporter import parse, print_report

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv(override=True)
_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


def _validate_api_key() -> bool:
    """Validate the API key format and print a diagnostic message."""
    key = _API_KEY.strip()

    if not key:
        print("✖  No API key found. Set ANTHROPIC_API_KEY in your .env file.")
        return False
    if not key.startswith("sk-ant-"):
        print("✖  API key found but format looks wrong (expected 'sk-ant-...').")
        return False
    if key != _API_KEY:
        print("✖  API key has leading/trailing whitespace — please remove it.")
        return False

    print("✔  API key looks good.")
    return True


# ---------------------------------------------------------------------------
# Agentic Loop
# ---------------------------------------------------------------------------

def run_agent(target_path: str) -> None:
    """
    Run the code review agent against a file or directory.

    The loop sends the conversation to Claude, handles tool calls
    when Claude requests them, and exits when Claude produces a
    final text response with no further tool calls.
    """
    client = anthropic.Anthropic(api_key=_API_KEY)

    # Build the opening user message based on what was passed in
    if os.path.isdir(target_path):
        opening = f"Please review all source code in this directory: {target_path}"
    elif os.path.isfile(target_path):
        opening = f"Please review this file for bugs and issues: {target_path}"
    else:
        print(f"✖  '{target_path}' is not a valid file or directory.")
        sys.exit(1)

    # Full conversation history — rebuilt and resent on every API call
    # because Claude is stateless between requests.
    messages = [{"role": "user", "content": opening}]

    print(f"\n  Reviewing: {target_path}")
    print("  " + "─" * 50)

    # ── Main loop ────────────────────────────────────────────────────────────
    while True:
        response = client.messages.create(
            model     = MODEL,
            max_tokens= MAX_TOKENS,
            system    = SYSTEM_PROMPT,
            tools     = TOOL_DEFINITIONS,
            messages  = messages
        )

        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        # ── Exit condition ───────────────────────────────────────────────────
        # Claude signals it is done with stop_reason="end_turn" and no tool calls.
        if response.stop_reason == "end_turn" and not tool_blocks:
            raw    = "\n".join(b.text for b in text_blocks)
            report = parse(raw)
            print_report(report)
            return

        # ── Tool execution ───────────────────────────────────────────────────
        # Claude wants one or more tools called. Execute each, collect results,
        # then append both Claude's request and our results to the history
        # before looping back.
        if tool_blocks:
            # Preserve Claude's full response (including its tool_use blocks)
            # in history — Claude needs to see its own requests on the next turn.
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in tool_blocks:
                print(f"  ⚙  {block.name}({_format_inputs(block.input)})")
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,     # must match Claude's request id
                    "content":     result
                })

            messages.append({"role": "user", "content": tool_results})
            # Loop continues — Claude reads the results and decides next step.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_inputs(inputs: dict) -> str:
    """Render tool inputs as a compact one-liner for the terminal log."""
    return ", ".join(f"{k}='{v}'" for k, v in inputs.items())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:   python agent.py <file_or_directory>")
        print("Example: python agent.py ./src/auth.py")
        print("Example: python agent.py ./src/")
        sys.exit(1)

    if not _validate_api_key():
        sys.exit(1)

    run_agent(sys.argv[1])
