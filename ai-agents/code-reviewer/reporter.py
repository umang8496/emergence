"""
reporter.py
───────────
Two responsibilities:
  1. parse()        — converts Claude's raw text output into a ReviewReport object.
  2. print_report() — renders a ReviewReport to the terminal with ANSI colors.

The parse() function relies on the output format contract defined in
config.py's SYSTEM_PROMPT. If you change the prompt's output structure,
update the parser here to match.
"""

import re
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

Severity = Literal["CRITICAL", "WARNING", "INFO"]


@dataclass
class Issue:
    file:        str
    line:        int | None
    description: str
    severity:    Severity


@dataclass
class ReviewReport:
    critical: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    info:     list[Issue] = field(default_factory=list)
    raw:      str = ""          # Claude's original output, always preserved

    @property
    def total(self) -> int:
        return len(self.critical) + len(self.warnings) + len(self.info)

    @property
    def has_issues(self) -> bool:
        return self.total > 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Matches:  - auth.py:23 — description
# and:      - auth.py — description   (no line number)
_ISSUE_RE = re.compile(
    r"-\s+(?P<file>[^\s:]+):?(?P<line>\d+)?\s*[—–-]+\s*(?P<desc>.+)"
)

_SECTION_HEADERS: dict[str, Severity] = {
    "CRITICAL": "CRITICAL",
    "WARNING":  "WARNING",
    "INFO":     "INFO",
}


def parse(raw_output: str) -> ReviewReport:
    """Parse Claude's structured text output into a ReviewReport."""
    report           = ReviewReport(raw=raw_output)
    current_severity = None

    for line in raw_output.splitlines():
        stripped = line.strip()

        # Detect section transitions
        if stripped in _SECTION_HEADERS:
            current_severity = _SECTION_HEADERS[stripped]
            continue

        if stripped.startswith(("SUMMARY", "==")):
            current_severity = None
            continue

        # Parse issue lines
        if current_severity and stripped.startswith("-"):
            match = _ISSUE_RE.match(stripped)
            if not match:
                continue

            issue = Issue(
                file        = match.group("file"),
                line        = int(match.group("line")) if match.group("line") else None,
                description = match.group("desc").strip(),
                severity    = current_severity
            )

            if current_severity == "CRITICAL":
                report.critical.append(issue)
            elif current_severity == "WARNING":
                report.warnings.append(issue)
            elif current_severity == "INFO":
                report.info.append(issue)

    return report


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------

# ANSI codes — no external dependencies
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"

_STYLE: dict[Severity, dict] = {
    "CRITICAL": {"color": _RED,    "icon": "✖"},
    "WARNING":  {"color": _YELLOW, "icon": "⚠"},
    "INFO":     {"color": _CYAN,   "icon": "ℹ"},
}


def _print_section(issues: list[Issue], severity: Severity) -> None:
    if not issues:
        return

    color = _STYLE[severity]["color"]
    icon  = _STYLE[severity]["icon"]

    print(f"\n{color}{_BOLD} {icon}  {severity}{_RESET}")
    print(f"{_DIM}{'─' * 50}{_RESET}")

    for issue in issues:
        location = f"{issue.file}:{issue.line}" if issue.line else issue.file
        print(f"  {color}•{_RESET} {_BOLD}{location}{_RESET}")
        print(f"    {issue.description}")


def print_report(report: ReviewReport) -> None:
    """Pretty print a ReviewReport to the terminal."""
    print(f"\n{_BOLD}{'═' * 52}{_RESET}")
    print(f"{_BOLD}  CODE REVIEW REPORT{_RESET}")
    print(f"{_BOLD}{'═' * 52}{_RESET}")

    if not report.has_issues:
        print(f"\n  {_GREEN}✔  No issues found.{_RESET}\n")
        return

    _print_section(report.critical, "CRITICAL")
    _print_section(report.warnings, "WARNING")
    _print_section(report.info,     "INFO")

    # Summary
    parts = []
    if report.critical:
        parts.append(f"{_RED}{len(report.critical)} critical{_RESET}")
    if report.warnings:
        parts.append(f"{_YELLOW}{len(report.warnings)} warning(s){_RESET}")
    if report.info:
        parts.append(f"{_CYAN}{len(report.info)} info{_RESET}")

    print(f"\n{_DIM}{'─' * 52}{_RESET}")
    print(f"  {_BOLD}Summary:{_RESET} {', '.join(parts)}")
    print(f"{_BOLD}{'═' * 52}{_RESET}\n")