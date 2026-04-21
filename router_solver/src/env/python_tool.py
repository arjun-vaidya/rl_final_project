"""Sandboxed Python execution tool.

Spec: docs/03_dataset.md §"Python tool".
- Input: a code string.
- Output: stdout or exception message, truncated to 256 chars.
- Implementation: subprocess with 5-second timeout, no filesystem/network.
- Adversarial code is NOT the threat model here.
"""
from dataclasses import dataclass


@dataclass
class ToolResult:
    output: str          # stdout or exception string, truncated to 256
    is_error: bool
    wall_time_ms: int


def run_python(code: str, timeout_s: float = 5.0, max_output_chars: int = 256) -> ToolResult:
    """Execute `code` in a subprocess and return the captured output."""
    raise NotImplementedError


def looks_sensible(output: str) -> bool:
    """Cheap heuristic: non-empty, not absurdly long, parses as a number or short string.

    Used by the Solver reward (see docs/04_design.md §Solver).
    """
    raise NotImplementedError
