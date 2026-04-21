"""Output parsing utilities.

Used by agents, rewards, and eval. Anything that parses an LLM generation lives here.
"""


def parse_plan_json(raw_text: str) -> dict | None:
    """Extract and parse the JSON plan from a Router generation. None on failure."""
    raise NotImplementedError


def extract_code_blocks(text: str) -> list[str]:
    """Return all <code>...</code> contents in order."""
    raise NotImplementedError


def extract_final_answer_tag(text: str) -> int | None:
    """Flat agent: return int inside <answer>...</answer>, or None."""
    raise NotImplementedError


def extract_trailing_number(text: str) -> int | None:
    """Fallback answer extractor: last integer in the text."""
    raise NotImplementedError
