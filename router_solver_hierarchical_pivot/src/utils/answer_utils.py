import re
from typing import Optional


BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}")
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")
ANSWER_CUE_RE = re.compile(
    r"\b(?:final answer|answer|the answer is|therefore|thus|so)\b\s*[:\-]?\s*(.*)",
    re.IGNORECASE,
)
FINAL_ANSWER_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:final answer|answer is|the answer is|therefore|thus|so)\s*[:=]?\s*(.+)$",
    re.IGNORECASE,
)
STEP_LABEL_RE = re.compile(r"^\s*(?:#+\s*)?step\s+\d+\b", re.IGNORECASE)


def clean_answer_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    cleaned = str(text).strip()
    cleaned = cleaned.replace("\r", "")
    cleaned = re.sub(r"\*\*", "", cleaned)
    return cleaned


def extract_numeric_value(text: Optional[str]) -> Optional[float]:
    cleaned = clean_answer_text(text)
    if not cleaned:
        return None

    boxed_matches = BOXED_RE.findall(cleaned)
    for boxed in reversed(boxed_matches):
        match = NUMBER_RE.findall(boxed.replace(",", ""))
        if match:
            try:
                return float(match[-1])
            except ValueError:
                pass

    matches = NUMBER_RE.findall(cleaned.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _line_numeric_value(line: str) -> Optional[float]:
    stripped = clean_answer_text(line)
    if not stripped or STEP_LABEL_RE.match(stripped):
        return None

    boxed_matches = BOXED_RE.findall(stripped)
    for boxed in reversed(boxed_matches):
        match = NUMBER_RE.findall(boxed.replace(",", ""))
        if match:
            try:
                return float(match[-1])
            except ValueError:
                pass

    if "=" in stripped:
        rhs = stripped.rsplit("=", 1)[-1]
        rhs_match = NUMBER_RE.findall(rhs.replace(",", ""))
        if rhs_match:
            try:
                return float(rhs_match[-1])
            except ValueError:
                pass

    matches = NUMBER_RE.findall(stripped.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def extract_final_answer(text: Optional[str], strict: bool = False) -> str:
    cleaned = clean_answer_text(text)
    if not cleaned:
        return ""

    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]

    if strict:
        for line in reversed(lines):
            match = FINAL_ANSWER_RE.match(line)
            if match:
                candidate = match.group(1).strip()
                numeric = _line_numeric_value(candidate)
                if numeric is not None:
                    return str(int(numeric)) if float(numeric).is_integer() else str(numeric)
        return ""

    for line in reversed(lines):
        if "\\boxed" in line and "{" in line:
            numeric = _line_numeric_value(line)
            if numeric is not None:
                return str(int(numeric)) if float(numeric).is_integer() else str(numeric)

    for line in reversed(lines):
        cue_match = ANSWER_CUE_RE.search(line)
        if cue_match:
            candidate = cue_match.group(1).strip()
            numeric = _line_numeric_value(candidate)
            if numeric is not None:
                return str(int(numeric)) if float(numeric).is_integer() else str(numeric)

    for line in reversed(lines):
        if len(line) <= 80:
            numeric = _line_numeric_value(line)
            if numeric is not None:
                return str(int(numeric)) if float(numeric).is_integer() else str(numeric)

    numeric = extract_numeric_value(cleaned)
    if numeric is not None:
        return str(int(numeric)) if float(numeric).is_integer() else str(numeric)

    for line in reversed(lines):
        if line:
            return line
    return cleaned
