import json
import re
from typing import Optional, List, Dict

def parse_plan_json(text: str) -> Optional[Dict]:
    """
    Attempts to extract and parse a JSON plan from the Router's output.
    Looks for JSON within { ... } blocks.
    """
    # Simple regex to find the first JSON object
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        return None
    
    try:
        data = json.loads(match.group(1))
        # Basic validation: must have "plan" key and be a list
        if "plan" in data and isinstance(data["plan"], list):
            return data
    except json.JSONDecodeError:
        pass
    return None

def extract_code_block(text: str) -> Optional[str]:
    """
    Extracts code from <code>...</code> blocks.
    Returns the first one found.
    """
    match = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_all_code_blocks(text: str) -> List[str]:
    """Extracts all code blocks from the trajectory."""
    return re.findall(r"<code>(.*?)</code>", text, re.DOTALL)
