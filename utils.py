"""
Utility functions.
"""
import re
from typing import Any, Dict
def mask_secret(key: str) -> str:
    """Mask an API key for safe logging.
    Args:
        key: raw API key string
    Returns:
        masked string, e.g. "sk-7a...917d"
    """
    if not key:
        return "<empty>"
    if len(key) <= 10:
        return key[:2] + "***"
    return key[:6] + "..." + key[-4:]
def get_time_period(hour: int) -> str:
    """Return a human-readable time-of-day label for the given hour."""
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 13:
        return "midday"
    if 13 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 23:
        return "evening"
    return "late night"
def clean_llm_output(text: str) -> str:
    """Clean common wrapper artifacts from LLM output.
    Strips markdown code fences and common prefixes.
    """
    if not text:
        return text
    text = text.strip()
    # Strip code fences
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    # Strip common prefixes
    text = re.sub(r"^(message body|content|reply)\s*[:：]", "", text).strip()
    return text.strip(" \n\t\r\"'""''")
