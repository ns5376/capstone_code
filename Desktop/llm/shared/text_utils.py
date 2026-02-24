"""
Common text/field normalization and slugify used for prompts and datasets.
"""

import re
from typing import Union


def normalize_text(value: Union[str, float, None]) -> str:
    """Strip and coerce to string; treat NaN/None as empty."""
    if value is None:
        return ""
    if isinstance(value, float):
        try:
            import pandas as pd
            if pd.isna(value):
                return ""
        except ImportError:
            pass
    return str(value).strip()


def normalize_script(value: Union[str, float, None]) -> str:
    """Normalize script field; default to 'Arabic' if empty."""
    text = normalize_text(value)
    if not text:
        return "Arabic"
    return text.capitalize()


def normalize_proficiency(value: Union[str, float, None]) -> str:
    """Normalize proficiency level; default to 'Unknown' if empty."""
    if value is None or value == "":
        return "Unknown"
    try:
        number = float(value)
        if number == int(number):
            return str(int(number))
        return str(number)
    except (TypeError, ValueError):
        return normalize_text(value) or "Unknown"


def slugify_cluster(name: str) -> str:
    """Turn cluster name into a safe slug like cluster_indo_european."""
    text = normalize_text(name)
    if not text:
        text = "unknown_cluster"
    text = text.lower()
    text = text.replace("&", "and")
    text = re.sub(r"[()]", " ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return f"cluster_{text}" if text else "cluster_unknown"
