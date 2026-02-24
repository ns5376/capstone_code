"""
Arabic text normalization used for evaluation across all systems.
No external normalizer library; regex-based rules only.
"""

import re
from typing import Union

import pandas as pd


def normalize_arabic(text: Union[str, float, None]) -> str:
    """Full normalization: remove diacritics, normalize alef/ya/ta marbuta."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    if not text:
        return ""
    text = str(text).strip()
    # Remove diacritics: [ًٌٍَُِّْ]
    text = re.sub(r"[ًٌٍَُِّْ]", "", text)
    # Normalize alef variants: [أإآ] -> ا
    text = re.sub(r"[أإآ]", "ا", text)
    # Normalize ya: [ى] -> ي
    text = re.sub(r"[ى]", "ي", text)
    # Normalize ta marbuta: [ة] -> ه
    text = re.sub(r"[ة]", "ه", text)
    return text


def normalize_arabic_diacritics_only(text: Union[str, float, None]) -> str:
    """Remove diacritics only; no other character normalization."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    if not text:
        return ""
    text = str(text).strip()
    text = re.sub(r"[ًٌٍَُِّْ]", "", text)
    return text
