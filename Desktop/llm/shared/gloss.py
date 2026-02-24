"""
Word list / gloss utilities shared across all systems.
Word list path is configurable so each system can point to project_root/Word Lists.xlsx (or override).
"""

import os
from typing import Dict, Optional

# Default: Word Lists.xlsx in the project root (parent of shared/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_default_path = os.path.join(_PROJECT_ROOT, "Word Lists.xlsx")

_word_list_path: Optional[str] = None
_word_gloss_map: Optional[Dict[str, str]] = None
_cached_path: Optional[str] = None


def set_word_list_path(path: Optional[str]) -> None:
    """Set the path to Word Lists.xlsx. Call from each system if using a different path."""
    global _word_list_path, _word_gloss_map, _cached_path
    _word_list_path = path
    _word_gloss_map = None
    _cached_path = None


def get_word_list_path() -> str:
    """Return the path used for the word list (default: project_root/Word Lists.xlsx)."""
    return _word_list_path if _word_list_path is not None else _default_path


def load_word_gloss_map(path: Optional[str] = None) -> Dict[str, str]:
    """Load Arabic word → gloss mapping from the 'combined' sheet (or first sheet). Cached per path."""
    global _word_gloss_map, _cached_path
    use_path = path or get_word_list_path()
    if _word_gloss_map is not None and _cached_path == use_path:
        return _word_gloss_map

    _word_gloss_map = {}
    _cached_path = use_path
    if not use_path or not os.path.isfile(use_path):
        return _word_gloss_map

    try:
        import pandas as pd
        xls = pd.ExcelFile(use_path)
        combined = [s for s in xls.sheet_names if s.lower() == "combined"]
        target_sheet = combined[0] if combined else xls.sheet_names[0]
        df = pd.read_excel(xls, target_sheet, header=None)
        if df.shape[1] < 3:
            return _word_gloss_map
        word_col, gloss_col = 1, 2
        for _, row in df[[word_col, gloss_col]].dropna().iterrows():
            w = str(row[word_col]).strip()
            g = str(row[gloss_col]).strip()
            if w and g:
                _word_gloss_map[w] = g
    except Exception:
        _word_gloss_map = {}

    return _word_gloss_map


def get_lemma_and_gloss(true_word: str, word_list_path: Optional[str] = None) -> Dict[str, str]:
    """Return dict with 'lemma' and 'gloss' if word is in the list, else {}."""
    result: Dict[str, str] = {}
    true_word = (true_word or "").strip()
    if not true_word:
        return result
    mp = load_word_gloss_map(word_list_path)
    gloss = mp.get(true_word)
    if gloss:
        result["lemma"] = true_word
        result["gloss"] = gloss
    return result


def strip_pos_from_gloss(gloss: str) -> str:
    """Remove POS tag prefixes; return meaning-only text. Handles | and :."""
    if not gloss:
        return ""
    text = str(gloss).strip()
    parts = text.split("|")
    meanings = []
    for entry in parts:
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            _, meaning = entry.split(":", 1)
            meaning = meaning.strip()
            if meaning:
                meanings.append(meaning)
        else:
            meanings.append(entry)
    return ";".join(meanings) if meanings else ""
