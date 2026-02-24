"""
Shared prompt templates and builders for all systems.
Supports optional gloss for the current query via {gloss_line} (empty string when no gloss).
"""

import json
import os
from typing import Callable, Dict, List, Optional

from shared.text_utils import normalize_text, normalize_script, normalize_proficiency, slugify_cluster
from shared.gloss import get_lemma_and_gloss, strip_pos_from_gloss


# Optional gloss line in templates: systems can leave it "" or add "The word means: ...\n\n"
ZERO_SHOT_PROMPT_TEMPLATE = (
    "You are a linguist with expertise in Arabic pronunciation and transcription. "
    "You will analyze user transcriptions of Arabic words along with information about their native language and Arabic proficiency level.\n\n"
    "Given the following user transcription, which may be correct or wrong, predict the intended Arabic word.\n\n"
    "User transcription: {transcription}\n\n"
    "Script: {script}\n\n"
    "{gloss_line}"
    "Native Language: {native_language}\n\n"
    "Arabic Proficiency: {arabic_proficiency}\n\n"
    "Return exactly 10 different possible Arabic words, without any short vowels from most likely to least likely.\n\n"
    "Each word must be unique. Write only 10 words, numbered 1 to 10, with no explanations or diacritics."
)

EXAMPLE_PROMPT_TEMPLATE = (
    "You are a linguist with expertise in Arabic pronunciation and transcription.\n\n"
    "You will be given examples of user transcriptions of Arabic words, the script they typed in, their native language, their Arabic proficiency level, and the correct words.\n\n"
    "Examples:\n"
    "{examples_text}\n\n"
    "Now, given the following user transcription, which may be correct or wrong, predict the intended Arabic word.\n\n"
    "User transcription: {transcription}\n\n"
    "Script: {script}\n\n"
    "{gloss_line}"
    "Native language: {native_language}\n\n"
    "Arabic proficiency: {proficiency_level}\n\n"
    "Return exactly 10 different possible Arabic words, without any short vowels from most likely to least likely.\n\n"
    "Each word must be unique. Write only 10 words, numbered 1 to 10, with no explanations or diacritics."
)


def load_examples(path: str) -> List[Dict]:
    """Load JSON list of example entries from path."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Examples file {path} must contain a list.")
    return data


def make_prompt_example(entry: Dict, word_list_path: Optional[str] = None) -> Dict[str, str]:
    """Build one example dict (transcription, script, true_word, gloss, etc.) from raw entry."""
    true_word = normalize_text(entry.get("true_word"))
    gloss_info = get_lemma_and_gloss(true_word, word_list_path)
    return {
        "transcription": normalize_text(entry.get("transcription")),
        "script": normalize_script(entry.get("script")),
        "native_language": normalize_text(entry.get("native_language")) or "Unknown",
        "proficiency": normalize_proficiency(entry.get("proficiency_level")),
        "true_word": true_word,
        "lemma": gloss_info.get("lemma", ""),
        "gloss": gloss_info.get("gloss", ""),
    }


def build_cluster_examples(
    entries: List[Dict],
    word_list_path: Optional[str] = None,
) -> Dict[str, Dict]:
    """Group entries by cluster; each value has cluster_name and list of examples (from make_prompt_example)."""
    out: Dict[str, Dict] = {}
    for entry in entries:
        cluster_name = normalize_text(entry.get("cluster"))
        slug = entry.get("cluster_id") or slugify_cluster(cluster_name)
        if slug not in out:
            out[slug] = {"cluster_name": cluster_name or "Unknown", "examples": []}
        out[slug]["examples"].append(make_prompt_example(entry, word_list_path))
    return out


def _gloss_line(gloss: Optional[str]) -> str:
    """Current-query gloss line for templates; empty if no gloss."""
    if not gloss:
        return ""
    clean = strip_pos_from_gloss(gloss)
    if not clean:
        return ""
    return f"The word means: {clean}\n\n"


def build_zero_shot_prompt(
    transcription: str,
    script: str,
    native_language: str,
    proficiency: str,
    gloss: Optional[str] = None,
) -> str:
    """Build zero-shot prompt. Pass gloss=None or '' for no gloss line."""
    return ZERO_SHOT_PROMPT_TEMPLATE.format(
        transcription=transcription,
        script=script,
        gloss_line=_gloss_line(gloss),
        native_language=native_language,
        arabic_proficiency=proficiency,
    )


def build_example_prompt(
    examples: List[Dict[str, str]],
    include_gloss_in_current_query: bool = False,
) -> Callable[..., str]:
    """
    Build a few-shot prompt builder from a list of example dicts (from make_prompt_example).
    If include_gloss_in_current_query is True, the returned builder accepts (transcription, script, native_language, proficiency, gloss).
    If False, it accepts (transcription, script, native_language, proficiency) only and gloss_line is always empty.
    """

    def format_example(ex: Dict[str, str]) -> str:
        true_word = ex["true_word"]
        gloss_text = ""
        if ex.get("gloss"):
            meaning = strip_pos_from_gloss(ex["gloss"])
            if meaning:
                gloss_text = f" The word means: {meaning}"
        return (
            "Transcription: {transcription} | Script: {script}{gloss_text} | Native language: {native_language} | "
            "Arabic proficiency: {proficiency} → True word: {true_word}"
        ).format(
            transcription=ex["transcription"],
            script=ex["script"],
            gloss_text=gloss_text,
            native_language=ex["native_language"],
            proficiency=ex["proficiency"],
            true_word=true_word,
        )

    examples_text = "\n".join(format_example(ex) for ex in examples)

    def _builder(
        transcription: str,
        script: str,
        native_language: str,
        proficiency: str,
        gloss: Optional[str] = None,
    ) -> str:
        gloss_line = _gloss_line(gloss) if include_gloss_in_current_query else ""
        return EXAMPLE_PROMPT_TEMPLATE.format(
            examples_text=examples_text,
            transcription=transcription,
            script=script,
            gloss_line=gloss_line,
            native_language=native_language,
            proficiency_level=proficiency,
        )

    return _builder
