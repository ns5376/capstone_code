"""
Shared utilities for all LLM experiment systems (FANAR, OPENAI, CLAUDE, GEMINI, JAIS, LLAMA).
Model-specific code (API clients, etc.) stays in each system folder; this package holds common logic.
"""

from shared.normalization import (
    normalize_arabic,
    normalize_arabic_diacritics_only,
)
from shared.text_utils import (
    normalize_text,
    normalize_script,
    normalize_proficiency,
    slugify_cluster,
)
from shared.gloss import (
    set_word_list_path,
    get_word_list_path,
    load_word_gloss_map,
    get_lemma_and_gloss,
    strip_pos_from_gloss,
)
from shared.prompts import (
    ZERO_SHOT_PROMPT_TEMPLATE,
    EXAMPLE_PROMPT_TEMPLATE,
    load_examples,
    make_prompt_example,
    build_cluster_examples,
    build_zero_shot_prompt,
    build_example_prompt,
)
from shared.analysis_utils import (
    levenshtein_distance,
    extract_arabic_text,
    get_shot_type,
)

__all__ = [
    "normalize_arabic",
    "normalize_arabic_diacritics_only",
    "normalize_text",
    "normalize_script",
    "normalize_proficiency",
    "slugify_cluster",
    "set_word_list_path",
    "get_word_list_path",
    "load_word_gloss_map",
    "get_lemma_and_gloss",
    "strip_pos_from_gloss",
    "ZERO_SHOT_PROMPT_TEMPLATE",
    "EXAMPLE_PROMPT_TEMPLATE",
    "load_examples",
    "make_prompt_example",
    "build_cluster_examples",
    "build_zero_shot_prompt",
    "build_example_prompt",
    "levenshtein_distance",
    "extract_arabic_text",
    "get_shot_type",
]
