"""
Shared analysis utilities:
- Levenshtein distance
- Arabic extraction / normalization wrappers
- Experiment ID → shot type
- Generic TOP-1 / TOP-10 metrics computation
"""

import re
from typing import Union, Dict

import pandas as pd

from shared.normalization import normalize_arabic, normalize_arabic_diacritics_only


def levenshtein_distance(s1: str, s2: str) -> int:
    """Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (1 if c1 != c2 else 0),
            ))
        prev = curr
    return prev[-1]


def extract_arabic_text(text: Union[str, float, None]) -> str:
    """
    Extract Arabic from mixed English/Arabic. If no Arabic is found, return the original stripped text.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).strip()
    if not text:
        return ""
    arabic_pattern = r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+"
    matches = re.findall(arabic_pattern, text)
    if matches:
        result = " ".join(matches)
        result = re.sub(r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]", "", result)
        result = re.sub(r"\s+", " ", result).strip()
        return result
    # No Arabic found: return original stripped text (for comparison with whole prediction)
    return text


def get_shot_type(prompt_id: str) -> str:
    """Map experiment prompt_id to shot type: '0', 'c1'..'c7', '35', or 'unknown'."""
    pid = (prompt_id or "").strip().lower()
    if "zero_shot" in pid or pid.endswith("_0"):
        return "0"
    if "all_examples" in pid or "35" in pid:
        return "35"
    if "cluster_indo_euro_germanic_romance" in pid or "cluster_indo_european_germanic_and_romance" in pid:
        return "c1"
    if "cluster_indo_euro_slavic" in pid or "cluster_indo_european_slavic" in pid:
        return "c2"
    if "cluster_indo_euro_indo_aryan" in pid or "cluster_indo_european_indo_aryan" in pid or "cluster_indo_iranian" in pid or "cluster_indo_european_indo_iranian" in pid:
        return "c3"
    if "cluster_semitic" in pid or "cluster_afro_asiatic" in pid:
        return "c4"
    if "cluster_turkic" in pid or "cluster_niger_congo" in pid or "cluster_uralic_turkic" in pid:
        return "c5"
    if "cluster_sino_tibetan" in pid or "cluster_east_asian" in pid:
        return "c6"
    if "cluster_other" in pid:
        return "c7"
    return "unknown"


def calculate_metrics(df: pd.DataFrame, normalization_type: str = "full") -> Dict[str, float]:
    """
    Calculate TOP-1, TOP-10, and average Levenshtein distance for a result table.

    Expected columns:
    - 'row_num'         : groups a single test case across up to 10 ranks
    - 'true_word'       : gold word
    - 'prediction'      : model prediction (possibly mixed-language)
    - 'rank'            : 1–10

    normalization_type:
    - "full"            : use normalize_arabic (diacritics + alef/ya/ta marbuta)
    - "diacritics_only" : use normalize_arabic_diacritics_only
    - "none"            : raw strings
    """
    if df is None or len(df) == 0:
        return {
            "top1_accuracy": 0.0,
            "top10_accuracy": 0.0,
            "avg_levenshtein": 0.0,
        }

    df = df.copy()

    # Clean prediction values and extract Arabic (or fall back to full string)
    df["prediction"] = df["prediction"].apply(
        lambda x: "" if pd.isna(x) else str(x).strip()
    )
    df["prediction"] = df["prediction"].apply(extract_arabic_text)

    if len(df) == 0:
        return {
            "top1_accuracy": 0.0,
            "top10_accuracy": 0.0,
            "avg_levenshtein": 0.0,
        }

    # Ensure rank is numeric
    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    # Prepare processed columns according to normalization setting
    if normalization_type == "full":
        df["prediction_processed"] = df["prediction"].apply(normalize_arabic)
        df["true_word_processed"] = df["true_word"].apply(normalize_arabic)
    elif normalization_type == "diacritics_only":
        df["prediction_processed"] = df["prediction"].apply(
            normalize_arabic_diacritics_only
        )
        df["true_word_processed"] = df["true_word"].apply(
            normalize_arabic_diacritics_only
        )
    else:  # "none"
        df["prediction_processed"] = df["prediction"].astype(str).str.strip()
        df["true_word_processed"] = df["true_word"].astype(str).str.strip()

    # Group by unique test cases
    unique_cases = df.groupby("row_num").first()

    results = []
    for case_id, case_data in unique_cases.iterrows():
        case_predictions = df[df["row_num"] == case_id].sort_values("rank")
        true_word_processed = str(case_data["true_word_processed"]).strip()

        if not true_word_processed or true_word_processed in ("", "nan"):
            continue

        matches = case_predictions[
            case_predictions["prediction_processed"] == true_word_processed
        ]

        top1_match = 0
        top10_match = 0
        if len(matches) > 0:
            first_rank = matches.iloc[0]["rank"]
            top10_match = 1
            if first_rank == 1:
                top1_match = 1

        # Average Levenshtein over top-10 predictions
        top10_predictions = case_predictions.head(10)
        if len(top10_predictions) > 0:
            dists = [
                levenshtein_distance(true_word_processed, pred)
                for pred in top10_predictions["prediction_processed"]
                if pred
            ]
            avg_lev = (
                sum(dists) / len(dists) if dists else len(true_word_processed)
            )
        else:
            avg_lev = len(true_word_processed)

        results.append(
            {
                "case_id": case_id,
                "top1_match": top1_match,
                "top10_match": top10_match,
                "avg_levenshtein": avg_lev,
            }
        )

    total_cases = len(results)
    if total_cases == 0:
        return {
            "top1_accuracy": 0.0,
            "top10_accuracy": 0.0,
            "avg_levenshtein": 0.0,
        }

    top1_correct = sum(1 for r in results if r["top1_match"] == 1)
    top10_correct = sum(1 for r in results if r["top10_match"] == 1)
    avg_levenshtein_distances = [r["avg_levenshtein"] for r in results]

    return {
        "top1_accuracy": top1_correct / total_cases * 100.0,
        "top10_accuracy": top10_correct / total_cases * 100.0,
        "avg_levenshtein": sum(avg_levenshtein_distances)
        / len(avg_levenshtein_distances)
        if avg_levenshtein_distances
        else 0.0,
    }

