"""
Single analysis entry point for FANAR.

Processes all result folders (with gloss, without gloss, context, no context).
Uses shared.analysis_utils. Writes experiment_results_analysis_fanar.xlsx
(columns: FANAR_with_gloss, FANAR_without_gloss).

Usage: python analyze_results.py
"""
import os
import sys
import glob
import pandas as pd
from collections import defaultdict

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CURRENT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
from shared.analysis_utils import get_shot_type, calculate_metrics as shared_calculate_metrics

SYSTEM_LABEL = "FANAR"


def has_context(prompt_id: str, folder_path: str) -> str:
    if "no context" in folder_path.lower():
        return "No"
    if "context" in folder_path.lower():
        return "Yes"
    if "no_context" in (prompt_id or "").lower():
        return "No"
    if "context" in (prompt_id or "").lower():
        return "Yes"
    return "No"


def get_script_type(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "Ar"
    scripts = df["script"].dropna().unique()
    script_str = " ".join(str(s).lower() for s in scripts)
    if any(k in script_str for k in ["romanized", "english", "rom"]):
        return "Rom"
    return "Ar"


def _process_folders(folder_list, all_norm, all_no_norm, all_diac, label: str) -> None:
    for folder_path in folder_list:
        if not os.path.exists(folder_path):
            print(f"⚠️  Folder not found: {folder_path}")
            continue
        files = [f for f in glob.glob(os.path.join(folder_path, "*.xlsx")) if not os.path.basename(f).startswith("~$")]
        for excel_file in files:
            print(f"Processing {label}: {os.path.basename(excel_file)}")
            try:
                df = pd.read_excel(excel_file, engine="openpyxl")
                if len(df) == 0:
                    continue
                prompt_id = df["prompt_id"].iloc[0] if "prompt_id" in df.columns else ""
                shot_type = get_shot_type(prompt_id)
                context = has_context(prompt_id, folder_path)
                script = get_script_type(df)
                key = (context, shot_type, script)
                all_norm[key][SYSTEM_LABEL] = shared_calculate_metrics(df, "full")
                all_no_norm[key][SYSTEM_LABEL] = shared_calculate_metrics(df, "none")
                all_diac[key][SYSTEM_LABEL] = shared_calculate_metrics(df, "diacritics_only")
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()


def analyze_all_results():
    """Analyze WITH GLOSS and WITHOUT GLOSS folders; return 6 result dicts."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folders_with_gloss = [
        os.path.join(base_dir, "arabic no context with gloss"),
        os.path.join(base_dir, "english no context with gloss"),
        os.path.join(base_dir, "arabic context with gloss"),
        os.path.join(base_dir, "english context with gloss"),
    ]
    folders_without_gloss = [
        os.path.join(base_dir, "arabic no context without gloss"),
        os.path.join(base_dir, "english no context without gloss"),
        os.path.join(base_dir, "arabic context without gloss"),
        os.path.join(base_dir, "english context without gloss"),
    ]
    all_norm_with = defaultdict(lambda: defaultdict(dict))
    all_no_norm_with = defaultdict(lambda: defaultdict(dict))
    all_diac_with = defaultdict(lambda: defaultdict(dict))
    all_norm_without = defaultdict(lambda: defaultdict(dict))
    all_no_norm_without = defaultdict(lambda: defaultdict(dict))
    all_diac_without = defaultdict(lambda: defaultdict(dict))

    print("=" * 80)
    print("Processing WITH GLOSS results...")
    print("=" * 80)
    _process_folders(folders_with_gloss, all_norm_with, all_no_norm_with, all_diac_with, "WITH GLOSS")
    print("\n" + "=" * 80)
    print("Processing WITHOUT GLOSS results...")
    print("=" * 80)
    _process_folders(folders_without_gloss, all_norm_without, all_no_norm_without, all_diac_without, "WITHOUT GLOSS")

    return (
        all_norm_with, all_no_norm_with, all_diac_with,
        all_norm_without, all_no_norm_without, all_diac_without,
    )


def create_excel_tables(
    all_norm_with, all_no_norm_with, all_diac_with,
    all_norm_without, all_no_norm_without, all_diac_without,
):
    def create_rows(all_with, all_without):
        rows_top1 = []
        rows_top10 = []
        rows_avg_lev = []
        for context in ["No", "Yes"]:
            for shot in ["0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "35"]:
                for script in ["Ar", "Rom"]:
                    key = (context, shot, script)
                    m_with = all_with.get(key, {}).get(SYSTEM_LABEL, {})
                    m_without = all_without.get(key, {}).get(SYSTEM_LABEL, {})
                    rows_top1.append({
                        "Context": context, "Shot": shot, "Script": script,
                        "FANAR_with_gloss": round(m_with.get("top1_accuracy", 0), 2),
                        "FANAR_without_gloss": round(m_without.get("top1_accuracy", 0), 2),
                    })
                    rows_top10.append({
                        "Context": context, "Shot": shot, "Script": script,
                        "FANAR_with_gloss": round(m_with.get("top10_accuracy", 0), 2),
                        "FANAR_without_gloss": round(m_without.get("top10_accuracy", 0), 2),
                    })
                    rows_avg_lev.append({
                        "Context": context, "Shot": shot, "Script": script,
                        "FANAR_with_gloss": round(m_with.get("avg_levenshtein", 0), 2),
                        "FANAR_without_gloss": round(m_without.get("avg_levenshtein", 0), 2),
                    })
        return {
            "top1": pd.DataFrame(rows_top1),
            "top10": pd.DataFrame(rows_top10),
            "avg_levenshtein": pd.DataFrame(rows_avg_lev),
        }

    dfs_norm = create_rows(all_norm_with, all_norm_without)
    dfs_no_norm = create_rows(all_no_norm_with, all_no_norm_without)
    dfs_diac = create_rows(all_diac_with, all_diac_without)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(base_dir, "experiment_results_analysis_fanar.xlsx")
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        dfs_norm["top1"].to_excel(writer, sheet_name="TOP1_ACCURACY_NORM", index=False)
        dfs_norm["top10"].to_excel(writer, sheet_name="TOP10_ACCURACY_NORM", index=False)
        dfs_norm["avg_levenshtein"].to_excel(writer, sheet_name="AVG_LEVENSHTEIN_NORM", index=False)
        dfs_no_norm["top1"].to_excel(writer, sheet_name="TOP1_ACCURACY_NO_NORM", index=False)
        dfs_no_norm["top10"].to_excel(writer, sheet_name="TOP10_ACCURACY_NO_NORM", index=False)
        dfs_no_norm["avg_levenshtein"].to_excel(writer, sheet_name="AVG_LEVENSHTEIN_NO_NORM", index=False)
        dfs_diac["top1"].to_excel(writer, sheet_name="TOP1_ACCURACY_DIACRITICS_ONLY", index=False)
        dfs_diac["top10"].to_excel(writer, sheet_name="TOP10_ACCURACY_DIACRITICS_ONLY", index=False)
        dfs_diac["avg_levenshtein"].to_excel(writer, sheet_name="AVG_LEVENSHTEIN_DIACRITICS_ONLY", index=False)
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    print("=" * 80)
    print("FANAR: Single analysis (with gloss + without gloss)")
    print("=" * 80)
    r_norm_w, r_no_w, r_diac_w, r_norm_wo, r_no_wo, r_diac_wo = analyze_all_results()
    print("\nCreating Excel...")
    create_excel_tables(r_norm_w, r_no_w, r_diac_w, r_norm_wo, r_no_wo, r_diac_wo)
    print("✅ Analysis complete.")
