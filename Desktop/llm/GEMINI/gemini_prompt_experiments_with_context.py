import os
import re
import sys
import time
from datetime import datetime
from typing import Callable, Dict, List
import glob

import pandas as pd
import google.generativeai as genai

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from shared.gloss import set_word_list_path, get_lemma_and_gloss
from shared.text_utils import normalize_text, normalize_script, normalize_proficiency, slugify_cluster
from shared.prompts import (
    load_examples,
    build_cluster_examples,
    build_zero_shot_prompt,
    build_example_prompt,
    build_zero_shot_prompt_no_context,
    build_example_prompt_no_context,
)

# Point shared gloss at Word Lists.xlsx (parent of GEMINI folder)
WORD_LIST_PATH = os.path.join(PARENT_DIR, "Word Lists.xlsx")
set_word_list_path(WORD_LIST_PATH)


class GeminiArabicTop10Tester:
    """Gemini tester for 10-choice prediction. Inlined from former gemini_base."""
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model_name = os.getenv("GEMINI_MODEL_NAME", "models/gemini-2.5-flash-lite")
        self.client = genai.GenerativeModel(self.model_name)
        self.model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", self.model_name.replace("/", "_"))
        print(f"✓ Gemini tester initialized with {self.model_name}")

    def load_dataset(self, file_path: str, sample_size: int = 100) -> pd.DataFrame:
        df = pd.read_excel(file_path)
        print(f"Dataset loaded: {len(df)} rows")
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Sample size: {len(df)} rows")
        return df

    def create_top10_prompt(self, transcription: str, script: str, gloss: str | None = None) -> str:
        """Use shared zero-shot prompt (same as CLAUDE)."""
        return build_zero_shot_prompt(transcription, script, "Unknown", "Unknown", gloss)

    def parse_top10_response(self, response: str) -> list:
        lines = response.strip().split("\n")
        words = []
        for line in lines:
            match = re.match(r"^\d+[\.\)]?\s*(.+)$", line.strip())
            if match:
                words.append(match.group(1).strip())
            elif line.strip():
                words.append(line.strip())
        return words[:10]

    def query_gemini_top10(self, prompt: str) -> list:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.generate_content(prompt)
                if not hasattr(response, "candidates") or not response.candidates:
                    print("⚠️ No candidates in response")
                    return ["ERROR"] * 10
                candidate = response.candidates[0]
                finish_reason_value = None
                finish_reason = getattr(candidate, "finish_reason", None)
                if finish_reason is not None:
                    if hasattr(finish_reason, "value"):
                        finish_reason_value = finish_reason.value
                    elif hasattr(finish_reason, "name"):
                        finish_reason_value = getattr(finish_reason, "value", None)
                        if finish_reason_value is None:
                            enum_map = {"STOP": 1, "MAX_TOKENS": 2, "SAFETY": 3, "RECITATION": 4}
                            finish_reason_value = enum_map.get(finish_reason.name)
                    elif isinstance(finish_reason, int):
                        finish_reason_value = finish_reason
                    elif isinstance(finish_reason, str):
                        enum_map = {"STOP": 1, "MAX_TOKENS": 2, "SAFETY": 3, "RECITATION": 4,
                                    "FINISH_REASON_STOP": 1, "FINISH_REASON_MAX_TOKENS": 2,
                                    "FINISH_REASON_SAFETY": 3, "FINISH_REASON_RECITATION": 4}
                        finish_reason_value = enum_map.get(finish_reason.upper())
                    if finish_reason_value is not None and finish_reason_value != 1 and finish_reason_value != 2:
                        reason_names = {1: "STOP", 2: "MAX_TOKENS", 3: "SAFETY", 4: "RECITATION"}
                        print(f"⚠️ Response issue: finish_reason={finish_reason_value} ({reason_names.get(finish_reason_value, 'UNKNOWN')})")
                text_output = None
                has_valid_parts = False
                if hasattr(candidate, "content") and candidate.content and hasattr(candidate.content, "parts") and candidate.content.parts:
                    has_valid_parts = True
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            text_output = part.text.strip()
                            break
                if not text_output and has_valid_parts and hasattr(response, "text"):
                    try:
                        text_output = response.text.strip()
                    except Exception:
                        pass
                if not text_output:
                    debug_info = [f"finish_reason={finish_reason_value}" if finish_reason_value is not None else "no finish_reason"]
                    if finish_reason_value == 3:
                        print(f"⚠️ Response blocked by safety filters ({', '.join(debug_info)})")
                    else:
                        print(f"⚠️ No valid text in response ({', '.join(debug_info)})")
                    return ["ERROR"] * 10
                return self.parse_top10_response(text_output)
            except Exception as e:
                print(f"⚠️ Error: {e}")
                if attempt == max_retries - 1:
                    import traceback
                    traceback.print_exc()
                    return ["ERROR"] * 10
                time.sleep(2)
        return ["ERROR"] * 10

    def test_model(self, df: pd.DataFrame):
        results = []
        for idx, row in df.iterrows():
            transcription = str(row.get("transcription", "")).strip()
            script = str(row.get("script", "Arabic")).strip()
            true_word = str(row.get("true_word", row.get("reference", ""))).strip()
            if not transcription:
                continue
            prompt = self.create_top10_prompt(transcription, script)
            predictions = self.query_gemini_top10(prompt)
            for rank, prediction in enumerate(predictions, 1):
                results.append({
                    "row_num": idx + 1, "word_id": row.get("word_id", ""), "user_id": row.get("user_id", ""),
                    "native_language": row.get("native_language", ""), "proficiency_level": row.get("proficiency_level", ""),
                    "script": script, "transcription": transcription, "true_word": true_word,
                    "rank": rank, "prediction": prediction, "model": self.model_name,
                })
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(df)} examples...")
            time.sleep(0.1)
        return results

    def save_results(self, results: list, dataset_label: str):
        """Legacy save (runner uses overrides)."""
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_name = f"{dataset_label}_top10_predictions_{self.model_slug}_{timestamp}.xlsx"
        df.to_excel(excel_name, index=False, engine="openpyxl")
        print(f"\n✅ Results saved to: {excel_name}")
        return excel_name


def load_dataset(file_path: str, sample_size: int = 100) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return df


class PromptExperimentTesterWithContext(GeminiArabicTop10Tester):
    def __init__(self, api_key: str, experiment_id: str, prompt_builder: Callable[[str, str, str, str, str | None], str]):
        self.prompt_builder = prompt_builder
        self.experiment_id = experiment_id
        super().__init__(api_key=api_key)

    def make_prompt(self, transcription: str, script: str, native_language: str, proficiency: str, gloss: str | None) -> str:
        script_value = normalize_script(script)
        native_language_value = normalize_text(native_language) or "Unknown"
        proficiency_value = normalize_proficiency(proficiency)
        return self.prompt_builder(
            transcription,
            script_value,
            native_language_value,
            proficiency_value,
            gloss,
        )

    def test_model(self, df: pd.DataFrame):
        results: List[Dict[str, object]] = []
        for idx, row in df.iterrows():
            transcription = normalize_text(row.get("transcription"))
            script = normalize_text(row.get("script")) or "Arabic"
            true_word = normalize_text(row.get("true_word", row.get("reference", "")))
            native_language = normalize_text(row.get("native_language"))
            proficiency = normalize_proficiency(row.get("proficiency_level"))

            if not transcription:
                continue

            gloss_info = get_lemma_and_gloss(true_word)
            gloss = (gloss_info.get("gloss") or "").strip() or None

            prompt = self.make_prompt(transcription, script, native_language, proficiency, gloss)

            predictions = self.query_gemini_top10(prompt)

            for rank, prediction in enumerate(predictions, 1):
                results.append(
                    {
                        "row_num": idx + 1,
                        "word_id": row.get("word_id", ""),
                        "user_id": row.get("user_id", ""),
                        "native_language": native_language,
                        "proficiency_level": proficiency,
                        "script": script,
                        "transcription": transcription,
                        "true_word": true_word,
                        "rank": rank,
                        "prediction": prediction,
                        "model": self.model_name,
                        "prompt_id": self.experiment_id,
                    }
                )

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(df)} examples...")

            time.sleep(0.1)  # Reduced to 0.1s for faster processing (adjust if rate limited)

        return results

    def save_results(self, results: List[Dict[str, object]], dataset_label: str):
        df = pd.DataFrame(results)
        
        # Add normalized columns
        def normalize_arabic(text: str) -> str:
            if pd.isna(text) or not text:
                return ""
            text = str(text).strip()
            text = re.sub(r'[ًٌٍَُِّْ]', '', text)
            text = re.sub(r'[أإآ]', 'ا', text)
            text = re.sub(r'[ى]', 'ي', text)
            text = re.sub(r'[ة]', 'ه', text)
            return text

        if 'true_word' in df.columns:
            df['true_word_normalized'] = df['true_word'].apply(normalize_arabic)
        if 'prediction' in df.columns:
            df['prediction_normalized'] = df['prediction'].apply(normalize_arabic)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_name = f"{dataset_label}_top10_predictions_{self.experiment_id}_{self.model_slug}_{timestamp}.xlsx"

        # Decide target folder based on dataset, context/no-context, and whether gloss is enabled.
        # We detect "no gloss" via the monkey-patched get_lemma_and_gloss used in *_no_gloss runners.
        base_dir = os.path.dirname(os.path.abspath(__file__))

        experiment_id_lower = (self.experiment_id or "").lower()

        # Detect whether gloss has been disabled (no-gloss runners monkey-patch get_lemma_and_gloss to return {}).
        try:
            test_word = "أَبَوِيّ"
            test_result = get_lemma_and_gloss(test_word)
            gloss_disabled = test_result == {}  # type: ignore[comparison-overlap]
        except Exception:
            gloss_disabled = False

        if "no_context" in experiment_id_lower:
            # No-context experiments
            if gloss_disabled:
                subdir = f"{dataset_label} no context without gloss"
            else:
                subdir = f"{dataset_label} no context with gloss"
        else:
            # Context experiments (zero-shot, clusters, all-examples)
            if gloss_disabled:
                subdir = f"{dataset_label} context without gloss"
            else:
                subdir = f"{dataset_label} context with gloss"

        target_dir = os.path.join(base_dir, subdir)
        os.makedirs(target_dir, exist_ok=True)

        output_path = os.path.join(target_dir, excel_name)
        df.to_excel(output_path, index=False, engine='openpyxl')

        print("\n✅ Results saved to:")
        print(f"- {output_path}")

        return output_path


API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it before running.")
# Paths relative to project root (parent of GEMINI/); works when repo is cloned anywhere
DATASET_DIR = os.path.join(PARENT_DIR, "shared_data", "DATASET_NEW")
EXAMPLES_DIR = os.path.join(PARENT_DIR, "shared_data", "examples")

# Full test sets from shared data (no subsampling when sample_size is large)
DATASETS: Dict[str, Dict[str, str]] = {
    "arabic": {
        "path": f"{DATASET_DIR}/ARABIC_TEST_FULL.xlsx",
        "examples_path": f"{EXAMPLES_DIR}/arabic_examples.json",
    },
    "english": {
        "path": f"{DATASET_DIR}/ENGLISH_TEST_FULL.xlsx",
        "examples_path": f"{EXAMPLES_DIR}/english_examples.json",
    },
}

def prepare_dataset_configs() -> Dict[str, Dict[str, object]]:
    dataset_configs: Dict[str, Dict[str, object]] = {}
    for label, info in DATASETS.items():
        entries = load_examples(info["examples_path"])
        cluster_examples = build_cluster_examples(entries)
        all_examples: List[Dict[str, str]] = []
        for cluster in cluster_examples.values():
            all_examples.extend(cluster["examples"])
        dataset_configs[label] = {
            "path": info["path"],
            "cluster_examples": cluster_examples,
            "all_examples": all_examples,
        }
    return dataset_configs


def build_experiments_for_dataset(dataset_label: str, cluster_examples: Dict[str, Dict[str, object]], all_examples: List[Dict[str, str]]):
    experiments: List[Dict[str, object]] = [
        {
            "experiment_id": f"context_zero_shot_{dataset_label}",
            "prompt_builder": build_zero_shot_prompt,
        }
    ]

    for slug, cluster_info in cluster_examples.items():
        experiments.append(
            {
                "experiment_id": f"context_{slug}_{dataset_label}",
                "prompt_builder": build_example_prompt(cluster_info["examples"], include_gloss_in_current_query=True),
            }
        )

    experiments.append(
        {
            "experiment_id": f"context_all_examples_{dataset_label}",
            "prompt_builder": build_example_prompt(all_examples, include_gloss_in_current_query=True),
        }
    )

    return experiments


class PromptExperimentTesterUnified(PromptExperimentTesterWithContext):
    def __init__(self, api_key: str, experiment_id: str, prompt_builder: Callable[[str, str, str, str, str | None], str]):
        super().__init__(api_key=api_key, experiment_id=experiment_id, prompt_builder=prompt_builder)

    def save_results(self, results: List[Dict[str, object]], dataset_label: str):
        df = pd.DataFrame(results)
        
        # Add normalized columns
        def normalize_arabic(text: str) -> str:
            if pd.isna(text) or not text:
                return ""
            text = str(text).strip()
            text = re.sub(r'[ًٌٍَُِّْ]', '', text)
            text = re.sub(r'[أإآ]', 'ا', text)
            text = re.sub(r'[ى]', 'ي', text)
            text = re.sub(r'[ة]', 'ه', text)
            return text

        if 'true_word' in df.columns:
            df['true_word_normalized'] = df['true_word'].apply(normalize_arabic)
        if 'prediction' in df.columns:
            df['prediction_normalized'] = df['prediction'].apply(normalize_arabic)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_name = f"{dataset_label}_top10_predictions_{self.experiment_id}_{self.model_slug}_{timestamp}.xlsx"

        # Decide target folder based on dataset, context/no-context, and whether gloss is enabled.
        # We detect "no gloss" via the monkey-patched get_lemma_and_gloss used in *_no_gloss runners.
        base_dir = os.path.dirname(os.path.abspath(__file__))

        experiment_id_lower = (self.experiment_id or "").lower()

        # Detect whether gloss has been disabled (no-gloss runners monkey-patch get_lemma_and_gloss to return {}).
        try:
            test_word = "أَبَوِيّ"
            test_result = get_lemma_and_gloss(test_word)
            gloss_disabled = test_result == {}  # type: ignore[comparison-overlap]
        except Exception:
            gloss_disabled = False

        if "no_context" in experiment_id_lower:
            # No-context experiments
            if gloss_disabled:
                subdir = f"{dataset_label} no context without gloss"
            else:
                subdir = f"{dataset_label} no context with gloss"
        else:
            # Context experiments (zero-shot, clusters, all-examples)
            if gloss_disabled:
                subdir = f"{dataset_label} context without gloss"
            else:
                subdir = f"{dataset_label} context with gloss"

        target_dir = os.path.join(base_dir, subdir)
        os.makedirs(target_dir, exist_ok=True)

        output_path = os.path.join(target_dir, excel_name)
        df.to_excel(output_path, index=False, engine='openpyxl')

        print("\n✅ Results saved to:")
        print(f"- {output_path}")

        return output_path


def run_with_context_experiments(sample_size: int = 100) -> Dict[str, List[str]]:
    dataset_configs = prepare_dataset_configs()
    generated_files: Dict[str, List[str]] = {}

    for dataset_label, config in dataset_configs.items():
        cluster_examples = config["cluster_examples"]
        all_examples = config["all_examples"]

        experiments = build_experiments_for_dataset(dataset_label, cluster_examples, all_examples)

        dataset_path = config["path"]
        df = load_dataset(dataset_path, sample_size=sample_size)
        print(f"\nDataset loaded for {dataset_label.upper()}: {len(df)} rows")
        base_dir = os.path.dirname(os.path.abspath(__file__))

        for experiment in experiments:
            experiment_id = experiment["experiment_id"]
            prompt_builder = experiment["prompt_builder"]

            # Determine expected output directory, matching the logic in save_results.
            exp_id_lower = experiment_id.lower()

            # Detect whether gloss has been disabled (no-gloss runners monkey-patch get_lemma_and_gloss to return {}).
            try:
                test_word = "أَبَوِيّ"
                test_result = get_lemma_and_gloss(test_word)
                gloss_disabled = test_result == {}  # type: ignore[comparison-overlap]
            except Exception:
                gloss_disabled = False

            if "no_context" in exp_id_lower:
                if gloss_disabled:
                    subdir = f"{dataset_label} no context without gloss"
                else:
                    subdir = f"{dataset_label} no context with gloss"
            else:
                if gloss_disabled:
                    subdir = f"{dataset_label} context without gloss"
                else:
                    subdir = f"{dataset_label} context with gloss"

            target_dir = os.path.join(base_dir, subdir)
            pattern = os.path.join(
                target_dir,
                f"{dataset_label}_top10_predictions_{experiment_id}_*.xlsx",
            )
            existing = glob.glob(pattern)

            if existing:
                print("\n============================")
                print(f"Skipping context experiment (already has results): {experiment_id}")
                print(f"Existing files (first shown): {existing[0]}")
                print("============================")
                generated_files.setdefault(dataset_label, []).extend(
                    [str(p) for p in existing]
                )
                continue

            print("\n============================")
            print(f"Running context experiment: {experiment_id}")
            print("============================")

            tester = PromptExperimentTesterUnified(
                api_key=API_KEY,
                experiment_id=experiment_id,
                prompt_builder=prompt_builder,
            )

            results = tester.test_model(df)
            if not results:
                print("No results generated.")
                continue

            output_path = tester.save_results(results, dataset_label=dataset_label)
            if output_path:
                generated_files.setdefault(dataset_label, []).append(str(output_path))
            time.sleep(0.5)  # Reduced from 1.0s to 0.5s between experiments

    return generated_files


def run_no_context_experiments(sample_size: int = 100) -> Dict[str, List[str]]:
    """Run all NO-CONTEXT experiments (zero-shot + 7 clusters + all-examples). Sequential, no parallel."""
    dataset_configs = prepare_dataset_configs()

    def build_no_context_experiments(dataset_label: str, cluster_examples: Dict[str, Dict[str, object]], all_examples: List[Dict[str, str]]):
        experiments: List[Dict[str, object]] = [
            {
                "experiment_id": f"no_context_zero_shot_{dataset_label}",
                "prompt_builder": build_zero_shot_prompt_no_context,
            },
        ]
        for slug, cluster_info in cluster_examples.items():
            experiments.append({
                "experiment_id": f"no_context_{slug}_{dataset_label}",
                "prompt_builder": build_example_prompt_no_context(
                    cluster_info["examples"],
                    include_gloss_in_current_query=True,
                ),
            })
        experiments.append({
            "experiment_id": f"no_context_all_examples_{dataset_label}",
            "prompt_builder": build_example_prompt_no_context(
                all_examples,
                include_gloss_in_current_query=True,
            ),
        })
        return experiments

    generated_files: Dict[str, List[str]] = {}
    for dataset_label, config in dataset_configs.items():
        cluster_examples = config["cluster_examples"]
        all_examples = config["all_examples"]
        experiments = build_no_context_experiments(dataset_label, cluster_examples, all_examples)
        dataset_path = config["path"]
        df = load_dataset(dataset_path, sample_size=sample_size)
        print(f"\nDataset loaded for {dataset_label.upper()} (NO CONTEXT): {len(df)} rows")
        base_dir = os.path.dirname(os.path.abspath(__file__))

        for experiment in experiments:
            experiment_id = experiment["experiment_id"]
            prompt_builder = experiment["prompt_builder"]
            try:
                test_result = get_lemma_and_gloss("أَبَوِيّ")
                gloss_disabled = test_result == {}
            except Exception:
                gloss_disabled = False
            subdir = f"{dataset_label} no context without gloss" if gloss_disabled else f"{dataset_label} no context with gloss"
            target_dir = os.path.join(base_dir, subdir)
            pattern = os.path.join(target_dir, f"{dataset_label}_top10_predictions_{experiment_id}_*.xlsx")
            if glob.glob(pattern):
                print(f"\nSkipping (already has results): {experiment_id}")
                continue
            print(f"\nRunning NO-CONTEXT experiment: {experiment_id}")
            tester = PromptExperimentTesterUnified(api_key=API_KEY, experiment_id=experiment_id, prompt_builder=prompt_builder)
            results = tester.test_model(df)
            if not results:
                continue
            output_path = tester.save_results(results, dataset_label=dataset_label)
            if output_path:
                generated_files.setdefault(dataset_label, []).append(str(output_path))
            time.sleep(0.5)
    return generated_files


if __name__ == "__main__":
    # Full test set (no subsampling)
    SAMPLE_SIZE = 1_000_000

    # Pass 1: WITH GLOSS (no-context + context)
    print("=" * 80)
    print("GEMINI: WITH GLOSS (no-context + context, full test set)")
    print("=" * 80)
    run_no_context_experiments(sample_size=SAMPLE_SIZE)
    run_with_context_experiments(sample_size=SAMPLE_SIZE)

    # Pass 2: WITHOUT GLOSS
    def _no_gloss(_true_word: str) -> dict:
        return {}

    _orig_get = get_lemma_and_gloss
    globals()["get_lemma_and_gloss"] = _no_gloss
    try:
        print("\n" + "=" * 80)
        print("GEMINI: WITHOUT GLOSS (no-context + context, full test set)")
        print("=" * 80)
        run_no_context_experiments(sample_size=SAMPLE_SIZE)
        run_with_context_experiments(sample_size=SAMPLE_SIZE)
    finally:
        globals()["get_lemma_and_gloss"] = _orig_get

