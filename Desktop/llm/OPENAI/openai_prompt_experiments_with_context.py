"""
OpenAI prompt experiments: no-context and context, with/without gloss.
Uses shared prompts, gloss, text_utils. Model is set via OPENAI_MODEL_NAME.
Run via run_openai_gpt4o.py or run_openai_gpt5.py (they set model and call with/without gloss).
"""
import os
import re
import sys
import time
import glob
from datetime import datetime
from typing import Callable, Dict, List

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

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

set_word_list_path(os.path.join(PARENT_DIR, "Word Lists.xlsx"))

SAMPLE_SIZE = 1_000_000
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")
# Paths relative to project root (parent of OPENAI/); works when repo is cloned anywhere
DATASET_DIR = os.path.join(PARENT_DIR, "shared_data", "DATASET_NEW")
EXAMPLES_DIR = os.path.join(PARENT_DIR, "shared_data", "examples")

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


def _model_folder_suffix(model_name: str) -> str:
    """Return 'gpt4o' for gpt-4o, 'gpt5nano' for gpt-5-nano, 'gpt5' for gpt-5."""
    if not model_name:
        return "gpt5"
    m = (model_name or "").lower()
    if "gpt-4o" in m or "gpt-4-o" in m:
        return "gpt4o"
    if "nano" in m:
        return "gpt5nano"
    return "gpt5"


def _normalize_arabic_cell(text: str) -> str:
    if pd.isna(text) or not text:
        return ""
    text = str(text).strip()
    text = re.sub(r"[ًٌٍَُِّْ]", "", text)
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"[ى]", "ي", text)
    text = re.sub(r"[ة]", "ه", text)
    return text


# ---------- Inlined OpenAIArabicTop10Tester (from openai_base) ----------
class OpenAIArabicTop10Tester:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        model_env = os.getenv("OPENAI_MODEL_NAME")
        if model_env and str(model_env).strip():
            self.model_name = str(model_env).strip()
            print(f"✓ OpenAI tester initialized with {self.model_name} (from OPENAI_MODEL_NAME)")
        else:
            self.model_name = "gpt-5"
            print(f"✓ OpenAI tester initialized with {self.model_name} (default)")

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
        if not response or not response.strip():
            return []
        lines = response.strip().split("\n")
        words = []
        for line in lines:
            match = re.match(r"^\d+[\.\)]?\s*(.+)$", line.strip())
            if match:
                words.append(match.group(1).strip())
            elif line.strip():
                words.append(line.strip())
        return words[:10]

    def query_openai_top10(self, prompt: str) -> list:
        max_retries = 5
        backoff = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                if "gpt-5" in self.model_name.lower():
                    response = self.client.responses.create(
                        model=self.model_name,
                        input=prompt,
                        reasoning={"effort": "minimal"},
                        max_output_tokens=120,
                    )
                    text_output = ""
                    if hasattr(response, "output_text") and response.output_text:
                        text_output = str(response.output_text)
                    else:
                        try:
                            text_output = str(response.output[0].content[0].text)
                        except Exception:
                            text_output = str(response)
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=300,
                        timeout=40,
                        temperature=0.4,
                    )
                    msg = response.choices[0].message
                    text_output = (getattr(msg, "content", "") or "").strip()
                return self.parse_top10_response(text_output)
            except Exception as e:
                msg = str(e)
                if "rate_limit_exceeded" in msg or "429" in msg:
                    print(f"⚠️ Rate limit (attempt {attempt}/{max_retries}), sleeping {backoff:.1f}s...")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                print(f"⚠️ Error from OpenAI: {e}")
                return ["ERROR"] * 10
        print("⚠️ Rate limit persisted; returning ERROR placeholders.")
        return ["ERROR"] * 10


def load_dataset(file_path: str, sample_size: int = 100) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return df


def experiment_already_exists(dataset_label: str, experiment_id: str, model_name: str, folder_suffix: str) -> bool:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, f"{dataset_label} {folder_suffix}")
    if not os.path.exists(folder):
        return False
    pattern = os.path.join(folder, f"{dataset_label}_top10_predictions_{experiment_id}_{model_name}_*.xlsx")
    matching = glob.glob(pattern)
    for fpath in matching:
        try:
            df = pd.read_excel(fpath)
            if len(df) > 0:
                return True
        except Exception:
            continue
    return False


# ---------- Context experiments (use shared prompts) ----------
class PromptExperimentTesterWithContext(OpenAIArabicTop10Tester):
    def __init__(self, api_key: str, experiment_id: str, prompt_builder: Callable[[str, str, str, str, str | None], str]):
        self.prompt_builder = prompt_builder
        self.experiment_id = experiment_id
        super().__init__(api_key=api_key)

    def make_prompt(self, transcription: str, script: str, native_language: str, proficiency: str, gloss: str | None) -> str:
        script_value = normalize_script(script)
        native_language_value = normalize_text(native_language) or "Unknown"
        proficiency_value = normalize_proficiency(proficiency)
        gloss_str = (gloss or "").strip() or None
        return self.prompt_builder(
            transcription, script_value, native_language_value, proficiency_value, gloss_str
        )

    def test_model(self, df: pd.DataFrame):
        results: List[Dict[str, object]] = []
        rows = list(df.iterrows())
        for i, (idx, row) in enumerate(rows):
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
            predictions = self.query_openai_top10(prompt)
            while len(predictions) < 10:
                predictions.append("")
            predictions = predictions[:10]
            for rank, prediction in enumerate(predictions, 1):
                results.append({
                    "row_num": idx + 1, "word_id": row.get("word_id", ""), "user_id": row.get("user_id", ""),
                    "native_language": native_language, "proficiency_level": proficiency,
                    "script": script, "transcription": transcription, "true_word": true_word,
                    "rank": rank, "prediction": prediction, "model": self.model_name, "prompt_id": self.experiment_id,
                })
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(rows)} examples...")
            time.sleep(0.15)
        return results

    def save_results(self, results: List[Dict[str, object]], dataset_label: str):
        df = pd.DataFrame(results)
        if "true_word" in df.columns:
            df["true_word_normalized"] = df["true_word"].apply(_normalize_arabic_cell)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        suffix = _model_folder_suffix(self.model_name)
        folder = os.path.join(base_dir, f"{dataset_label} context with gloss {suffix}")
        os.makedirs(folder, exist_ok=True)
        excel_name = os.path.join(folder, f"{dataset_label}_top10_predictions_{self.experiment_id}_{self.model_name}_{timestamp}.xlsx")
        df.to_excel(excel_name, index=False, engine="openpyxl")
        print(f"\n✅ Results saved to: {excel_name}")
        return excel_name


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


def build_experiments_for_dataset(
    dataset_label: str,
    cluster_examples: Dict[str, Dict[str, object]],
    all_examples: List[Dict[str, str]],
):
    experiments: List[Dict[str, object]] = [
        {"experiment_id": f"context_zero_shot_{dataset_label}", "prompt_builder": build_zero_shot_prompt},
    ]
    for slug, cluster_info in cluster_examples.items():
        experiments.append({
            "experiment_id": f"context_{slug}_{dataset_label}",
            "prompt_builder": build_example_prompt(cluster_info["examples"], include_gloss_in_current_query=True),
        })
    experiments.append({
        "experiment_id": f"context_all_examples_{dataset_label}",
        "prompt_builder": build_example_prompt(all_examples, include_gloss_in_current_query=True),
    })
    return experiments


# ---------- No-context experiments (same shared prompts as context, like CLAUDE) ----------
class OpenAINoContextPromptTester(OpenAIArabicTop10Tester):
    """No-context tester using same 5-arg prompt_builder as context (shared prompts)."""
    def __init__(self, api_key: str, experiment_id: str, prompt_builder: Callable[[str, str, str, str, str | None], str]):
        self.prompt_builder = prompt_builder
        self.experiment_id = experiment_id
        super().__init__(api_key=api_key)

    def make_prompt(self, transcription: str, script: str, native_language: str, proficiency: str, gloss: str | None) -> str:
        script_value = normalize_script(script)
        # NO-CONTEXT: do not pass user metadata (native language, proficiency) into the prompt.
        # We still provide placeholder values so the shared templates render correctly, but
        # they no longer reflect the actual user-level fields from the dataset.
        native_language_value = "Unknown"
        proficiency_value = "Unknown"
        gloss_str = (gloss or "").strip() or None
        return self.prompt_builder(
            transcription, script_value, native_language_value, proficiency_value, gloss_str
        )

    def test_model(self, df: pd.DataFrame):
        results: List[Dict[str, object]] = []
        rows = list(df.iterrows())
        for i, (idx, row) in enumerate(rows):
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
            predictions = self.query_openai_top10(prompt)
            while len(predictions) < 10:
                predictions.append("")
            predictions = predictions[:10]
            for rank, prediction in enumerate(predictions, 1):
                results.append({
                    "row_num": idx + 1, "word_id": row.get("word_id", ""), "user_id": row.get("user_id", ""),
                    "native_language": native_language, "proficiency_level": proficiency,
                    "script": script, "transcription": transcription, "true_word": true_word,
                    "rank": rank, "prediction": prediction or "", "model": self.model_name, "prompt_id": self.experiment_id,
                })
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(rows)} examples...")
            time.sleep(0.15)
        return results

    def save_results(self, results: list, dataset_label: str):
        df = pd.DataFrame(results)
        if "true_word" in df.columns:
            df["true_word_normalized"] = df["true_word"].apply(_normalize_arabic_cell)
        if "prediction" in df.columns:
            df["prediction_normalized"] = df["prediction"].apply(_normalize_arabic_cell)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        suffix = _model_folder_suffix(self.model_name)
        folder = os.path.join(base_dir, f"{dataset_label} no context with gloss {suffix}")
        os.makedirs(folder, exist_ok=True)
        excel_name = os.path.join(
            folder,
            f"{dataset_label}_top10_predictions_{self.experiment_id}_{self.model_name}_{timestamp}.xlsx",
        )
        df.to_excel(excel_name, index=False, engine="openpyxl")
        print(f"\n✅ Results saved to: {excel_name}")
        return excel_name


def build_no_context_experiments_for_dataset(
    dataset_label: str,
    cluster_examples: Dict[str, Dict[str, object]],
    all_examples: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    """Prompt builders that exclude user metadata from the prompt text."""
    experiments: List[Dict[str, object]] = []
    zero_id = "zero_shot" if dataset_label == "arabic" else "no_context_zero_shot_english"
    all_id = "all_examples" if dataset_label == "arabic" else "no_context_all_examples_english"
    experiments.append({"experiment_id": zero_id, "prompt_builder": build_zero_shot_prompt_no_context})
    for slug, cluster_info in cluster_examples.items():
        exp_id = slug if dataset_label == "arabic" else f"no_context_{slug}_english"
        experiments.append({
            "experiment_id": exp_id,
            "prompt_builder": build_example_prompt_no_context(
                cluster_info["examples"],
                include_gloss_in_current_query=True,
            ),
        })
    experiments.append({
        "experiment_id": all_id,
        "prompt_builder": build_example_prompt_no_context(
            all_examples,
            include_gloss_in_current_query=True,
        ),
    })
    return experiments


class PromptExperimentTesterUnified(PromptExperimentTesterWithContext):
    def __init__(self, api_key: str, experiment_id: str, prompt_builder: Callable[[str, str, str, str, str | None], str]):
        super().__init__(api_key=api_key, experiment_id=experiment_id, prompt_builder=prompt_builder)

    def save_results(self, results: List[Dict[str, object]], dataset_label: str):
        df = pd.DataFrame(results)
        if "true_word" in df.columns:
            df["true_word_normalized"] = df["true_word"].apply(_normalize_arabic_cell)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        suffix = _model_folder_suffix(self.model_name)
        folder = os.path.join(base_dir, f"{dataset_label} context with gloss {suffix}")
        os.makedirs(folder, exist_ok=True)
        excel_name = os.path.join(
            folder,
            f"{dataset_label}_top10_predictions_{self.experiment_id}_{self.model_name}_{timestamp}.xlsx",
        )
        df.to_excel(excel_name, index=False, engine="openpyxl")
        print(f"\n✅ Results saved to: {excel_name}")
        return excel_name


def run_with_context_experiments(sample_size: int = SAMPLE_SIZE) -> None:
    dataset_configs = prepare_dataset_configs()
    for dataset_label, config in dataset_configs.items():
        cluster_examples = config["cluster_examples"]
        all_examples = config["all_examples"]
        experiments = build_experiments_for_dataset(dataset_label, cluster_examples, all_examples)
        df = load_dataset(config["path"], sample_size=sample_size)
        print(f"\nDataset loaded for {dataset_label.upper()}: {len(df)} rows")
        for experiment in experiments:
            experiment_id = experiment["experiment_id"]
            prompt_builder = experiment["prompt_builder"]
            temp_tester = PromptExperimentTesterUnified(API_KEY, experiment_id, prompt_builder)
            model_name = temp_tester.model_name
            ctx_suffix = _model_folder_suffix(model_name)
            if experiment_already_exists(dataset_label, experiment_id, model_name, f"context with gloss {ctx_suffix}"):
                print(f"\n⏭️  SKIPPING context experiment: {experiment_id}")
                continue
            print(f"\n==== Running context experiment: {experiment_id} ====")
            tester = PromptExperimentTesterUnified(API_KEY, experiment_id, prompt_builder)
            results = tester.test_model(df)
            if results:
                tester.save_results(results, dataset_label=dataset_label)
            time.sleep(1.0)


def run_no_context_experiments(sample_size: int = SAMPLE_SIZE) -> None:
    print("=" * 80)
    print("OPENAI NO-CONTEXT EXPERIMENTS (Arabic + English)")
    print("=" * 80)
    dataset_configs = prepare_dataset_configs()
    for dataset_label, config in dataset_configs.items():
        cluster_examples = config["cluster_examples"]
        all_examples = config["all_examples"]
        experiments = build_no_context_experiments_for_dataset(dataset_label, cluster_examples, all_examples)
        df = load_dataset(config["path"], sample_size=sample_size)
        print(f"\nDataset loaded for {dataset_label.upper()} (no context): {len(df)} rows")
        for experiment in experiments:
            experiment_id = experiment["experiment_id"]
            prompt_builder = experiment["prompt_builder"]
            temp_tester = OpenAINoContextPromptTester(API_KEY, experiment_id, prompt_builder)
            model_name = temp_tester.model_name
            nc_suffix = _model_folder_suffix(model_name)
            if experiment_already_exists(dataset_label, experiment_id, model_name, f"no context with gloss {nc_suffix}"):
                print(f"\n⏭️  SKIPPING no-context experiment: {experiment_id}")
                continue
            print(f"\n==== Running no-context experiment: {experiment_id} ({dataset_label}) ====")
            tester = OpenAINoContextPromptTester(API_KEY, experiment_id, prompt_builder)
            results = tester.test_model(df)
            if results:
                tester.save_results(results, dataset_label=dataset_label)
            time.sleep(1.0)


def main() -> None:
    """Default: run with gloss only. Use run_openai_gpt4o.py / run_openai_gpt5.py for full with+without gloss."""
    print("=" * 80)
    print("OPENAI EXPERIMENT RUNNER (no-context + context, with gloss)")
    print("=" * 80)
    run_no_context_experiments()
    run_with_context_experiments()


if __name__ == "__main__":
    main()
