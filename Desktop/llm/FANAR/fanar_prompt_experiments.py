"""
FANAR prompt experiments: no-context and context, with/without gloss.
Uses shared prompts, gloss, text_utils. One runner; main() runs with gloss then without gloss (full test set, sequential).
"""
import os
import re
import sys
import time
import glob
from datetime import datetime
from typing import Any, Callable, Dict, List

import pandas as pd
import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from shared.gloss import set_word_list_path, get_lemma_and_gloss
from shared.text_utils import normalize_text, normalize_script, normalize_proficiency
from shared.normalization import normalize_arabic
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
API_KEY = os.getenv("FANAR_API_KEY")
if not API_KEY:
    raise ValueError("FANAR_API_KEY environment variable is not set.")
BASE_URL = os.getenv("FANAR_BASE_URL", "https://api.fanar.qa/v1").rstrip("/")
# Paths relative to project root (parent of FANAR/); works when repo is cloned anywhere
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


# ---------- Inlined FanarArabicTop10Tester (from fanar_base) ----------
class FanarArabicTop10Tester:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = os.getenv("FANAR_MODEL_NAME", "FANAR C-1-8.7B")
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def _build_payload(self, prompt: str) -> Dict[str, Any]:
        model_id = os.getenv("FANAR_MODEL_ID", "Fanar-C-1-8.7B")
        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(os.getenv("FANAR_TEMPERATURE", "0")),
            "top_p": float(os.getenv("FANAR_TOP_P", "1")),
            "max_tokens": int(os.getenv("FANAR_MAX_TOKENS", "256")),
        }
        seed_env = os.getenv("FANAR_SEED")
        if seed_env is not None:
            payload["seed"] = int(seed_env)
        return payload

    def _call_fanar(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        resp = self._session.post(url, json=self._build_payload(prompt), timeout=60)
        if resp.status_code >= 400:
            try:
                err = resp.json()
                detail = str(err)
            except Exception:
                detail = resp.text[:500]
            raise requests.exceptions.HTTPError(f"HTTP {resp.status_code}: {detail}")
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            if isinstance(data, dict) and data.get("choices"):
                c = data["choices"][0]
                if isinstance(c, dict) and "text" in c:
                    return str(c["text"])
            return str(data)

    @staticmethod
    def _parse_top10_from_text(text: str) -> List[str]:
        if not text:
            return []
        candidates: List[str] = []
        for line in str(text).splitlines():
            m = re.match(r"\s*\d+\s*[\).\-\:]\s*(.+)", line)
            if m:
                w = m.group(1).strip()
                if w:
                    candidates.append(w)
        if not candidates:
            for part in re.split(r"[,;\n]", str(text)):
                w = part.strip()
                if w:
                    candidates.append(w)
        seen = set()
        unique: List[str] = []
        for w in candidates:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return unique[:10]

    def query_fanar_top10(self, prompt: str, *, max_retries: int = 3, retry_delay: float = 2.0) -> List[str]:
        for attempt in range(1, max_retries + 1):
            try:
                raw = self._call_fanar(prompt)
                return self._parse_top10_from_text(raw)
            except Exception as exc:
                print(f"⚠️ FANAR API failed (attempt {attempt}/{max_retries}): {exc}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
        return []


def load_dataset(file_path: str, sample_size: int = 100) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return df


def experiment_already_exists(dataset_label: str, experiment_id: str, folder_suffix: str) -> bool:
    folder = os.path.join(CURRENT_DIR, f"{dataset_label} {folder_suffix}")
    if not os.path.exists(folder):
        return False
    pattern = os.path.join(folder, f"{dataset_label}_top10_predictions_{experiment_id}_*.xlsx")
    for fpath in glob.glob(pattern):
        try:
            df = pd.read_excel(fpath)
            if len(df) > 0:
                return True
        except Exception:
            continue
    return False


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


def build_no_context_experiments_for_dataset(
    dataset_label: str,
    cluster_examples: Dict[str, Dict[str, object]],
    all_examples: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    """No-context experiments using builders that exclude user metadata from the prompt text."""
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


# ---------- Context tester (shared 5-arg prompt_builder) ----------
class FanarPromptExperimentTesterWithContext(FanarArabicTop10Tester):
    def __init__(self, api_key: str, base_url: str, experiment_id: str, prompt_builder: Callable[[str, str, str, str, str | None], str]):
        self.prompt_builder = prompt_builder
        self.experiment_id = experiment_id
        super().__init__(api_key=api_key, base_url=base_url)

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
            predictions = self.query_fanar_top10(prompt)
            while len(predictions) < 10:
                predictions.append("")
            predictions = predictions[:10]
            for rank, prediction in enumerate(predictions, 1):
                pred_val = "" if (prediction is None or not str(prediction).strip()) else str(prediction).strip()
                results.append({
                    "row_num": idx + 1, "word_id": row.get("word_id", ""), "user_id": row.get("user_id", ""),
                    "native_language": native_language, "proficiency_level": proficiency,
                    "script": script, "transcription": transcription, "true_word": true_word,
                    "rank": rank, "prediction": pred_val, "model": self.model_name, "prompt_id": self.experiment_id,
                })
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(rows)} examples...")
            time.sleep(1.0)
        return results

    def save_results(self, results: List[Dict[str, object]], dataset_label: str):
        df = pd.DataFrame(results)
        if "true_word" in df.columns:
            df["true_word_normalized"] = df["true_word"].apply(normalize_arabic)
        if "prediction" in df.columns:
            df["prediction_normalized"] = df["prediction"].apply(normalize_arabic)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(CURRENT_DIR, f"{dataset_label} context with gloss")
        os.makedirs(folder, exist_ok=True)
        excel_name = os.path.join(folder, f"{dataset_label}_top10_predictions_{self.experiment_id}_{self.model_name}_{timestamp}.xlsx")
        df.to_excel(excel_name, index=False, engine="openpyxl")
        print(f"\n✅ Results saved to: {excel_name}")
        return excel_name


# ---------- No-context tester (same 5-arg shared prompts as context) ----------
class FanarPromptExperimentTesterNoContext(FanarArabicTop10Tester):
    def __init__(self, api_key: str, base_url: str, experiment_id: str, prompt_builder: Callable[[str, str, str, str, str | None], str]):
        self.prompt_builder = prompt_builder
        self.experiment_id = experiment_id
        super().__init__(api_key=api_key, base_url=base_url)

    def make_prompt(self, transcription: str, script: str, native_language: str, proficiency: str, gloss: str | None) -> str:
        script_value = normalize_script(script)
        # NO-CONTEXT: do not pass user metadata (native language, proficiency) into the prompt.
        # Keep placeholder values so the shared templates render, but they are no longer
        # tied to the actual user metadata from the dataset.
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
            predictions = self.query_fanar_top10(prompt)
            while len(predictions) < 10:
                predictions.append("")
            predictions = predictions[:10]
            for rank, prediction in enumerate(predictions, 1):
                pred_val = "" if (prediction is None or not str(prediction).strip()) else str(prediction).strip()
                results.append({
                    "row_num": idx + 1, "word_id": row.get("word_id", ""), "user_id": row.get("user_id", ""),
                    "native_language": native_language, "proficiency_level": proficiency,
                    "script": script, "transcription": transcription, "true_word": true_word,
                    "rank": rank, "prediction": pred_val, "model": self.model_name, "prompt_id": self.experiment_id,
                })
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(rows)} examples...")
            time.sleep(1.0)
        return results

    def save_results(self, results: list, dataset_label: str):
        df = pd.DataFrame(results)
        if "true_word" in df.columns:
            df["true_word_normalized"] = df["true_word"].apply(normalize_arabic)
        if "prediction" in df.columns:
            df["prediction_normalized"] = df["prediction"].apply(normalize_arabic)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(CURRENT_DIR, f"{dataset_label} no context with gloss")
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
            if experiment_already_exists(dataset_label, experiment_id, "context with gloss"):
                print(f"\n⏭️  SKIPPING context experiment: {experiment_id}")
                continue
            print(f"\n==== Running FANAR context experiment: {experiment_id} ====")
            tester = FanarPromptExperimentTesterWithContext(API_KEY, BASE_URL, experiment_id, prompt_builder)
            results = tester.test_model(df)
            if results:
                tester.save_results(results, dataset_label=dataset_label)
            time.sleep(1.0)


def run_no_context_experiments(sample_size: int = SAMPLE_SIZE) -> None:
    print("=" * 80)
    print("FANAR NO-CONTEXT EXPERIMENTS (Arabic + English)")
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
            if experiment_already_exists(dataset_label, experiment_id, "no context with gloss"):
                print(f"\n⏭️  SKIPPING no-context experiment: {experiment_id}")
                continue
            print(f"\n==== Running FANAR no-context experiment: {experiment_id} ({dataset_label}) ====")
            tester = FanarPromptExperimentTesterNoContext(API_KEY, BASE_URL, experiment_id, prompt_builder)
            results = tester.test_model(df)
            if results:
                tester.save_results(results, dataset_label=dataset_label)
            time.sleep(1.0)


def main() -> None:
    """Run with gloss, then without gloss (patch get_lemma_and_gloss and save paths)."""
    import shared.gloss as shared_gloss

    def _no_gloss(_word: str) -> dict:
        return {}

    def _patch_save_to_without_gloss():
        def save_context_wo(self, results, dataset_label):
            df = pd.DataFrame(results)
            if "true_word" in df.columns:
                df["true_word_normalized"] = df["true_word"].apply(normalize_arabic)
            if "prediction" in df.columns:
                df["prediction_normalized"] = df["prediction"].apply(normalize_arabic)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = os.path.join(CURRENT_DIR, f"{dataset_label} context without gloss")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{dataset_label}_top10_predictions_{self.experiment_id}_{self.model_name}_{timestamp}.xlsx")
            df.to_excel(path, index=False, engine="openpyxl")
            print(f"\n✅ Results saved to: {path}")
            return path

        def save_no_context_wo(self, results, dataset_label):
            df = pd.DataFrame(results)
            if "true_word" in df.columns:
                df["true_word_normalized"] = df["true_word"].apply(normalize_arabic)
            if "prediction" in df.columns:
                df["prediction_normalized"] = df["prediction"].apply(normalize_arabic)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = os.path.join(CURRENT_DIR, f"{dataset_label} no context without gloss")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{dataset_label}_top10_predictions_{self.experiment_id}_{self.model_name}_{timestamp}.xlsx")
            df.to_excel(path, index=False, engine="openpyxl")
            print(f"\n✅ Results saved to: {path}")
            return path

        FanarPromptExperimentTesterWithContext.save_results = save_context_wo
        FanarPromptExperimentTesterNoContext.save_results = save_no_context_wo

    print("=" * 80)
    print("FANAR: Running WITH GLOSS (no-context + context)")
    print("=" * 80)
    run_no_context_experiments()
    run_with_context_experiments()

    print("\n" + "=" * 80)
    print("FANAR: Running WITHOUT GLOSS (no-context + context)")
    print("=" * 80)
    shared_gloss.get_lemma_and_gloss = _no_gloss
    globals()["get_lemma_and_gloss"] = _no_gloss
    _patch_save_to_without_gloss()
    _orig_exists = experiment_already_exists
    def _exists_wo(dataset_label: str, experiment_id: str, folder_suffix: str) -> bool:
        return _orig_exists(dataset_label, experiment_id, folder_suffix.replace("with gloss", "without gloss"))
    globals()["experiment_already_exists"] = _exists_wo
    run_no_context_experiments()
    run_with_context_experiments()

    print("\n✅ FANAR experiments complete.")


if __name__ == "__main__":
    main()
