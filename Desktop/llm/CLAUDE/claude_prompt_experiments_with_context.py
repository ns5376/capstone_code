import os
import re
import sys
import time
from datetime import datetime
from typing import Callable, Dict, List

import pandas as pd
from anthropic import Anthropic

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from shared.gloss import set_word_list_path, get_lemma_and_gloss, strip_pos_from_gloss
import shared.gloss as _shared_gloss
from shared.text_utils import normalize_text, normalize_script, normalize_proficiency
from shared.normalization import normalize_arabic as normalize_arabic_fn
from shared.prompts import (
    load_examples,
    build_cluster_examples,
    build_zero_shot_prompt,
    build_example_prompt,
    build_zero_shot_prompt_no_context,
    build_example_prompt_no_context,
)

set_word_list_path(os.path.join(PARENT_DIR, "Word Lists.xlsx"))


class ClaudeArabicTop10Tester:
    """Minimal base tester for Claude top-10 predictions (inlined for a single-runner file)."""

    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        # Latest Claude Sonnet by default; overridable via env
        self.model_name = os.getenv("CLAUDE_MODEL_NAME", "claude-sonnet-4-20250514")
        print(f"✓ Claude tester initialized with {self.model_name}")

    def parse_top10_response(self, response: str) -> list[str]:
        """Extract up to 10 predictions from model response, keeping all content."""
        lines = response.strip().split("\n")
        words: list[str] = []
        for line in lines:
            match = re.match(r"^\d+[\.\)]?\s*(.+)$", line.strip())
            if match:
                text = match.group(1).strip()
                words.append(text)
            elif line.strip():
                words.append(line.strip())
        return words[:10]

    def query_claude_top10(self, prompt: str) -> list[str]:
        """Query Claude and return up to 10 predicted words."""
        start_time = time.time()
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=300,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}],
                timeout=60.0,
            )
            elapsed = time.time() - start_time
            text_output = response.content[0].text.strip()
            return self.parse_top10_response(text_output)
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"⚠️ API Error after {elapsed:.1f}s: {e}", flush=True)
            # Preserve contract: always return 10 entries (may be \"ERROR\")
            return ["ERROR"] * 10


def load_dataset(file_path: str, sample_size: int = 100) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return df


class PromptExperimentTesterWithContext(ClaudeArabicTop10Tester):
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
        """Run top-10 prediction test sequentially (no parallel processing)."""
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
            meaning_only = strip_pos_from_gloss(gloss_info.get("gloss", ""))

            prompt = self.make_prompt(transcription, script, native_language, proficiency, meaning_only)
            predictions = self.query_claude_top10(prompt)

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

        return results

    def save_results(self, results: List[Dict[str, object]], dataset_label: str):
        df = pd.DataFrame(results)
        
        # Add normalized column (use shared normalization)
        if 'true_word' in df.columns:
            df['true_word_normalized'] = df['true_word'].apply(normalize_arabic_fn)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine folder based on experiment_id and whether gloss is disabled (no-gloss patches get_lemma_and_gloss)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        test_word = "أَبَوِيّ"
        test_result = get_lemma_and_gloss(test_word)
        gloss_disabled = test_result == {}
        
        # Route outputs to the correct folders:
        #   - WITH GLOSS   → "<lang> (no) context with gloss"
        #   - WITHOUT GLOSS → "<lang> (no) context without gloss"
        if "no_context" in self.experiment_id:
            if gloss_disabled:
                folder = os.path.join(base_dir, f"{dataset_label} no context without gloss")
            else:
                folder = os.path.join(base_dir, f"{dataset_label} no context with gloss")
        else:
            if gloss_disabled:
                folder = os.path.join(base_dir, f"{dataset_label} context without gloss")
            else:
                folder = os.path.join(base_dir, f"{dataset_label} context with gloss")
        os.makedirs(folder, exist_ok=True)
        
        excel_name = os.path.join(folder, f"{dataset_label}_top10_predictions_{self.experiment_id}_{self.model_name}_{timestamp}.xlsx")
        df.to_excel(excel_name, index=False, engine='openpyxl')
        
        print("\n✅ Results saved to:")
        print(f"- {excel_name}")
        
        return excel_name


API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set. Please set it before running.")
# Paths relative to project root (parent of CLAUDE/); works when repo is cloned anywhere
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

        # Add normalized column (use shared normalization)
        if "true_word" in df.columns:
            df["true_word_normalized"] = df["true_word"].apply(normalize_arabic_fn)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        test_word = "أَبَوِيّ"
        test_result = get_lemma_and_gloss(test_word)
        gloss_disabled = test_result == {}

        # Match the folder naming that _experiment_already_run and analyze_results expect:
        #   WITH GLOSS:    "<dataset> (no) context with gloss"
        #   WITHOUT GLOSS: "<dataset> (no) context without gloss"
        if "no_context" in self.experiment_id:
            if gloss_disabled:
                folder = os.path.join(base_dir, f"{dataset_label} no context without gloss")
            else:
                folder = os.path.join(base_dir, f"{dataset_label} no context with gloss")
        else:
            if gloss_disabled:
                folder = os.path.join(base_dir, f"{dataset_label} context without gloss")
            else:
                folder = os.path.join(base_dir, f"{dataset_label} context with gloss")
        os.makedirs(folder, exist_ok=True)

        excel_name = os.path.join(
            folder,
            f"{dataset_label}_top10_predictions_{self.experiment_id}_{self.model_name}_{timestamp}.xlsx",
        )
        df.to_excel(excel_name, index=False, engine="openpyxl")

        print("\n✅ Results saved to:")
        print(f"- {excel_name}")

        return excel_name


def _experiment_already_run(dataset_label: str, experiment_id: str, check_no_gloss: bool = False) -> bool:
    """Check if an experiment has already been run by looking for existing files."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if gloss is disabled using the module-level flag (set by no-gloss script)
    # or by testing the function directly
    current_module = sys.modules[__name__]
    gloss_disabled = getattr(current_module, '_gloss_disabled_flag', False)
    if not gloss_disabled:
        # Fallback: test the function to see if it's been monkey-patched
        test_word = "أَبَوِيّ"  # This word exists in the word list
        current_get_gloss = current_module.__dict__.get('get_lemma_and_gloss', get_lemma_and_gloss)
        test_result = current_get_gloss(test_word)
        gloss_disabled = test_result == {}  # If monkey-patched, always returns {}
    
    # Only check the appropriate folder based on whether gloss is enabled
    # When running WITH GLOSS experiments, only check "with gloss" folders
    # When running WITHOUT GLOSS experiments, only check the new \"without gloss\" folders
    folders_to_check = []
    if "no_context" in experiment_id:
        if gloss_disabled:
            folders_to_check.append(os.path.join(base_dir, f"{dataset_label} no context without gloss"))
        else:
            folders_to_check.append(os.path.join(base_dir, f"{dataset_label} no context with gloss"))
    else:
        if gloss_disabled:
            folders_to_check.append(os.path.join(base_dir, f"{dataset_label} context without gloss"))
        else:
            folders_to_check.append(os.path.join(base_dir, f"{dataset_label} context with gloss"))
    
    import glob
    pattern = f"{dataset_label}_top10_predictions_{experiment_id}_*.xlsx"
    for folder in folders_to_check:
        if os.path.exists(folder):
            existing_files = glob.glob(os.path.join(folder, pattern))
            if len(existing_files) > 0:
                return True
    
    return False


def run_with_context_experiments(sample_size: int = 100) -> None:
    """Run all WITH-CONTEXT experiments (zero-shot + clusters + all-examples)."""
    dataset_configs = prepare_dataset_configs()

    for dataset_label, config in dataset_configs.items():
        cluster_examples = config["cluster_examples"]
        all_examples = config["all_examples"]

        experiments = build_experiments_for_dataset(dataset_label, cluster_examples, all_examples)

        dataset_path = config["path"]
        df = load_dataset(dataset_path, sample_size=sample_size)
        print(f"\nDataset loaded for {dataset_label.upper()}: {len(df)} rows")

        for experiment in experiments:
            experiment_id = experiment["experiment_id"]

            # Skip if experiment already run (checks appropriate gloss/without-gloss folders)
            if _experiment_already_run(dataset_label, experiment_id, check_no_gloss=False):
                print(f"\n⏭️  Skipping {experiment_id} - already completed")
                continue

            prompt_builder = experiment["prompt_builder"]

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

            tester.save_results(results, dataset_label=dataset_label)
            time.sleep(1.0)


def run_no_context_experiments(sample_size: int = 100) -> None:
    """
    Run all NO-CONTEXT experiments (zero-shot + 7 clusters + all-examples).

    This reuses the same runner by temporarily swapping the experiment builder
    to emit 'no_context_*' experiment IDs.
    """
    dataset_configs = prepare_dataset_configs()

    # Temporarily modify build_experiments_for_dataset to generate no-context experiments
    original_build = build_experiments_for_dataset

    def build_no_context_experiments(dataset_label: str, cluster_examples: Dict[str, Dict[str, object]], all_examples: List[Dict[str, str]]):
        experiments: List[Dict[str, object]] = []

        # Zero-shot
        experiments.append(
            {
                "experiment_id": f"no_context_zero_shot_{dataset_label}",
                "prompt_builder": build_zero_shot_prompt_no_context,
            }
        )

        # Cluster experiments
        for slug, cluster_info in cluster_examples.items():
            experiments.append(
                {
                    "experiment_id": f"no_context_{slug}_{dataset_label}",
                    "prompt_builder": build_example_prompt_no_context(
                        cluster_info["examples"],
                        include_gloss_in_current_query=True,
                    ),
                }
            )

        # All-examples
        experiments.append(
            {
                "experiment_id": f"no_context_all_examples_{dataset_label}",
                "prompt_builder": build_example_prompt_no_context(
                    all_examples,
                    include_gloss_in_current_query=True,
                ),
            }
        )

        return experiments

    try:
        globals()["build_experiments_for_dataset"] = build_no_context_experiments  # type: ignore[assignment]
        for dataset_label, config in dataset_configs.items():
            cluster_examples = config["cluster_examples"]
            all_examples = config["all_examples"]

            experiments = build_no_context_experiments(dataset_label, cluster_examples, all_examples)

            dataset_path = config["path"]
            df = load_dataset(dataset_path, sample_size=sample_size)
            print(f"\nDataset loaded for {dataset_label.upper()} (NO CONTEXT): {len(df)} rows")

            for experiment in experiments:
                experiment_id = experiment["experiment_id"]

                if _experiment_already_run(dataset_label, experiment_id, check_no_gloss=False):
                    print(f"\n⏭️  Skipping {experiment_id} - already completed")
                    continue

                prompt_builder = experiment["prompt_builder"]

                print("\n============================")
                print(f"Running NO-CONTEXT experiment: {experiment_id}")
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

                tester.save_results(results, dataset_label=dataset_label)
                time.sleep(1.0)
    finally:
        globals()["build_experiments_for_dataset"] = original_build  # type: ignore[assignment]


if __name__ == "__main__":
    # Single entry point: run both context and no-context experiments
    # WITH GLOSS and WITHOUT GLOSS on the FULL shared test sets.
    SAMPLE_SIZE = 1_000_000  # large enough to avoid subsampling

    # --- PASS 1: WITH GLOSS ---------------------------------------------------
    print("=" * 80)
    print("CLAUDE EXPERIMENT RUNNER (no-context + context, WITH GLOSS, full test set)")
    print("=" * 80)
    run_no_context_experiments(sample_size=SAMPLE_SIZE)
    run_with_context_experiments(sample_size=SAMPLE_SIZE)

    # --- PASS 2: WITHOUT GLOSS -----------------------------------------------
    # Patch gloss lookups so prompts are built WITHOUT gloss and results go to
    # the '<dataset> (no) context without gloss' folders.
    def _no_gloss(_true_word: str) -> dict[str, str]:
        return {}

    _orig_local_get = get_lemma_and_gloss
    _orig_shared_get = _shared_gloss.get_lemma_and_gloss
    globals()["get_lemma_and_gloss"] = _no_gloss  # used inside this module
    _shared_gloss.get_lemma_and_gloss = _no_gloss  # used by shared prompts/gloss

    try:
        print("\n" + "=" * 80)
        print("CLAUDE EXPERIMENT RUNNER (no-context + context, WITHOUT GLOSS, full test set)")
        print("=" * 80)
        run_no_context_experiments(sample_size=SAMPLE_SIZE)
        run_with_context_experiments(sample_size=SAMPLE_SIZE)
    finally:
        # Restore original gloss behavior
        globals()["get_lemma_and_gloss"] = _orig_local_get
        _shared_gloss.get_lemma_and_gloss = _orig_shared_get

