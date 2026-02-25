"""
Run all OPENAI experiments for GPT-5: no-context + context, with gloss then without gloss.
Set OPENAI_MODEL_NAME=gpt-5 before loading the experiment module. Full test set, sequential.
"""
import os
import sys

os.environ["OPENAI_MODEL_NAME"] = "gpt-5"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CURRENT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import openai_prompt_experiments as exp
import shared.gloss as shared_gloss

SAMPLE = exp.SAMPLE_SIZE


def _no_gloss(_word: str) -> dict:
    return {}


def _patch_save_results_to_without_gloss():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    def _save_context_no_gloss(self, results, dataset_label):
        import pandas as pd
        from datetime import datetime
        df = pd.DataFrame(results)
        if "true_word" in df.columns:
            df["true_word_normalized"] = df["true_word"].apply(exp._normalize_arabic_cell)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = exp._model_folder_suffix(self.model_name)
        folder = os.path.join(base_dir, f"{dataset_label} context without gloss {suffix}")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(
            folder,
            f"{dataset_label}_top10_predictions_{self.experiment_id}_{self.model_name}_{timestamp}.xlsx",
        )
        df.to_excel(path, index=False, engine="openpyxl")
        print(f"\n✅ Results saved to: {path}")
        return path

    def _save_no_context_no_gloss(self, results, dataset_label):
        import pandas as pd
        from datetime import datetime
        df = pd.DataFrame(results)
        if "true_word" in df.columns:
            df["true_word_normalized"] = df["true_word"].apply(exp._normalize_arabic_cell)
        if "prediction" in df.columns:
            df["prediction_normalized"] = df["prediction"].apply(exp._normalize_arabic_cell)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = exp._model_folder_suffix(self.model_name)
        folder = os.path.join(base_dir, f"{dataset_label} no context without gloss {suffix}")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(
            folder,
            f"{dataset_label}_top10_predictions_{self.experiment_id}_{self.model_name}_{timestamp}.xlsx",
        )
        df.to_excel(path, index=False, engine="openpyxl")
        print(f"\n✅ Results saved to: {path}")
        return path

    exp.PromptExperimentTesterUnified.save_results = _save_context_no_gloss
    exp.OpenAINoContextPromptTester.save_results = _save_no_context_no_gloss


def _patch_experiment_already_exists_for_no_gloss():
    _orig = exp.experiment_already_exists

    def _wrapper(dataset_label: str, experiment_id: str, model_name: str, folder_suffix: str) -> bool:
        no_gloss_suffix = folder_suffix.replace("with gloss", "without gloss")
        return _orig(dataset_label, experiment_id, model_name, no_gloss_suffix)

    exp.experiment_already_exists = _wrapper


def main():
    print("=" * 80)
    print("OPENAI GPT-5: Running WITH GLOSS (no-context + context)")
    print("=" * 80)
    exp.run_no_context_experiments(sample_size=SAMPLE)
    exp.run_with_context_experiments(sample_size=SAMPLE)

    print("\n" + "=" * 80)
    print("OPENAI GPT-5: Running WITHOUT GLOSS (no-context + context)")
    print("=" * 80)
    shared_gloss.get_lemma_and_gloss = _no_gloss
    exp.get_lemma_and_gloss = _no_gloss
    _patch_save_results_to_without_gloss()
    _patch_experiment_already_exists_for_no_gloss()
    exp.run_no_context_experiments(sample_size=SAMPLE)
    exp.run_with_context_experiments(sample_size=SAMPLE)

    print("\n✅ GPT-5 experiments complete.")


if __name__ == "__main__":
    main()
