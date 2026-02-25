# How to Run All Experiments (CLAUDE, GEMINI, OPENAI, FANAR)

This guide is for someone who needs to run the full experiment suite from scratch and produce the analysis Excel files.

---

## 1. What Code to Give Them

**Clone the repo** — the folder will be named **capstone_experiments** (project root). It must contain:

### Required folders and files

| What | Where |
|------|--------|
| Shared utilities | `shared/` (gloss.py, prompts.py, text_utils.py, normalization.py, analysis_utils.py) |
| **Shared data** | `shared_data/` (see “What data to give them” below) |
| **Word list (glosses)** | `Word Lists.xlsx` (in the project root) |
| CLAUDE runner + analysis | `CLAUDE/claude_prompt_experiments.py`, `CLAUDE/analyze_results.py` |
| GEMINI runner + analysis | `GEMINI/gemini_prompt_experiments.py`, `GEMINI/analyze_results.py` |
| OPENAI runners + analysis | `OPENAI/openai_prompt_experiments.py`, `OPENAI/run_openai_gpt4o.py`, `OPENAI/run_openai_gpt5.py`, `OPENAI/analyze_results_openai_gpt4o.py`, `OPENAI/analyze_results_openai_gpt5.py` |
| FANAR runner + analysis | `FANAR/fanar_prompt_experiments.py`, `FANAR/analyze_results.py` |

You can zip the project folder (including `shared`, `shared_data`, `CLAUDE`, `GEMINI`, `OPENAI`, `FANAR`, and `Word Lists.xlsx`) and send that.

---

## 2. What Data to Give Them

Put these in the right places under the **project root** (capstone_experiments):

### 2.1 Test sets (required)

- **`shared_data/DATASET_NEW/ARABIC_TEST_FULL.xlsx`**
- **`shared_data/DATASET_NEW/ENGLISH_TEST_FULL.xlsx`**

All systems read from these two files. If the runner is on a different machine, you can either:

- Keep the same folder structure and put the two files in `shared_data/DATASET_NEW/`, or  
- Change the `DATASET_DIR` (and optionally `EXAMPLES_DIR`) inside each runner to point to the actual location (see “Path and environment” below).

### 2.2 Example sets (required for few-shot prompts)

- **`shared_data/examples/arabic_examples.json`**
- **`shared_data/examples/english_examples.json`**

CLAUDE, OPENAI, and FANAR use these from `shared_data/examples`. GEMINI is set to use the same path.

### 2.3 Word list (required for gloss)

- **`Word Lists.xlsx`**
  Must sit in the **project root** (parent of `CLAUDE`, `GEMINI`, `OPENAI`, `FANAR`). It should have a sheet named `combined` (or the first sheet is used) with word and gloss columns. Used for “with gloss” experiments.

---

## 3. Environment and API Keys

They need a Python environment (e.g. 3.10+) with dependencies installed, and API keys for each system they run.

### 3.1 Install dependencies

From the project root (the cloned folder, capstone_experiments):

```bash
pip install pandas openpyxl requests
# If using OpenAI / Claude / Gemini official SDKs:
pip install openai anthropic google-generativeai
# Optional: for .env
pip install python-dotenv
```

(Adjust to your actual `requirements.txt` if you have one.)

### 3.2 API keys and optional env vars

Set these in the shell or in a **`.env`** file in the **project root** (or in each system folder, depending on where the scripts load `.env` from):

| System | Required | Optional |
|--------|----------|----------|
| **CLAUDE** | `ANTHROPIC_API_KEY` | — |
| **GEMINI** | `GOOGLE_API_KEY` | `GEMINI_MODEL_NAME` (default: `models/gemini-2.5-flash-lite`) |
| **OPENAI** | `OPENAI_API_KEY` | `OPENAI_MODEL_NAME` — set by runners: use `run_openai_gpt4o.py` or `run_openai_gpt5.py` (they set the model) |
| **FANAR** | `FANAR_API_KEY` | `FANAR_BASE_URL` (default: `https://api.fanar.qa/v1`), `FANAR_MODEL_ID`, etc. |

Example `.env` (create in the project root or next to the scripts):

```bash
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
OPENAI_API_KEY=sk-...
FANAR_API_KEY=...
FANAR_BASE_URL=https://api.fanar.qa/v1
```

---

## 4. Path and environment (if not using your exact paths)

Scripts use **`DATASET_DIR`** and **`EXAMPLES_DIR`** relative to the project root (the folder containing `shared/`, `CLAUDE/`, etc.). When you clone, that folder is **capstone_experiments**, so paths are e.g.:

- `.../capstone_experiments/shared_data/DATASET_NEW`
- `.../capstone_experiments/shared_data/examples`

If you clone the repo to a different path, the code resolves the project root from the script location, so no change is needed as long as the layout (shared/, shared_data/, CLAUDE/, etc.) is preserved. To use a custom base path, set **`DATASET_DIR`** and **`EXAMPLES_DIR`** in each runner, or add an env var (e.g. a project root env) and point the scripts to it.

---

## 5. How to Run the Experiments

All commands below are run from the **project root** (the folder you get when you clone — **capstone_experiments**; it contains `shared/`, `shared_data/`, `CLAUDE/`, etc.). Each runner uses the full test set and runs with-gloss then without-gloss.

### 5.1 CLAUDE

```bash
cd CLAUDE
python claude_prompt_experiments_with_context.py
```

- Runs no-context and context experiments, first with gloss, then without gloss.  
- Results are written under `CLAUDE/` into folders like `arabic context with gloss`, `english no context without gloss`, etc.

### 5.2 GEMINI

```bash
cd GEMINI
python gemini_prompt_experiments_with_context.py
```

- Same idea: no-context + context, with gloss then without gloss.  
- Results under `GEMINI/` in `... with gloss` and `... without gloss` folders.

### 5.3 OPENAI (two models: GPT-4o and GPT-5)

Run **one script per model** (from project root):

```bash
cd OPENAI
python run_openai_gpt4o.py
```

```bash
cd OPENAI
python run_openai_gpt5.py
```

- Each script runs no-context + context, with gloss then without gloss, and writes to model-specific folders (e.g. names ending with `gpt4o` or `gpt5`).

### 5.4 FANAR

```bash
cd FANAR
python fanar_prompt_experiments_with_context.py
```

- No-context + context, with gloss then without gloss.  
- Results under `FANAR/` in `... with gloss` and `... without gloss` folders.

---

## 6. How to Run the Analysis (Excel outputs)

After the experiments have produced the result folders, run the analysis script **in each system folder**. From the **project root** (capstone_experiments):

```bash
cd CLAUDE
python analyze_results.py
# -> experiment_results_analysis_claude.xlsx

cd ../GEMINI
python analyze_results.py
# -> experiment_results_analysis_gemini.xlsx

cd ../OPENAI
python analyze_results_openai_gpt4o.py
# -> experiment_results_analysis_openai_gpt4o.xlsx

python analyze_results_openai_gpt5.py
# -> experiment_results_analysis_openai_gpt5.xlsx

cd ../FANAR
python analyze_results.py
# -> experiment_results_analysis_fanar.xlsx
```

Each analysis script reads the **with-gloss** and **without-gloss** result folders for that system (and for OPENAI, for that model) and writes one Excel file with TOP-1, TOP-10, and average Levenshtein (normalized, no-norm, diacritics-only), with columns for with_gloss and without_gloss.

---

## 7. Short “copy-paste” summary for the runner

You can send them something like this:

**Code & data:**

- Clone the repo (folder will be **capstone_experiments**) or give them the project folder including:
  - `shared/`, `shared_data/`, `CLAUDE/`, `GEMINI/`, `OPENAI/`, `FANAR/`
  - `Word Lists.xlsx` in the project root
  - In `shared_data/DATASET_NEW/`: `ARABIC_TEST_FULL.xlsx`, `ENGLISH_TEST_FULL.xlsx`
  - In `shared_data/examples/`: `arabic_examples.json`, `english_examples.json`

**Setup:**

- Python 3.10+
- `pip install pandas openpyxl requests openai anthropic google-generativeai python-dotenv` (or use your project’s `requirements.txt`)
- Set API keys (see table above), e.g. in a `.env` file in the project root

**Run (from project root, capstone_experiments):**

```bash
cd CLAUDE   && python claude_prompt_experiments.py && python analyze_results.py
cd ../GEMINI && python gemini_prompt_experiments.py && python analyze_results.py
cd ../OPENAI && python run_openai_gpt4o.py && python run_openai_gpt5.py && python analyze_results_openai_gpt4o.py && python analyze_results_openai_gpt5.py
cd ../FANAR  && python fanar_prompt_experiments.py   && python analyze_results.py
```

They will get:

- `CLAUDE/experiment_results_analysis_claude.xlsx`
- `GEMINI/experiment_results_analysis_gemini.xlsx`
- `OPENAI/experiment_results_analysis_openai_gpt4o.xlsx`
- `OPENAI/experiment_results_analysis_openai_gpt5.xlsx`
- `FANAR/experiment_results_analysis_fanar.xlsx`

