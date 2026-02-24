# LLM Experiment Runners (CLAUDE, GEMINI, OPENAI, FANAR)

Run Arabic top-10 prediction experiments across multiple LLM APIs (with/without gloss, context/no-context) and produce analysis Excel files.

## Quick start

1. **Clone this repo** — the folder will be named `capstone_experiments`. `cd` into it.
2. **Add data** (see below).
3. **Install deps:** `pip install -r requirements.txt`
4. **Set API keys** in a `.env` file or environment (see [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md)).
5. **Run experiments and analysis** as in [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md).

## Data you must provide

- **`shared_data/DATASET_NEW/ARABIC_TEST_FULL.xlsx`** and **`ENGLISH_TEST_FULL.xlsx`**
- **`shared_data/examples/arabic_examples.json`** and **`english_examples.json`** (included in repo if present)
- **`Word Lists.xlsx`** in this folder (parent of `CLAUDE/`, `GEMINI/`, etc.) for gloss experiments

## Push to GitHub

1. Create a new repository on GitHub (empty, no README).
2. Commit and push from the project root (e.g. your local folder; when someone clones the repo this is `capstone_experiments`):

```bash
cd /path/to/capstone_experiments
git add .gitignore README.md RUN_EXPERIMENTS.md requirements.txt shared/ shared_data/
git add CLAUDE/claude_prompt_experiments_with_context.py CLAUDE/analyze_results.py
git add GEMINI/gemini_prompt_experiments_with_context.py GEMINI/analyze_results.py
git add OPENAI/openai_prompt_experiments_with_context.py OPENAI/run_openai_gpt4o.py OPENAI/run_openai_gpt5.py
git add OPENAI/analyze_results_openai_gpt4o.py OPENAI/analyze_results_openai_gpt5.py OPENAI/.env.example
git add FANAR/fanar_prompt_experiments_with_context.py FANAR/analyze_results.py
git commit -m "Minimal experiment runners: CLAUDE, GEMINI, OPENAI, FANAR"
git push experiments main
```

Repo: **https://github.com/ns5376/capstone_experiments**

To push to this repo (one-time remote setup):

```bash
git remote add experiments https://github.com/ns5376/capstone_experiments.git
git push -u experiments main
```

If you prefer to make `capstone_experiments` the default remote:

```bash
git remote set-url origin https://github.com/ns5376/capstone_experiments.git
git push -u origin main
```

## Full instructions

See **[RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md)** for:

- Exact code and data to give someone
- Environment variables and API keys
- Commands to run each system (CLAUDE, GEMINI, OPENAI GPT-4o/GPT-5, FANAR)
- Commands to run analysis and get the Excel outputs

## Repo layout (minimal)

When you clone, the top-level folder is **capstone_experiments**. Full directory tree:

```
capstone_experiments/                    # project root (name can be anything)
├── .env                                 # you create; API keys (not in repo)
├── .gitignore
├── README.md
├── RUN_EXPERIMENTS.md
├── requirements.txt
├── Word Lists.xlsx                      # you add
├── shared/
│   ├── __init__.py
│   ├── gloss.py
│   ├── prompts.py
│   ├── text_utils.py
│   ├── normalization.py
│   └── analysis_utils.py
├── shared_data/
│   ├── DATASET_NEW/
│   │   ├── ARABIC_TEST_FULL.xlsx        # you add
│   │   └── ENGLISH_TEST_FULL.xlsx      # you add
│   └── examples/
│       ├── arabic_examples.json
│       └── english_examples.json
├── CLAUDE/
│   ├── claude_prompt_experiments_with_context.py
│   └── analyze_results.py
├── GEMINI/
│   ├── gemini_prompt_experiments_with_context.py
│   └── analyze_results.py
├── OPENAI/
│   ├── openai_prompt_experiments_with_context.py
│   ├── run_openai_gpt4o.py
│   ├── run_openai_gpt5.py
│   ├── analyze_results_openai_gpt4o.py
│   ├── analyze_results_openai_gpt5.py
│   └── .env.example
└── FANAR/
    ├── fanar_prompt_experiments_with_context.py
    └── analyze_results.py
```
