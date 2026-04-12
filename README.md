# When LLMs Evaluate LLMs in Hiring
### A Multi-Model Auditing Framework for Uncovering Trends in Preferences

This repository contains the full replication code for the bachelor's thesis investigating systematic biases in LLM-based hiring pipelines. The framework simulates a closed-loop hiring environment in which distinct LLM **Writer Agents** generate cover letters and distinct LLM **Evaluator Agents** score candidates — enabling a systematic audit of inter-annotator agreement, self-preference bias, and rank displacement (the "Leapfrog Effect").

---

## Pipeline Overview

```
dataset/
├── jobs/                  ← Job descriptions (one .txt per job)
└── resumes/               ← Resumes per job (job_{id}_{title}/)

get_data.py                ← Step 0: Collect and stratify dataset
cover_letter_generation.py ← Step 1: Generate cover letters (8 writer models)
cover_letter_evaluation.py ← Step 2: Evaluate candidates (6 evaluator models)
check_variance.py          ← Step 3: Intra-model variance check
basic_analysis.py          ← Step 4: Core analysis & plots
advanced_analysis.py       ← Step 5: Advanced analysis & plots
competitive_advantage_plots.py ← Step 6: Leapfrog / NCA plots
sankey_plots.py            ← Step 7: Rank displacement visualizations
```

---

## Setup

### 1. Install Dependencies

```bash
pip install openai anthropic google-genai ollama pandas numpy matplotlib seaborn scipy
```

For local model inference (Llama, DeepSeek-R1), install [Ollama](https://ollama.com) and pull the required models:

```bash
ollama pull llama3.1
ollama pull deepseek-r1
```

### 2. Configure API Keys

Open `cover_letter_generation.py` and `cover_letter_evaluation.py` and fill in your API keys:

```python
OPENAI_API_KEY    = "your-key-here"
ANTHROPIC_API_KEY = "your-key-here"
GEMINI_API_KEY    = "your-key-here"
DEEPSEEK_API_KEY  = "your-key-here"
```

> ⚠️ Do not commit API keys to version control. Consider using environment variables or a `.env` file.

---

## Step-by-Step Execution

### Step 0 — Collect Dataset (`get_data.py`)

Reads job descriptions and resumes from the Kaggle datasets, computes cosine similarity scores using `all-MiniLM-L6-v2`, and writes stratified candidate pools to disk.

```bash
python get_data.py
```

**Output:** `dataset/jobs/*.txt` and `dataset/resumes/job_{id}_{title}/*.txt`

The two source datasets are:
- [Job Listings Dataset](https://www.kaggle.com/datasets/hammadfarooq470/job-listings-dataset-for-data-analysis-and-nlp) (Hammad, n.d.)
- [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) (Snehaan, 2021)

Download both CSVs and place them at `../../dataset/job_desc_data.csv` and `../../dataset/resume_data.csv` relative to the script (or adjust the paths in the script).

---

### Step 1 — Generate Cover Letters (`cover_letter_generation.py`)

Runs 8 Writer Agents in parallel to generate tailored cover letters for every candidate × job combination.

```bash
python cover_letter_generation.py
```

**Output:** `output_cl/{job_folder}/{writer_model}_cover_letter_cv{n}.txt`

- Skips files that already exist, so the script is safe to resume after interruptions.
- DeepSeek-R1 `<think>` tags are automatically stripped from outputs.

**Writer models:** `gpt-4o-mini`, `gpt-5-mini`, `gemini-2.0-flash`, `gemini-3-flash-preview`, `deepseek-chat`, `deepseek-r1` (local), `llama3.1` (local), `claude-haiku-4-5`

---

### Step 2 — Evaluate Candidates (`cover_letter_evaluation.py`)

Runs 6 Evaluator Agents across three evaluation modes, repeated 4 times each for variance estimation.

```bash
python cover_letter_evaluation.py
```

**Output:** `output_eval/{job_folder}/run_{1-4}/{eval_type}/*.txt`

Evaluation modes:
| Mode | Description |
|---|---|
| `cv_only` | Evaluator sees only the raw resume |
| `cl_evaluations` | Evaluator sees only the cover letter |
| `cv_cl_evaluations` | Evaluator sees both CV and cover letter |

**Evaluator models:** `gpt-4o-mini`, `gpt-5-mini`, `gemini-2.0-flash`, `gemini-3-flash-preview`, `deepseek-chat`, `claude-haiku-4-5`

Total evaluations: ~204,000

---

### Step 3 — Check Variance (`check_variance.py`)

Calculates intra-model standard deviation across the 4 evaluation runs to validate consistency before analysis.

```bash
python check_variance.py
```

**Output:** Printed table of average standard deviation per evaluator model.

---

### Step 4–7 — Analysis & Visualization

Run these scripts in any order once Steps 1–3 are complete:

```bash
python basic_analysis.py               # Score distributions, strictness evolution
python advanced_analysis.py            # Inter-annotator agreement, head-to-head win rates
python competitive_advantage_plots.py  # Self-preference heatmaps, NCA matrices
python sankey_plots.py                 # Rank displacement / Leapfrog visualizations
```

**Output:** `output_plots/` subdirectories with PNG figures.

---

## Output Directory Structure

```
output_cl/
└── job_{id}_{title}/
    └── {writer_model}_cover_letter_cv{n}.txt

output_eval/
└── job_{id}_{title}/
    └── run_{1..4}/
        ├── cv_only/
        ├── cl_evaluations/
        └── cv_cl_evaluations/

output_plots/
├── basic_analysis/
├── advanced_analysis/
├── competitive_advantage/
└── sankey/
```

---

## Key Concepts

- **Incumbent** — A Top-25 candidate (high cosine similarity to job description)
- **Challenger** — A Lower-25 candidate (cosine similarity ≤ 0.4)
- **Leapfrog Event** — A Challenger rises into the Top 25 after cover letter inclusion, displacing an Incumbent
- **Net Competitive Advantage (NCA)** — Raw leapfrog rate minus the same-model baseline noise rate
- **Self-Preference Bias (∆self)** — Score premium an evaluator awards to text generated by its own underlying architecture

---

## Notes

- All analysis scripts build the master DataFrame by scanning `output_eval/` at runtime — no separate data export step is needed.
- The `check_variance.py` script uses the same ingestion logic as the analysis scripts and can be run at any point after Step 2.
- Parallelism is controlled by `MAX_WORKERS` at the top of the generation and evaluation scripts. Reduce this value if you hit rate limits.
- Gemini and Claude models include exponential backoff retry logic for rate limit errors (429/529).
