# Email Generation Assistant

An LLM-powered assistant that generates professional emails from structured inputs
(**intent**, **key facts**, and **tone**), then **evaluates** the output with three
custom metrics and **compares** two prompting strategies (Few-Shot vs. Chain-of-Thought)
to recommend the best one for production.

The project is fully runnable end-to-end with a single command, supports an offline
**mock mode** (no API key required), and produces a structured evaluation report.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [The Three Custom Metrics](#the-three-custom-metrics)
- [Setup](#setup)
- [Running the Project](#running-the-project)
- [Runtime &amp; Model Selection](#runtime--model-selection)
- [Output Files](#output-files)
- [Configuration Reference](#configuration-reference)
- [How This Maps to the Assessment](#how-this-maps-to-the-assessment)
- [License](#license)

---

## Features

- **Structured email generation** from three inputs: intent, key facts, and tone.
- **Two advanced prompting strategies**, both documented in `prompts.py`:
  - **Few-Shot** — provides high-quality input&rarr;email examples so the model mimics
    structure, tone, and fact integration.
  - **Chain-of-Thought (CoT)** — instructs the model to reason step-by-step (audience,
    structure, fact checklist, tone check) inside `<reasoning>` tags and emit the final
    email inside `<email>` tags, which the assistant parses out.
- **Three custom evaluation metrics** combining automated Python checks and LLM-as-a-judge.
- **Automated model/strategy comparison** over 10 diverse scenarios with hand-written
  reference emails.
- **Self-generating report** (`analysis_report.md`) whose narrative is derived from the
  actual computed scores, plus machine-readable `results.json` / `results.csv`.
- **Mock mode** to run the entire pipeline offline with zero API cost.
- **Robustness**: retry with exponential backoff on API errors, defensive JSON parsing of
  judge responses, and safe fallbacks so a single bad response never crashes a run.

---

## Project Structure

```text
email-generation-assistant/
├── assistant.py         # Email generation (Few-Shot & CoT) + shared OpenAI client, mock mode
├── prompts.py           # Few-Shot and Chain-of-Thought prompt templates
├── metrics.py           # The 3 custom evaluation metrics
├── config.py            # Central config: models, API settings, retries, temperatures
├── scenarios.json       # 10 test scenarios + human reference emails
├── run_evaluation.py    # Orchestrates generation, scoring, and report building
├── requirements.txt     # Python dependencies
├── .env.example         # Template for your API key
├── .gitignore           # Ignores .env, __pycache__, venv, etc.
├── LICENSE              # MIT License
├── README.md            # This file
│
│  # Generated when you run the evaluation:
├── results.csv          # Flat per-scenario raw scores + generated emails
├── results.json         # Metric definitions, averages, and full nested results
└── analysis_report.md   # Consolidated report (summary, analysis, prompts, metrics, raw data)
```

---

## How It Works

1. **Generation** — `assistant.generate_email(intent, key_facts, tone, model, strategy)`
   builds a prompt from the chosen strategy and calls the model through an
   OpenAI-compatible client (OpenRouter by default).
2. **Evaluation** — each generated email is scored by the three metrics in `metrics.py`.
3. **Comparison** — `run_evaluation.py` runs all 10 scenarios through **both** strategies
   (Config A: Few-Shot, Config B: Chain-of-Thought), aggregates the scores, and writes the
   results plus a comparative analysis.

Both strategies use the **same model** by default, so the comparison isolates the effect of
the *prompting technique*. (You can also point each config at a different model — see
[Configuration Reference](#configuration-reference).)

---

## The Three Custom Metrics

All metrics are scored on a **0&ndash;100** scale (higher is better). Judge calls use
`temperature=0` for reproducibility.

### 1. Fact Recall &mdash; *automated, LLM-as-a-judge*
Measures how many of the scenario's key facts actually appear in the email. An LLM judge
marks each fact `yes`/`no`.
> **Score = (facts present / total facts) &times; 100**

### 2. Tone Accuracy &mdash; *LLM-as-a-judge*
An LLM judge rates how well the email matches the requested tone on a 1&ndash;5 scale and
returns strict JSON (`{"score": ..., "reason": ...}`).
> **Score = (rating &minus; 1) / 4 &times; 100**

### 3. Conciseness &amp; Fluency &mdash; *hybrid Python + LLM*
Averages a Python readability/length component with an LLM grammar/fluency rating.
- **Word-count score**: 100 if 80&ndash;220 words; otherwise penalized.
- **Sentence-length score**: 100 if avg sentence length is 10&ndash;20 words; otherwise penalized.
- **Python score** = average of the two above.
- **LLM fluency** rated 1&ndash;5, normalized to 0&ndash;100.
> **Final Score = (Python score + Normalized fluency) / 2**

Full definitions/logic are stored in `metrics.METRIC_DEFINITIONS` and reproduced in the
generated report.

---

## Setup

### 1. Create and activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API key
This project uses **OpenRouter** (OpenAI-compatible) and its **free** models by default.
Get a free key at <https://openrouter.ai/keys> (no credit card required), then:
```bash
cp .env.example .env
```
Edit `.env` and set:
```text
OPENROUTER_API_KEY=your_actual_key_here
```

---

## Running the Project

### Live mode (calls the model; requires the API key)
```bash
python run_evaluation.py
```

### Mock mode (no API key, no network, no cost)
Use this to verify the whole pipeline before spending any quota:
```bash
python run_evaluation.py --mock
```

Both modes generate `results.csv`, `results.json`, and `analysis_report.md`, and print a
summary table to the console.

---

## Runtime &amp; Model Selection

> ⏱️ **Heads-up on runtime:** A full live run with the default model
> **`openai/gpt-oss-120b:free`** takes **more than 10 minutes**. This is expected — free
> models are subject to rate limits, and each run makes many calls (email generation plus
> several LLM-as-a-judge calls per scenario, across 10 scenarios and 2 strategies).

**You can choose a different model** to suit your needs:

- **Faster / higher quality:** switch to another **free** model (e.g. a smaller/faster
  OpenRouter free model) or a **paid** model. Paid or lighter models typically respond
  faster and may improve output quality, at the cost of either money or, sometimes, accuracy.
- **How to change it:** edit the model constants in **`config.py`**:
  ```python
  DEFAULT_MODEL = "openai/gpt-oss-120b:free"   # model used to generate emails
  JUDGE_MODEL   = "openai/gpt-oss-120b:free"   # model used by the judge metrics
  ```
  Any OpenRouter model ID works. To compare **two different models** instead of two
  prompting strategies, point `CONFIG_A` and `CONFIG_B` in `run_evaluation.py` at different
  model names.

Tip: run `--mock` first (finishes in seconds) to confirm everything is wired up, then do the
longer live run once.

---

## Output Files

| File | Description |
|------|-------------|
| `results.csv` | Flat table: one row per scenario &times; strategy, with all raw metric scores and the generated email. |
| `results.json` | Structured output: the 3 metric definitions, per-config averages, and full nested per-scenario results. |
| `analysis_report.md` | Consolidated report: (1) summary table, (2) data-driven comparative analysis, (3) prompt templates, (4) metric definitions, (5) raw scenario scores. |

---

## Configuration Reference

All tunables live in **`config.py`**:

| Constant | Purpose | Default |
|----------|---------|---------|
| `DEFAULT_MODEL` | Model used to generate emails | `openai/gpt-oss-120b:free` |
| `JUDGE_MODEL` | Model used by the LLM-as-a-judge metrics | `openai/gpt-oss-120b:free` |
| `BASE_URL` | OpenAI-compatible API endpoint | OpenRouter |
| `MAX_TOKENS` | Max tokens per completion | `1000` |
| `MAX_RETRIES` | Retry attempts on transient API errors | `3` |
| `FEW_SHOT_TEMP` | Temperature for Few-Shot generation | `0.7` |
| `COT_TEMP` | Temperature for Chain-of-Thought generation | `0.4` |
| `JUDGE_TEMP` | Temperature for judge calls (deterministic) | `0.0` |

---

## How This Maps to the Assessment

| Requirement | Where it lives |
|-------------|----------------|
| Email assistant (intent, facts, tone) | `assistant.py` |
| Advanced prompting technique(s) | `prompts.py` (Few-Shot + Chain-of-Thought) |
| 10 scenarios + human reference emails | `scenarios.json` |
| 3 custom metrics (defined &amp; implemented) | `metrics.py` |
| Structured evaluation output (CSV/JSON) | `results.csv`, `results.json` |
| Two-model/strategy comparison + analysis | `run_evaluation.py`, `analysis_report.md` |
| Consolidated final report | `analysis_report.md` (and the exported PDF) |

---

## License

Released under the [MIT License](LICENSE).
