"""
Evaluation Runner for Email Generation Assistant.
Loads scenarios, executes Few-Shot and Chain-of-Thought configurations,
computes metrics, saves results, and generates the comparison report.

Usage:
    python run_evaluation.py [--mock]
"""

import sys
import os

# Parse CLI arguments before importing packages that load env variables
if "--mock" in sys.argv or os.environ.get("MOCK", "").lower() == "true":
    os.environ["MOCK"] = "true"
    print("Running in MOCK mode (No API requests will be made).")
else:
    print("Running in LIVE API mode. Ensure OPENROUTER_API_KEY is configured.")

import json
import csv
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load local modules
import config
import assistant
import metrics
import prompts

# Configurations to evaluate
CONFIG_A = {
    "name": "Config A (Few-Shot)",
    "model": config.DEFAULT_MODEL,
    "strategy": "few_shot"
}

CONFIG_B = {
    "name": "Config B (Chain-of-Thought)",
    "model": config.DEFAULT_MODEL,
    "strategy": "chain_of_thought"
}

REPORT_HEADER_TEMPLATE = """# Model / Strategy Comparison & Analysis Report
**Project: Email Generation Assistant**

This report compares two email generation configurations using 10 diverse scenarios.

---

## 1. Quantitative Performance Summary

The 10 scenarios in `scenarios.json` were evaluated against the three custom metrics. Below are the final average results:

| Metric | Config A: Few-Shot | Config B: Chain-of-Thought | Difference (B - A) |
|---|---|---|---|
| **Fact Recall** | {few_shot_fact_recall:.2f}% | {cot_fact_recall:.2f}% | {diff_fact_recall:.2f}% |
| **Tone Accuracy** | {few_shot_tone_accuracy:.2f}/100 | {cot_tone_accuracy:.2f}/100 | {diff_tone_accuracy:.2f} |
| **Conciseness & Fluency** | {few_shot_conciseness_fluency:.2f}/100 | {cot_conciseness_fluency:.2f}/100 | {diff_conciseness_fluency:.2f} |
| **Overall Average** | {few_shot_overall:.2f}/100 | {cot_overall:.2f}/100 | {diff_overall:.2f} |

*Evaluation Mode: {evaluation_mode}*
*Model Used: {model_used}*

---
"""


def build_analysis_section(
    a_name: str,
    b_name: str,
    a_scores: dict,
    b_scores: dict,
) -> str:
    """
    Builds Section 2 of the report dynamically from the ACTUAL computed averages,
    so the narrative (winner, failure mode, recommendation) always matches the data.

    a_scores / b_scores are dicts with keys:
        'fact_recall', 'tone_accuracy', 'conciseness_fluency', 'overall'
    """
    # Determine overall winner / loser from the data (ties -> treat A as winner but flag it)
    a_overall = a_scores["overall"]
    b_overall = b_scores["overall"]
    tie = abs(a_overall - b_overall) < 0.01

    if a_overall >= b_overall:
        win_name, win = a_name, a_scores
        lose_name, lose = b_name, b_scores
    else:
        win_name, win = b_name, b_scores
        lose_name, lose = a_name, a_scores

    overall_gap = abs(a_overall - b_overall)

    metric_labels = {
        "fact_recall": "Fact Recall",
        "tone_accuracy": "Tone Accuracy",
        "conciseness_fluency": "Conciseness & Fluency",
    }

    def cmp_line(key: str) -> str:
        a_val = a_scores[key]
        b_val = b_scores[key]
        label = metric_labels[key]
        unit = "%" if key == "fact_recall" else "/100"
        if abs(a_val - b_val) < 0.01:
            return (f"- **{label}**: Both configurations scored equally "
                    f"({a_val:.2f}{unit}), so this metric did not differentiate them.")
        higher_name = a_name if a_val > b_val else b_name
        lower_name = b_name if a_val > b_val else a_name
        higher_val = max(a_val, b_val)
        lower_val = min(a_val, b_val)
        diff = higher_val - lower_val
        return (f"- **{label}**: {higher_name} scored {higher_val:.2f}{unit}, "
                f"{diff:.2f} points higher than {lower_name} ({lower_val:.2f}{unit}).")

    cmp_lines = "\n".join(cmp_line(k) for k in metric_labels)

    if tie:
        q1 = (f"The two configurations were effectively **tied** "
              f"({a_name}: {a_overall:.2f}/100, {b_name}: {b_overall:.2f}/100). "
              f"Neither strategy showed a meaningful overall advantage on these scenarios.\n"
              f"{cmp_lines}")
    else:
        q1 = (f"Based on the metrics above, **{win_name} ({win['overall']:.2f}/100)** "
              f"performed better overall than **{lose_name} ({lose['overall']:.2f}/100)**, "
              f"a margin of {overall_gap:.2f} points.\n{cmp_lines}")

    gaps = {k: win[k] - lose[k] for k in metric_labels}
    worst_key = max(gaps, key=gaps.get)
    worst_gap = gaps[worst_key]

    if tie:
        q2 = ("With the two configurations effectively tied, there was no single dominant "
              "failure mode. Scores across all three metrics were closely matched, indicating "
              "both strategies handled these scenarios comparably.")
    elif worst_gap <= 0.01:
        q2 = (f"The lower-performing config (**{lose_name}**) did not trail on any individual "
              f"metric by a meaningful margin; its lower overall score came from small, "
              f"consistent differences rather than one clear weakness.")
    else:
        q2 = (f"The biggest weakness of the lower-performing config (**{lose_name}**) was on "
              f"**{metric_labels[worst_key]}**, where it trailed {win_name} by {worst_gap:.2f} "
              f"points ({lose[worst_key]:.2f} vs {win[worst_key]:.2f}). "
              f"This was the metric that most dragged down its overall score.")

    if tie:
        q3 = (f"Because the two strategies scored almost identically, the recommendation should "
              f"be driven by operational cost rather than quality. **Few-Shot** is the more "
              f"economical choice for production: it uses fewer tokens and has lower latency than "
              f"Chain-of-Thought, while delivering equivalent quality on these scenarios. "
              f"Chain-of-Thought remains a strong fallback for harder, fact-dense emails where "
              f"explicit step-by-step planning may help.")
    else:
        if "Few-Shot" in win_name:
            rec_extra = ("It also has the practical advantage of lower token usage and latency "
                         "than Chain-of-Thought, making it cheaper to run at scale.")
        else:
            rec_extra = ("Although it uses more tokens and has slightly higher latency due to the "
                         "step-by-step reasoning phase, the measured quality gain justifies the cost.")
        q3 = (f"Based on this data, we recommend **{win_name}** for production. "
              f"It achieved the higher overall score ({win['overall']:.2f}/100 vs "
              f"{lose['overall']:.2f}/100, a {overall_gap:.2f}-point lead) and was at least as "
              f"strong on every individual metric that mattered. {rec_extra}")

    return (
        "## 2. Model / Strategy Comparison & Analysis\n\n"
        "### 1. Which model/strategy performed better according to the 3 custom metrics?\n"
        f"{q1}\n\n"
        "### 2. What was the biggest failure mode of the lower-performing config, based on the data?\n"
        f"{q2}\n\n"
        "### 3. Which would you recommend for production and why, justified with the metric data?\n"
        f"{q3}\n"
    )


def load_scenarios(filepath: str) -> list:
    """Loads evaluation scenarios from scenarios.json."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Scenarios file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scenarios_path = os.path.join(base_dir, "scenarios.json")
    results_json_path = os.path.join(base_dir, "results.json")
    results_csv_path = os.path.join(base_dir, "results.csv")
    report_path = os.path.join(base_dir, "analysis_report.md")

    scenarios = load_scenarios(scenarios_path)
    print(f"Loaded {len(scenarios)} scenarios.")

    evaluation_runs = []

    a_fact_recalls = []
    a_tone_accuracies = []
    a_conciseness_fluency = []

    b_fact_recalls = []
    b_tone_accuracies = []
    b_conciseness_fluency = []

    print("\n--- Running Evaluation ---")

    for scenario in tqdm(scenarios, desc="Evaluating Scenarios"):
        email_a = assistant.generate_email(
            intent=scenario["intent"],
            key_facts=scenario["key_facts"],
            tone=scenario["tone"],
            model=CONFIG_A["model"],
            strategy=CONFIG_A["strategy"]
        )

        recall_a = metrics.evaluate_fact_recall(email_a, scenario["key_facts"], model=config.JUDGE_MODEL)
        tone_a = metrics.evaluate_tone_accuracy(email_a, scenario["tone"], model=config.JUDGE_MODEL)
        cf_a = metrics.evaluate_conciseness_fluency(email_a, model=config.JUDGE_MODEL)

        a_fact_recalls.append(recall_a)
        a_tone_accuracies.append(tone_a)
        a_conciseness_fluency.append(cf_a)

        email_b = assistant.generate_email(
            intent=scenario["intent"],
            key_facts=scenario["key_facts"],
            tone=scenario["tone"],
            model=CONFIG_B["model"],
            strategy=CONFIG_B["strategy"]
        )

        recall_b = metrics.evaluate_fact_recall(email_b, scenario["key_facts"], model=config.JUDGE_MODEL)
        tone_b = metrics.evaluate_tone_accuracy(email_b, scenario["tone"], model=config.JUDGE_MODEL)
        cf_b = metrics.evaluate_conciseness_fluency(email_b, model=config.JUDGE_MODEL)

        b_fact_recalls.append(recall_b)
        b_tone_accuracies.append(tone_b)
        b_conciseness_fluency.append(cf_b)

        evaluation_runs.append({
            "scenario_id": scenario["id"],
            "intent": scenario["intent"],
            "tone": scenario["tone"],
            "key_facts": scenario["key_facts"],
            "reference_email": scenario["reference_email"],
            "runs": [
                {
                    "config_name": CONFIG_A["name"],
                    "model": CONFIG_A["model"],
                    "strategy": CONFIG_A["strategy"],
                    "generated_email": email_a,
                    "scores": {
                        "fact_recall": recall_a,
                        "tone_accuracy": tone_a,
                        "conciseness_fluency": cf_a,
                        "overall": (recall_a + tone_a + cf_a) / 3.0
                    }
                },
                {
                    "config_name": CONFIG_B["name"],
                    "model": CONFIG_B["model"],
                    "strategy": CONFIG_B["strategy"],
                    "generated_email": email_b,
                    "scores": {
                        "fact_recall": recall_b,
                        "tone_accuracy": tone_b,
                        "conciseness_fluency": cf_b,
                        "overall": (recall_b + tone_b + cf_b) / 3.0
                    }
                }
            ]
        })

    avg_a_recall = sum(a_fact_recalls) / len(scenarios)
    avg_a_tone = sum(a_tone_accuracies) / len(scenarios)
    avg_a_cf = sum(a_conciseness_fluency) / len(scenarios)
    avg_a_overall = (avg_a_recall + avg_a_tone + avg_a_cf) / 3.0

    avg_b_recall = sum(b_fact_recalls) / len(scenarios)
    avg_b_tone = sum(b_tone_accuracies) / len(scenarios)
    avg_b_cf = sum(b_conciseness_fluency) / len(scenarios)
    avg_b_overall = (avg_b_recall + avg_b_tone + avg_b_cf) / 3.0

    summary_data = [
        {
            "Config Name": CONFIG_A["name"],
            "Model": CONFIG_A["model"],
            "Strategy": CONFIG_A["strategy"],
            "Fact Recall (%)": f"{avg_a_recall:.2f}%",
            "Tone Accuracy": f"{avg_a_tone:.2f}",
            "Conciseness/Fluency": f"{avg_a_cf:.2f}",
            "Overall Score": f"{avg_a_overall:.2f}"
        },
        {
            "Config Name": CONFIG_B["name"],
            "Model": CONFIG_B["model"],
            "Strategy": CONFIG_B["strategy"],
            "Fact Recall (%)": f"{avg_b_recall:.2f}%",
            "Tone Accuracy": f"{avg_b_tone:.2f}",
            "Conciseness/Fluency": f"{avg_b_cf:.2f}",
            "Overall Score": f"{avg_b_overall:.2f}"
        }
    ]
    summary_df = pd.DataFrame(summary_data)

    print("\n========================= EVALUATION SUMMARY =========================")
    print(summary_df.to_string(index=False))
    print("======================================================================\n")

    output_json = {
        "metric_definitions": metrics.METRIC_DEFINITIONS,
        "config_averages": {
            CONFIG_A["name"]: {
                "fact_recall": avg_a_recall,
                "tone_accuracy": avg_a_tone,
                "conciseness_fluency": avg_a_cf,
                "overall": avg_a_overall
            },
            CONFIG_B["name"]: {
                "fact_recall": avg_b_recall,
                "tone_accuracy": avg_b_tone,
                "conciseness_fluency": avg_b_cf,
                "overall": avg_b_overall
            }
        },
        "scenarios_results": evaluation_runs
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)
    print(f"Saved JSON results to {results_json_path}")

    csv_headers = [
        "Scenario_ID", "Intent", "Target_Tone", "Config_Name", "Model",
        "Strategy", "Fact_Recall", "Tone_Accuracy", "Conciseness_Fluency",
        "Overall_Score", "Generated_Email"
    ]
    with open(results_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        for run in evaluation_runs:
            for conf_run in run["runs"]:
                writer.writerow([
                    run["scenario_id"],
                    run["intent"],
                    run["tone"],
                    conf_run["config_name"],
                    conf_run["model"],
                    conf_run["strategy"],
                    f"{conf_run['scores']['fact_recall']:.2f}",
                    f"{conf_run['scores']['tone_accuracy']:.2f}",
                    f"{conf_run['scores']['conciseness_fluency']:.2f}",
                    f"{conf_run['scores']['overall']:.2f}",
                    conf_run["generated_email"]
                ])
    print(f"Saved CSV results to {results_csv_path}")

    eval_mode_str = "Mock Mode (Simulated LLM Outputs)" if assistant.MOCK_MODE else "Live API Evaluation"

    report_header = REPORT_HEADER_TEMPLATE.format(
        few_shot_fact_recall=avg_a_recall,
        cot_fact_recall=avg_b_recall,
        diff_fact_recall=avg_b_recall - avg_a_recall,
        few_shot_tone_accuracy=avg_a_tone,
        cot_tone_accuracy=avg_b_tone,
        diff_tone_accuracy=avg_b_tone - avg_a_tone,
        few_shot_conciseness_fluency=avg_a_cf,
        cot_conciseness_fluency=avg_b_cf,
        diff_conciseness_fluency=avg_b_cf - avg_a_cf,
        few_shot_overall=avg_a_overall,
        cot_overall=avg_b_overall,
        diff_overall=avg_b_overall - avg_a_overall,
        evaluation_mode=eval_mode_str,
        model_used=config.DEFAULT_MODEL,
    )

    analysis_section = build_analysis_section(
        a_name=CONFIG_A["name"],
        b_name=CONFIG_B["name"],
        a_scores={
            "fact_recall": avg_a_recall,
            "tone_accuracy": avg_a_tone,
            "conciseness_fluency": avg_a_cf,
            "overall": avg_a_overall,
        },
        b_scores={
            "fact_recall": avg_b_recall,
            "tone_accuracy": avg_b_tone,
            "conciseness_fluency": avg_b_cf,
            "overall": avg_b_overall,
        },
    )

    # Build Prompt Templates section
    prompts_section = (
        "## 3. Prompt Templates Used\n\n"
        "### Config A: Few-Shot Prompting Templates\n\n"
        "#### System Prompt:\n"
        "```text\n"
        f"{prompts.FEW_SHOT_SYSTEM_PROMPT.strip()}\n"
        "```\n\n"
        "#### User Template:\n"
        "```text\n"
        f"{prompts.FEW_SHOT_USER_TEMPLATE.strip()}\n"
        "```\n\n"
        "### Config B: Chain-of-Thought Prompting Templates\n\n"
        "#### System Prompt:\n"
        "```text\n"
        f"{prompts.COT_SYSTEM_PROMPT.strip()}\n"
        "```\n\n"
        "#### User Template:\n"
        "```text\n"
        f"{prompts.COT_USER_TEMPLATE.strip()}\n"
        "```\n"
    )

    # Build Metrics Definitions section
    metrics_section = "## 4. Definitions and Logic for the 3 Custom Metrics\n\n"
    for metric_name, definition in metrics.METRIC_DEFINITIONS.items():
        metrics_section += f"### {metric_name.replace('_', ' ').title()}\n"
        metrics_section += f"```text\n{definition}\n```\n\n"

    # Build Raw Evaluation Data section
    raw_data_section = "## 5. Raw Scenario Evaluation Data\n\n"
    
    raw_data_section += "### Config A: Few-Shot Strategy Raw Scores\n\n"
    raw_data_section += "| Scenario ID | Target Tone | Fact Recall (%) | Tone Accuracy (/100) | Conciseness/Fluency (/100) | Overall (/100) |\n"
    raw_data_section += "|---|---|---|---|---|---|\n"
    for run in evaluation_runs:
        run_a = next(r for r in run["runs"] if r["strategy"] == "few_shot")
        raw_data_section += (
            f"| {run['scenario_id']} | {run['tone']} | "
            f"{run_a['scores']['fact_recall']:.2f} | "
            f"{run_a['scores']['tone_accuracy']:.2f} | "
            f"{run_a['scores']['conciseness_fluency']:.2f} | "
            f"{run_a['scores']['overall']:.2f} |\n"
        )
        
    raw_data_section += "\n### Config B: Chain-of-Thought Strategy Raw Scores\n\n"
    raw_data_section += "| Scenario ID | Target Tone | Fact Recall (%) | Tone Accuracy (/100) | Conciseness/Fluency (/100) | Overall (/100) |\n"
    raw_data_section += "|---|---|---|---|---|---|\n"
    for run in evaluation_runs:
        run_b = next(r for r in run["runs"] if r["strategy"] == "chain_of_thought")
        raw_data_section += (
            f"| {run['scenario_id']} | {run['tone']} | "
            f"{run_b['scores']['fact_recall']:.2f} | "
            f"{run_b['scores']['tone_accuracy']:.2f} | "
            f"{run_b['scores']['conciseness_fluency']:.2f} | "
            f"{run_b['scores']['overall']:.2f} |\n"
        )

    # Combine everything
    report_content = (
        report_header + "\n" +
        analysis_section + "\n" +
        prompts_section + "\n" +
        metrics_section + "\n" +
        raw_data_section
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Generated comprehensive analysis report at {report_path}")


if __name__ == "__main__":
    main()
