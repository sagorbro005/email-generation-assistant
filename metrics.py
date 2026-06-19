"""
Custom evaluation metrics for the Email Generation Assistant.

Implements three metrics, each scored on a 0-100 scale (higher is better):
    1. Fact Recall          - automated LLM-as-a-judge
    2. Tone Accuracy        - LLM-as-a-judge
    3. Conciseness & Fluency - hybrid Python (readability) + LLM (fluency)

All judge calls use temperature=0 for reproducibility, with robust JSON parsing.
"""

import json
import re
from typing import List, Dict, Any

from assistant import call_llm_with_retry, get_client, MOCK_MODE
import config

# Human-readable definitions and logic for each metric (surfaced in the report).
METRIC_DEFINITIONS: Dict[str, str] = {
    "fact_recall": (
        "FACT RECALL (Automated LLM-as-a-judge):\n"
        "Measures the fraction of the scenario's key facts present in the generated email.\n"
        "An LLM judge reviews the email and, for each key fact, decides whether it is clearly\n"
        "conveyed (yes/no). The final score is computed as (facts present / total facts) * 100."
    ),
    "tone_accuracy": (
        "TONE ACCURACY (LLM-as-a-judge):\n"
        "Evaluates how well the generated email matches the requested tone. An LLM judge rates the\n"
        "tone on a scale of 1 to 5. The judge outputs a JSON response containing 'score' and 'reason'.\n"
        "The raw score is normalized to a 0-100 scale using the formula: (score - 1) / 4 * 100."
    ),
    "conciseness_fluency": (
        "CONCISENESS & FLUENCY (Hybrid Python + LLM):\n"
        "Combines (a) a readability/length component computed locally in Python with (b) an LLM judge\n"
        "grammar and fluency rating (1-5, normalized to 0-100). The two scores are averaged 50/50.\n"
        "Python Readability/Length Formula:\n"
        "- Word Count (W) Score: 100 if 80 <= W <= 220. If W < 80, penalty is (80 - W) * 2. If W > 220, penalty is (W - 220) * 0.8.\n"
        "- Sentence Length (S) Score: 100 if 10 <= S <= 20. If S < 10, penalty is (10 - S) * 10. If S > 20, penalty is (S - 20) * 5.\n"
        "- Python Score = (Word Count Score + Sentence Length Score) / 2\n"
        "LLM Fluency Formula:\n"
        "- Normalized Fluency Score = (LLM Score - 1) / 4 * 100\n"
        "Final Score = (Python Score + Normalized Fluency Score) / 2"
    ),
}


def parse_judge_json(text: str) -> Dict[str, Any]:
    """Safely parse JSON from an LLM judge response.

    Handles Markdown code fences and stray text, with regex fallbacks so a
    malformed response never crashes the pipeline.
    """
    text_clean = text.strip()

    start = text_clean.find("{")
    end = text_clean.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text_clean[start:end + 1])
        except json.JSONDecodeError:
            pass

    score_match = re.search(r'"score"\s*:\s*([1-5])', text_clean, re.IGNORECASE)
    if score_match:
        return {"score": int(score_match.group(1)), "reason": "Regex fallback parsed score."}

    num_match = re.search(r'\b([1-5])\b', text_clean)
    if num_match:
        return {"score": int(num_match.group(1)), "reason": "Regex fallback parsed standalone number."}

    return {"score": 3, "reason": "Parsing failed completely. Returned default score."}


def evaluate_fact_recall(
    generated_email: str,
    key_facts: List[str],
    model: str = config.JUDGE_MODEL,
) -> float:
    """Score the fraction of key facts present in the email (0-100).

    Score = (facts present / total facts) * 100.
    """
    if not key_facts:
        return 100.0

    if MOCK_MODE:
        if any(m in generated_email for m in ["Senior Assistant", "Support Team", "chain_of_thought"]):
            return 100.0
        return 75.0

    client = get_client()

    facts_list_str = "\n".join(f"{i + 1}. {fact}" for i, fact in enumerate(key_facts))
    system_prompt = (
        "You are an objective evaluation auditor. Your task is to verify if specific key facts "
        "are clearly conveyed in a generated email. For each fact, you must answer with either "
        "'yes' or 'no'. Be strict: the fact must be clearly stated or directly implied without "
        "ambiguity.\n\n"
        "Output your evaluations as a JSON object where the keys are the exact facts from the list, "
        "and values are either 'yes' or 'no'. Example:\n"
        "{\n"
        "  \"Fact number one text\": \"yes\",\n"
        "  \"Fact number two text\": \"no\"\n"
        "}"
    )
    user_prompt = (
        f"EMAIL TO AUDIT:\n-----------------\n{generated_email}\n-----------------\n\n"
        f"KEY FACTS TO CHECK:\n{facts_list_str}\n\nProvide your JSON evaluations:"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_response = call_llm_with_retry(client, messages, model, config.JUDGE_TEMP)
        evals = parse_judge_json(raw_response)

        yes_count = 0
        for fact in key_facts:
            matched_val = "no"
            for k, v in evals.items():
                if fact.lower() in k.lower() or k.lower() in fact.lower():
                    matched_val = str(v).strip().lower()
                    break
            if matched_val in ("yes", "true"):
                yes_count += 1

        return (yes_count / len(key_facts)) * 100.0
    except Exception as e:
        print(f"Error during Fact Recall evaluation: {e}. Falling back to default score (50.0).")
        return 50.0


def evaluate_tone_accuracy(
    generated_email: str,
    target_tone: str,
    model: str = config.JUDGE_MODEL,
) -> float:
    """Score how well the email matches the requested tone (0-100).

    Score = (rating - 1) / 4 * 100, where rating is the judge's 1-5 score.
    """
    if MOCK_MODE:
        if any(m in generated_email for m in ["Senior Assistant", "Support Team", "chain_of_thought"]):
            return 100.0
        return 75.0

    client = get_client()

    system_prompt = (
        "You are an expert editor auditing tone. Evaluate how well the generated email "
        "matches the target tone requested on a scale of 1 to 5:\n"
        "1: Completely fails to match (e.g. extremely rude for a warm tone, or overly casual for formal)\n"
        "2: Poorly matches (wrong vibe, awkward fits)\n"
        "3: Moderately matches (acceptable, but inconsistent)\n"
        "4: Mostly matches (good tone throughout, fits the situation)\n"
        "5: Perfectly matches (excellent execution of tone)\n\n"
        "You must return a JSON response containing 'score' (integer 1-5) and 'reason' (string explaining your rating).\n"
        "Example:\n"
        "{\n"
        "  \"score\": 4,\n"
        "  \"reason\": \"The email is structured politely and matches the professional tone, though a bit verbose.\"\n"
        "}"
    )
    user_prompt = (
        f"TARGET TONE: {target_tone}\n\n"
        f"EMAIL TO AUDIT:\n-----------------\n{generated_email}\n-----------------\n\n"
        f"Provide your JSON evaluation:"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_response = call_llm_with_retry(client, messages, model, config.JUDGE_TEMP)
        parsed = parse_judge_json(raw_response)
        score = parsed.get("score", 3)
        return ((score - 1) / 4.0) * 100.0
    except Exception as e:
        print(f"Error during Tone Accuracy evaluation: {e}. Falling back to default score (50.0).")
        return 50.0


def evaluate_conciseness_fluency(
    generated_email: str,
    model: str = config.JUDGE_MODEL,
) -> float:
    """Score conciseness and fluency (0-100).

    Combines a Python readability/length component with an LLM fluency rating,
    averaged 50/50. See METRIC_DEFINITIONS['conciseness_fluency'] for the formula.
    """
    # 1. Python readability component.
    words = generated_email.split()
    word_count = len(words)

    if 80 <= word_count <= 220:
        word_score = 100.0
    elif word_count < 80:
        word_score = max(0.0, 100.0 - (80 - word_count) * 2.0)
    else:
        word_score = max(0.0, 100.0 - (word_count - 220) * 0.8)

    sentences = [s.strip() for s in re.split(r'[.!?]+', generated_email) if s.strip()]
    sentence_count = len(sentences)

    if sentence_count == 0:
        sentence_score = 0.0
    else:
        avg_sentence_len = word_count / sentence_count
        if 10 <= avg_sentence_len <= 20:
            sentence_score = 100.0
        elif avg_sentence_len < 10:
            sentence_score = max(0.0, 100.0 - (10 - avg_sentence_len) * 10.0)
        else:
            sentence_score = max(0.0, 100.0 - (avg_sentence_len - 20) * 5.0)

    python_score = (word_score + sentence_score) / 2.0

    # 2. LLM fluency component.
    if MOCK_MODE:
        if any(m in generated_email for m in ["Senior Assistant", "Support Team", "chain_of_thought"]):
            llm_score = 100.0
            python_score = min(python_score, 95.0)
        else:
            llm_score = 75.0
            python_score = min(python_score, 85.0)
        return (python_score + llm_score) / 2.0

    client = get_client()

    system_prompt = (
        "You are an expert editor auditing writing fluency. Rate the grammar, spelling, fluency, "
        "and overall readability of the following email on a scale of 1 to 5:\n"
        "1: Unreadable (broken English, incomprehensible sentence structures)\n"
        "2: Poor (numerous grammatical errors, hard to follow)\n"
        "3: Fair (discernible errors, somewhat robotic or awkward phrasing)\n"
        "4: Good (clear structure, minimal errors, natural phrasing)\n"
        "5: Excellent (flawless grammar, natural flow, highly professional syntax)\n\n"
        "You must return a JSON response containing 'score' (integer 1-5) and 'reason' (string explaining your rating).\n"
        "Example:\n"
        "{\n"
        "  \"score\": 5,\n"
        "  \"reason\": \"The grammar is perfect, and the email flows very naturally with professional styling.\"\n"
        "}"
    )
    user_prompt = (
        f"EMAIL TO AUDIT:\n-----------------\n{generated_email}\n-----------------\n\n"
        f"Provide your JSON evaluation:"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_response = call_llm_with_retry(client, messages, model, config.JUDGE_TEMP)
        parsed = parse_judge_json(raw_response)
        score = parsed.get("score", 3)
        normalized_fluency = ((score - 1) / 4.0) * 100.0
        return (python_score + normalized_fluency) / 2.0
    except Exception as e:
        print(f"Error during Fluency evaluation: {e}. Falling back to default fluency score (50.0).")
        return (python_score + 50.0) / 2.0
