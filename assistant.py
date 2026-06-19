"""
Email generation module.

Generates professional emails from structured inputs (intent, key facts, tone)
using two prompting strategies: Few-Shot and Chain-of-Thought. A mock mode lets
the full pipeline run without API keys or network access.
"""

import os
import re
import time
from typing import List

from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

import prompts
import config

load_dotenv()

# Mock mode is toggled via the MOCK environment variable (set by run_evaluation).
MOCK_MODE = os.environ.get("MOCK", "false").lower() == "true"


def get_client() -> OpenAI:
    """Create an OpenAI-compatible client for OpenRouter.

    Reads the API key from the OPENROUTER_API_KEY environment variable and
    points the client at the configured base URL. Raises a clear error if the
    key is missing.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Provide it in .env, or run in mock mode by setting MOCK=true."
        )
    return OpenAI(api_key=api_key, base_url=config.BASE_URL)


def _strip_code_fences(text: str) -> str:
    """Remove stray leading/trailing Markdown code fences from model output."""
    text = re.sub(r"^```(?:html|text|markdown|email)?\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n```$", "", text)
    return text.strip()


def call_llm_with_retry(
    client: OpenAI,
    messages: List[dict],
    model: str,
    temperature: float,
    max_retries: int = config.MAX_RETRIES,
) -> str:
    """Call the Chat Completions API with retry and exponential backoff.

    Args:
        client: An OpenAI-compatible client instance.
        messages: Chat messages in OpenAI format.
        model: Model identifier.
        temperature: Sampling temperature.
        max_retries: Attempts before giving up.

    Returns:
        The model's response content (empty string if none).
    """
    delay = 1.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=config.MAX_TOKENS,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except OpenAIError as e:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"OpenAI API call failed after {max_retries} attempts: {e}"
                ) from e
            print(
                f"Warning: OpenAI call failed (attempt {attempt + 1}/{max_retries}). "
                f"Retrying in {delay}s... Error: {e}"
            )
            time.sleep(delay)
            delay *= 2.0
    return ""


def get_mock_email(intent: str, key_facts: List[str], tone: str, strategy: str) -> str:
    """Generate a deterministic, structured mock email for offline testing.

    The two strategies produce slightly different styles so the evaluation
    pipeline can still distinguish them in mock mode.
    """
    subject = f"Subject: Follow-up on: {intent[:45]}"
    if len(intent) > 45:
        subject += "..."

    greeting = "Dear Partner," if tone in ["formal", "firm", "assertive", "sincere"] else "Hi there,"

    if strategy == "few_shot":
        # Few-shot mock: simpler, direct fact integration.
        body_intro = f"I am writing to you regarding our recent discussions about: '{intent}'."
        fact_sentences = [f"We would like to note that {fact.lower().rstrip('.')}." for fact in key_facts]
        body_text = " " + " ".join(fact_sentences)
        sign_off = "Sincerely,\nThe Assistant" if tone in ["formal", "firm", "assertive"] else "Best,\nYour Assistant"
    else:
        # Chain-of-thought mock: richer transitions, more tone-attuned.
        body_intro = (
            f"I hope you are having a productive week. Regarding the request to: "
            f"'{intent}', I wanted to reach out and share a few updates."
        )
        fact_sentences = []
        for i, fact in enumerate(key_facts):
            if i == 0:
                fact_sentences.append(f"First and foremost, {fact.lower().rstrip('.')}")
            elif i == len(key_facts) - 1:
                fact_sentences.append(f"finally, please be aware that {fact.lower().rstrip('.')}")
            else:
                fact_sentences.append(f"additionally, {fact.lower().rstrip('.')}")
        body_text = " " + ", and ".join(fact_sentences) + "."
        body_text += " We hope this provides the clarity required for our next steps."
        sign_off = (
            "With high regards,\nThe Senior Assistant"
            if tone in ["formal", "firm", "assertive"]
            else "Warmly,\nYour Support Team"
        )

    return f"{subject}\n\n{greeting}\n\n{body_intro}{body_text}\n\n{sign_off}"


def generate_email(intent: str, key_facts: List[str], tone: str, model: str, strategy: str) -> str:
    """Generate a professional email from structured inputs.

    Args:
        intent: The core purpose of the email.
        key_facts: Facts that must appear in the email.
        tone: Desired style/tone.
        model: Model identifier.
        strategy: 'few_shot' or 'chain_of_thought'.

    Returns:
        The generated email (subject, greeting, body, sign-off).
    """
    if MOCK_MODE:
        return get_mock_email(intent, key_facts, tone, strategy)

    key_facts_bullets = "\n".join(f"  * {fact}" for fact in key_facts)
    client = get_client()

    if strategy == "few_shot":
        messages = [
            {"role": "system", "content": prompts.FEW_SHOT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompts.FEW_SHOT_USER_TEMPLATE.format(
                    intent=intent, key_facts_bullets=key_facts_bullets, tone=tone
                ),
            },
        ]
        email_content = call_llm_with_retry(client, messages, model, config.FEW_SHOT_TEMP)
        return _strip_code_fences(email_content)

    elif strategy == "chain_of_thought":
        messages = [
            {"role": "system", "content": prompts.COT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompts.COT_USER_TEMPLATE.format(
                    intent=intent, key_facts_bullets=key_facts_bullets, tone=tone
                ),
            },
        ]
        raw_response = call_llm_with_retry(client, messages, model, config.COT_TEMP)

        # Extract only the final email from the <email>...</email> tags.
        email_match = re.search(r"<email>(.*?)</email>", raw_response, re.DOTALL | re.IGNORECASE)
        if email_match:
            email_content = email_match.group(1).strip()
        else:
            print("Warning: <email> tags missing from Chain-of-Thought response. Using raw text.")
            email_content = re.sub(
                r"<reasoning>.*?</reasoning>", "", raw_response, flags=re.DOTALL | re.IGNORECASE
            )
        return _strip_code_fences(email_content)

    else:
        raise ValueError(f"Unknown prompting strategy: {strategy}")
