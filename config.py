"""
Configuration for the Email Generation Assistant.

Centralizes model names, API/client settings, retry behavior, and sampling
temperatures so they can be tuned from a single place.
"""

# --- Model selection ---
DEFAULT_MODEL = "openai/gpt-oss-120b:free"   # model used to generate emails
JUDGE_MODEL = "openai/gpt-oss-120b:free"     # model used by the LLM-as-a-judge metrics

# --- API / client settings ---
BASE_URL = "https://openrouter.ai/api/v1"    # OpenAI-compatible OpenRouter endpoint
MAX_TOKENS = 1000                            # max tokens per completion
MAX_RETRIES = 3                              # retry attempts on transient API errors

# --- Sampling temperatures ---
FEW_SHOT_TEMP = 0.7    # email generation (few-shot)
COT_TEMP = 0.4         # email generation (chain-of-thought)
JUDGE_TEMP = 0.0       # judge calls: deterministic for reproducibility
