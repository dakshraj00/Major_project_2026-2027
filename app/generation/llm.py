"""Thin wrapper around the Groq chat API (default: openai/gpt-oss-120b; Llama 3.3 70B was retired from Groq)."""
from functools import lru_cache

from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL, PROMPTS_DIR


class MissingAPIKey(RuntimeError):
    pass


@lru_cache
def _client() -> Groq:
    if not GROQ_API_KEY:
        raise MissingAPIKey("GROQ_API_KEY is empty. Add it to vidyarag/.env (free key: https://console.groq.com)")
    return Groq(api_key=GROQ_API_KEY, max_retries=6)  # SDK backs off on 429s (free tier: 8k tokens/min)


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def chat(prompt: str, temperature: float = 0.0, json_mode: bool = False) -> str:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = _client().chat.completions.create(
        model=GROQ_MODEL, temperature=temperature,
        messages=[{"role": "user", "content": prompt}], **kwargs)
    return resp.choices[0].message.content.strip()
