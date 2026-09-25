"""LLM helpers: predict Zenodo record tags with a few-shot chat prompt."""

from __future__ import annotations

from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "gpt-oss:20b"
DEFAULT_MAX_EXAMPLES = 20
MAX_DESCRIPTION_CHARS = 2000

_SYSTEM_PROMPT = (
    "You are an expert at tagging scientific records on Zenodo. "
    "Given the description of a record, reply with a comma-separated list of "
    "relevant tags/keywords. Return ONLY the tags separated by commas — "
    "no explanations, no extra text."
)


def _truncate(text: str, max_chars: int = MAX_DESCRIPTION_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " …"


def build_messages(
    description: str,
    examples: list[tuple[str, list[str]]],
    *,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> list[dict]:
    """Build an OpenAI-compatible message list for few-shot tag prediction.

    Parameters
    ----------
    description:
        Plain-text description of the record whose tags we want to predict.
    examples:
        List of ``(description, tags)`` pairs from records that already have
        tags.  Used as few-shot examples.
    max_examples:
        Maximum number of few-shot examples to include (to stay within typical
        context-window limits).
    """
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for desc, tags in examples[-max_examples:]:
        messages.append({"role": "user", "content": _truncate(desc)})
        messages.append({"role": "assistant", "content": ", ".join(tags)})
    messages.append({"role": "user", "content": _truncate(description)})
    return messages


def predict_tags(
    description: str,
    examples: list[tuple[str, list[str]]],
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> list[str]:
    """Return predicted tags for *description* using an OpenAI-compatible LLM.

    Parameters
    ----------
    description:
        Plain-text description of the record to tag.
    examples:
        Few-shot training pairs ``(description, tags)`` from known records.
    model:
        Name of the model to use (e.g. ``"llama3"``).
    base_url:
        Base URL of the OpenAI-compatible endpoint
        (default: ``"http://localhost:11434/v1"`` for Ollama).
    api_key:
        API key for the LLM service.
    max_examples:
        Maximum number of few-shot examples to include.
    """
    client = OpenAI(base_url=base_url, api_key=api_key)
    messages = build_messages(description, examples, max_examples=max_examples)
    response = client.chat.completions.create(model=model, messages=messages)
    raw = response.choices[0].message.content or ""
    tags = [t.strip() for t in raw.split(",") if t.strip()]
    return tags
