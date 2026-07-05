import os
import re
import asyncio
import json
from functools import lru_cache
from typing import Any
from google import genai
from google.genai import types


MODEL_A = "gemma-4-31b-it"
MODEL_B = "gemma-4-26b-a4b-it"
MODELS_TO_TRY = [MODEL_A, MODEL_B] 
PER_MODEL_TIMEOUT_SECONDS = 30

SYSTEM_INSTRUCTION = """You are a code-mixed Indian language normalizer.
Input will be Romanized text mixing Bengali/Hindi with English (Benglish/Hinglish,
as typed casually in chat apps by users in Kolkata).

Your job:
1. Detect which words are Bengali, Hindi, or English.
2. Produce a fully normalized version in native script — Bengali words in Bengali
   script, Hindi words in Devanagari, English words kept as English.
3. Produce a natural, fluent English translation of the full message.
4. Return the percentage of words that are Bengali, Hindi, and English respectively.

Never guess wildly on ambiguous words — if a word could be either Bengali or Hindi
romanization, use surrounding context to decide."""

# Kept for documentation of the expected shape. NOT passed as response_schema —
# structured output was the confirmed trigger for the earlier 500 INTERNAL errors
# on Gemma 4. JSON is requested via prompt text instead (see normalize_async).
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "native_script": {"type": "string"},
        "english_translation": {"type": "string"},
        "language_ratio": {
            "type": "object",
            "properties": {
                "bengali_pct": {"type": "integer"},
                "hindi_pct": {"type": "integer"},
                "english_pct": {"type": "integer"},
            },
            "required": ["bengali_pct", "hindi_pct", "english_pct"],
        },
    },
    "required": ["native_script", "english_translation", "language_ratio"],
}


class NormalizerError(Exception):
    """Raised when normalization cannot return the required API contract."""


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazy singleton so we don't pay client-construction cost on every call."""
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise NormalizerError("GOOGLE_API_KEY is not set.")
        _client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=45_000),  # milliseconds
        )
    return _client


async def _try_model(client: genai.Client, model: str, prompt: str) -> dict:
    # NOTE: `timeout` is NOT a valid field on GenerateContentConfig in the
    # current google-genai SDK. Passing it there raised a validation error on
    # every call, which was the root cause of the "Gemini API request failed"
    # message showing up unconditionally. Real per-call timeout is enforced
    # below via asyncio.wait_for instead, which doesn't depend on SDK-specific
    # config fields.
    coro = client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=1024,
            response_mime_type="application/json",
        ),
    )
    response = await asyncio.wait_for(coro, timeout=PER_MODEL_TIMEOUT_SECONDS)

    parsed = getattr(response, "parsed", None)
    result = (
        parsed
        if isinstance(parsed, dict)
        else json.loads(_extract_json_text(response.text))
    )
    _validate_result(result)
    return result


async def normalize_async(text: str) -> dict:
    """
    Returns a dict matching SCHEMA above.
    Raises NormalizerError on API failure or malformed response — never
    raises a raw SDK exception to the caller.
    """
    if not text.strip():
        raise NormalizerError("Input text is required.")

    client = _get_client()
    prompt = (
        "Normalize this Kolkata Romanized Bengali/Hindi/English chat message.\n"
        "Return only valid JSON with this exact shape:\n"
        '{"native_script":"string","english_translation":"string",'
        '"language_ratio":{"bengali_pct":0,"hindi_pct":0,"english_pct":0}}\n'
        f"Input: {text}"
    )

    tasks = [
        asyncio.create_task(_try_model(client, m, prompt)) for m in MODELS_TO_TRY
    ]
    last_error: Exception | None = None

    try:
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                return result
            except Exception as exc:  # noqa: BLE001 - intentionally broad, we fall back
                last_error = exc
                continue
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        # Swallow the resulting CancelledError from any task we just cancelled
        # so it doesn't surface as an "unretrieved exception" warning in logs.
        await asyncio.gather(*tasks, return_exceptions=True)

    error_text = str(last_error) if last_error else "unknown error"

    if "API key not valid" in error_text or "API_KEY_INVALID" in error_text:
        raise NormalizerError("Invalid API key. Please check your credentials.") from last_error

    # Include the real cause in the message itself, not just as __cause__ —
    # this way even a plain `st.error(str(e))` in app.py surfaces enough to
    # debug, instead of a uniformly generic message every time.
    raise NormalizerError(
        f"Gemini API request failed: {error_text}"
    ) from last_error


@lru_cache(maxsize=32)
def normalize(text: str) -> dict:
    """Sync wrapper for use from Streamlit. Cached so repeated demo inputs
    (e.g. the example buttons) don't re-hit the API."""
    return asyncio.run(normalize_async(text))


def _extract_json_text(text: str) -> str:
    """Strip code fences, and fall back to regex-extracting the first
    {...} block if the model added stray preamble/explanation text
    despite instructions not to."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").strip()

    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return match.group(0)
        return cleaned  # let the caller's json.loads raise a clear error


def _validate_result(result: Any) -> None:
    if not isinstance(result, dict):
        raise NormalizerError("Malformed model response: top-level value is not an object.")

    required_keys = {"native_script", "english_translation", "language_ratio"}
    missing = required_keys - set(result)
    if missing:
        raise NormalizerError(f"Malformed model response: missing keys {sorted(missing)}.")

    if not isinstance(result["native_script"], str):
        raise NormalizerError("Malformed model response: native_script must be a string.")
    if not isinstance(result["english_translation"], str):
        raise NormalizerError("Malformed model response: english_translation must be a string.")

    ratio = result["language_ratio"]
    if not isinstance(ratio, dict):
        raise NormalizerError("Malformed model response: language_ratio must be an object.")

    ratio_keys = {"bengali_pct", "hindi_pct", "english_pct"}
    missing_ratio = ratio_keys - set(ratio)
    if missing_ratio:
        raise NormalizerError(
            f"Malformed model response: missing ratio keys {sorted(missing_ratio)}."
        )

    for key in ratio_keys:
        value = ratio[key]
        if not isinstance(value, (int, float)):
            raise NormalizerError(f"Malformed model response: {key} must be numeric.")
