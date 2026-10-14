from faster_whisper.tokenizer import _LANGUAGE_CODES

from app.pipeline.exceptions import InvalidOptionsError

TASKS = ("transcribe", "translate")


def validate_options(task: str, language: str | None) -> tuple[str, str | None]:
    """Normalize and validate the per-request Whisper options.

    task="translate" is Whisper's built-in speech translation: it always outputs
    English, whatever the spoken language (e.g. Urdu audio -> English text) — no
    second translation model needed.

    language is an optional hint for the *spoken* language. Leaving it out lets
    Whisper auto-detect, which is fine for clean audio but can misfire on short or
    code-switched clips (Urdu/English mixing is common), so callers who know the
    language should pass it. Validated against faster-whisper's own code list so a
    typo fails fast with a 422 instead of deep inside the model.
    """
    task = (task or "transcribe").strip().lower()
    if task not in TASKS:
        raise InvalidOptionsError(f"task must be one of {', '.join(TASKS)}")

    if language is not None:
        language = language.strip().lower() or None
    if language is not None and language not in _LANGUAGE_CODES:
        raise InvalidOptionsError(f"unsupported language code: {language!r}")

    return task, language
