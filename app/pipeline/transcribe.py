from functools import lru_cache

from faster_whisper import WhisperModel

from app.config import settings


@lru_cache(maxsize=1)
def get_model() -> WhisperModel:
    """Load the model exactly once and hold it in memory for the life of the process.

    Loading weights is the expensive part (seconds, not milliseconds) — doing this
    per-request would dominate every response instead of just startup.
    """
    return WhisperModel(
        settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )


def transcribe_audio(normalized_path: str, language: str | None = None, task: str = "transcribe") -> dict:
    """Transcribe (or translate to English) a normalized (16kHz mono WAV) file.

    faster-whisper already slides through arbitrarily long audio internally in ~30s
    windows and returns segments with timestamps correct relative to the whole file —
    vad_filter=True skips silence using its own bundled Silero VAD. For files under the
    long-audio threshold this call alone is the entire pipeline; no external chunking
    needed.
    """
    model = get_model()
    segment_iter, info = model.transcribe(normalized_path, vad_filter=True, language=language, task=task)

    segments = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segment_iter
    ]
    text = " ".join(s["text"] for s in segments)

    return {
        "text": text,
        "segments": segments,
        "language": info.language,
        "task": task,
        "duration_seconds": info.duration,
    }
