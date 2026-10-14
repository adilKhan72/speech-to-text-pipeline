from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

from app.config import settings
from app.pipeline.transcribe import get_model

SAMPLE_RATE = 16000
DEFAULT_TARGET_CHUNK_SECONDS = 300.0


def needs_chunking(duration_seconds: float) -> bool:
    """Chunking is a long-audio-specific stage, not a default step in the pipeline.

    faster-whisper already handles arbitrary-length audio correctly on its own; below
    the threshold, calling transcribe() directly is both simpler and just as correct.
    """
    return duration_seconds > settings.long_audio_threshold_seconds


def split_into_chunks(normalized_path: str, target_chunk_seconds: float = DEFAULT_TARGET_CHUNK_SECONDS):
    """Split long audio into VAD-aligned chunks — never mid-word.

    Reuses faster-whisper's own bundled Silero VAD (the same one vad_filter=True uses
    internally) instead of adding a second VAD dependency. get_speech_timestamps with
    max_speech_duration_s set finds speech regions and, when one would run longer than
    that, splits it at the last silence gap inside it rather than cutting aggressively
    at a fixed sample count.
    """
    audio = decode_audio(normalized_path, sampling_rate=SAMPLE_RATE)
    vad_options = VadOptions(max_speech_duration_s=target_chunk_seconds)
    speech_regions = get_speech_timestamps(audio, vad_options)

    return [
        {
            "audio": audio[region["start"]:region["end"]],
            "offset_seconds": region["start"] / SAMPLE_RATE,
        }
        for region in speech_regions
    ]


def offset_segments(segments: list[dict], offset_seconds: float) -> list[dict]:
    """Pure stitching math: a chunk's segments start at 0 locally — shift them back
    to where that chunk actually sits in the original file's timeline."""
    return [
        {"start": round(s["start"] + offset_seconds, 2), "end": round(s["end"] + offset_seconds, 2), "text": s["text"]}
        for s in segments
    ]


def transcribe_chunked(
    normalized_path: str,
    target_chunk_seconds: float = DEFAULT_TARGET_CHUNK_SECONDS,
    language: str | None = None,
    task: str = "transcribe",
) -> dict:
    """Transcribe a long file chunk by chunk, stitching timestamps back to one timeline.

    Each chunk is already a VAD-selected speech region, so vad_filter is disabled for
    the per-chunk calls — the outer split already did that job; running it twice would
    just be wasted compute.

    If no language hint was given, the language detected on the first chunk is pinned
    for the rest — otherwise every chunk re-detects independently and a quiet or
    code-switched chunk can flip the language mid-file.
    """
    model = get_model()
    all_segments: list[dict] = []

    for chunk in split_into_chunks(normalized_path, target_chunk_seconds):
        segment_iter, info = model.transcribe(chunk["audio"], vad_filter=False, language=language, task=task)
        language = language or info.language
        local_segments = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segment_iter
        ]
        all_segments.extend(offset_segments(local_segments, chunk["offset_seconds"]))

    return {
        "text": " ".join(s["text"] for s in all_segments),
        "segments": all_segments,
        "language": language,
        "task": task,
    }
