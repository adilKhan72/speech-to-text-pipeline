from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.pipeline.exceptions import InvalidOptionsError
from app.pipeline.options import validate_options
from app.pipeline.transcribe import transcribe_audio


def test_defaults_to_plain_transcription_with_autodetect():
    assert validate_options("transcribe", None) == ("transcribe", None)


def test_translate_with_urdu_hint_is_accepted():
    assert validate_options("translate", "ur") == ("translate", "ur")


def test_options_are_normalized():
    assert validate_options(" Translate ", " UR ") == ("translate", "ur")


def test_blank_language_means_autodetect():
    assert validate_options("transcribe", "  ") == ("transcribe", None)


def test_unknown_task_is_rejected():
    with pytest.raises(InvalidOptionsError):
        validate_options("summarize", None)


def test_unknown_language_is_rejected():
    with pytest.raises(InvalidOptionsError):
        validate_options("transcribe", "urdu")


def _fake_model(language="ur"):
    model = MagicMock()
    segments = [SimpleNamespace(start=0.0, end=1.5, text=" Hello, how are you? ")]
    model.transcribe.return_value = (iter(segments), SimpleNamespace(language=language, duration=1.5))
    return model


def test_transcribe_audio_passes_task_and_language_to_whisper():
    model = _fake_model()
    with patch("app.pipeline.transcribe.get_model", return_value=model):
        result = transcribe_audio("normalized.wav", language="ur", task="translate")

    model.transcribe.assert_called_once_with("normalized.wav", vad_filter=True, language="ur", task="translate")
    assert result["task"] == "translate"
    assert result["language"] == "ur"  # the *spoken* language, even though the text is English
    assert result["text"] == "Hello, how are you?"
