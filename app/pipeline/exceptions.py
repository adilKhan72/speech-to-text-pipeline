class UnsupportedAudioError(Exception):
    """Raised when a file can't be identified or normalized as audio. Permanent — never retried."""


class InvalidOptionsError(ValueError):
    """Raised for an unknown task or language code. A client error — never retried."""


class TransientPipelineError(Exception):
    """Raised for recoverable failures (storage IO, transient environment issues). Safe to retry."""
