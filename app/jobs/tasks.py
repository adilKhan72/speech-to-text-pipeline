from rq import get_current_job

from app.db.models import Job, JobStatus
from app.db.session import get_session
from app.pipeline.exceptions import InvalidOptionsError, UnsupportedAudioError
from app.pipeline.pipeline import run_pipeline
from app.storage.local_fs import get_storage


def process_transcription_job(job_id: str, task: str = "transcribe", language: str | None = None) -> None:
    """The function an rq worker actually runs for a queued job.

    task/language default to plain transcription so jobs enqueued before these options
    existed still run unchanged.

    Permanent failures (UnsupportedAudioError, InvalidOptionsError) are marked failed and never re-raised,
    so RQ's Retry never fires for them — retrying a corrupted file forever wastes
    compute and hides a problem that needs a different input, not another attempt.

    Anything else is treated as transient: re-raised so RQ's own Retry/backoff
    schedules another attempt. We only flip the DB row to FAILED ourselves once we can
    see (via retries_left) that this was the last attempt — RQ's on_failure callback
    fires on every attempt, not just the final one, so we check this explicitly
    instead of relying on it.
    """
    session = get_session()
    try:
        job = session.get(Job, job_id)
        if job is None:
            return

        job.status = JobStatus.PROCESSING
        job.attempts += 1
        session.commit()

        storage = get_storage()
        audio_path = storage.get_path(job.storage_key)

        try:
            result = run_pipeline(audio_path, task, language)
        except (UnsupportedAudioError, InvalidOptionsError) as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            session.commit()
            return

        job.status = JobStatus.DONE
        job.transcript = result
        session.commit()

    except Exception as e:
        current = get_current_job()
        retries_left = current.retries_left if current and current.retries_left else 0

        job = session.get(Job, job_id)
        if job is not None:
            job.error_message = str(e)
            job.status = JobStatus.FAILED if retries_left == 0 else JobStatus.QUEUED
            session.commit()

        raise  # let RQ's Retry actually schedule the next attempt (or record final failure)

    finally:
        session.close()
