from redis import Redis
from rq import Queue, Retry

from app.config import settings

_redis_conn: Redis | None = None
_queue: Queue | None = None

RETRY_MAX_ATTEMPTS = 3
RETRY_INTERVALS_SECONDS = [10, 30, 90]


def get_redis() -> Redis:
    global _redis_conn
    if _redis_conn is None:
        _redis_conn = Redis.from_url(settings.redis_url)
    return _redis_conn


def get_queue() -> Queue:
    global _queue
    if _queue is None:
        _queue = Queue("transcriptions", connection=get_redis())
    return _queue


def enqueue_transcription(job_id: str, task: str = "transcribe", language: str | None = None) -> None:
    """Drop a job onto the queue for a worker to pick up.

    Retry only covers transient failures — the task itself decides whether an error
    is worth retrying (see app/jobs/tasks.py); this just configures how many attempts
    and how long to back off between them.
    """
    from app.jobs.tasks import process_transcription_job

    get_queue().enqueue(
        process_transcription_job,
        job_id,
        task,
        language,
        job_timeout="30m",
        retry=Retry(max=RETRY_MAX_ATTEMPTS, interval=RETRY_INTERVALS_SECONDS),
    )
