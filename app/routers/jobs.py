import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.auth import require_api_key
from app.db.models import Job, JobStatus
from app.db.session import get_session
from app.jobs.queue import enqueue_transcription
from app.pipeline.exceptions import InvalidOptionsError
from app.pipeline.options import validate_options
from app.rate_limit import limiter
from app.config import settings
from app.storage.local_fs import get_storage

router = APIRouter(prefix="/v1", tags=["jobs"])


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    transcript: dict | None = None
    error: str | None = None


@router.post(
    "/jobs",
    status_code=202,
    dependencies=[Depends(require_api_key)],
    response_model=JobSubmitResponse,
)
@limiter.limit(settings.rate_limit)
async def submit_job(
    request: Request,
    file: UploadFile = File(...),
    task: str = Form("transcribe"),
    language: str | None = Form(None),
) -> JobSubmitResponse:
    """Asynchronous path: saves the audio, queues the job, returns immediately.

    This is where the real concurrency story lives — RQ worker count is the actual
    throttle on how many transcriptions run at once, not this endpoint.

    Options are validated here, before anything is stored or queued, and travel to the
    worker as job arguments (RQ keeps them across retries) rather than as new DB columns.
    """
    try:
        task, language = validate_options(task, language)
    except InvalidOptionsError as e:
        raise HTTPException(status_code=422, detail=str(e))

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")

    job_id = str(uuid.uuid4())
    storage_key = get_storage().save(job_id, file.filename or "upload.bin", content)

    session = get_session()
    try:
        job = Job(id=job_id, original_filename=file.filename or "upload.bin", storage_key=storage_key)
        session.add(job)
        session.commit()
    finally:
        session.close()

    enqueue_transcription(job_id, task, language)

    return JobSubmitResponse(job_id=job_id, status=JobStatus.QUEUED.value)


@router.get(
    "/jobs/{job_id}",
    dependencies=[Depends(require_api_key)],
    response_model=JobStatusResponse,
)
async def get_job(job_id: str) -> JobStatusResponse:
    session = get_session()
    try:
        job = session.get(Job, job_id)
    finally:
        session.close()

    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    return JobStatusResponse(
        job_id=job.id,
        status=job.status.value,
        transcript=job.transcript,
        error=job.error_message,
    )
