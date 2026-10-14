import asyncio
import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.auth import require_api_key
from app.config import settings
from app.pipeline.exceptions import InvalidOptionsError, UnsupportedAudioError
from app.pipeline.options import validate_options
from app.pipeline.pipeline import run_pipeline
from app.rate_limit import limiter

router = APIRouter(prefix="/v1", tags=["transcribe"])

# Whisper inference is CPU-bound and blocking; even on the "synchronous" path we
# don't want unlimited concurrent calls stampeding the model at once.
_inference_semaphore = asyncio.Semaphore(os.cpu_count() or 2)


@router.post("/transcribe", dependencies=[Depends(require_api_key)])
@limiter.limit(settings.rate_limit)
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    task: str = Form("transcribe"),
    language: str | None = Form(None),
) -> dict:
    """Synchronous path: for short clips and quick demos. Runs the full pipeline and
    returns the transcript directly instead of a job id — anything long should go
    through POST /v1/jobs instead so the caller isn't holding a connection open.

    task="translate" returns English text from any spoken language; language is an
    optional hint for the spoken language (e.g. "ur")."""
    try:
        task, language = validate_options(task, language)
    except InvalidOptionsError as e:
        raise HTTPException(status_code=422, detail=str(e))

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")

    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(content)
        tmp.flush()

        async with _inference_semaphore:
            try:
                result = await asyncio.to_thread(run_pipeline, tmp.name, task, language)
            except UnsupportedAudioError as e:
                raise HTTPException(status_code=422, detail=str(e))

    return result
