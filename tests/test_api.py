import io
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import settings
from app.db.models import Job, JobStatus
from app.db.session import get_session, init_db
from app.main import app
from app.storage.local_fs import get_storage

init_db()  # TestClient(app) alone doesn't run the app's lifespan, so create tables explicitly
client = TestClient(app)

FAKE_TRANSCRIPT = {
    "text": "hello world",
    "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
    "language": "en",
    "duration_seconds": 1.0,
}


def _cleanup_job(job_id: str, storage_key: str | None = None) -> None:
    session = get_session()
    job = session.get(Job, job_id)
    if job is not None:
        session.delete(job)
        session.commit()
    session.close()
    if storage_key:
        get_storage().delete(storage_key)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_transcribe_rejects_missing_api_key():
    response = client.post("/v1/transcribe", files={"file": ("a.wav", io.BytesIO(b"fake"), "audio/wav")})
    assert response.status_code == 401


@patch("app.routers.transcribe.run_pipeline", return_value=FAKE_TRANSCRIPT)
def test_transcribe_sync_returns_transcript_without_touching_the_real_model(mock_run_pipeline):
    response = client.post(
        "/v1/transcribe",
        headers={"X-API-Key": settings.api_key},
        files={"file": ("a.wav", io.BytesIO(b"fake"), "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == FAKE_TRANSCRIPT
    mock_run_pipeline.assert_called_once()


def test_transcribe_rejects_empty_file():
    response = client.post(
        "/v1/transcribe",
        headers={"X-API-Key": settings.api_key},
        files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
    )
    assert response.status_code == 400


def test_jobs_rejects_missing_api_key():
    response = client.post("/v1/jobs", files={"file": ("a.wav", io.BytesIO(b"fake"), "audio/wav")})
    assert response.status_code == 401


@patch("app.routers.jobs.enqueue_transcription")
def test_submit_job_creates_queued_job_and_enqueues_it(mock_enqueue):
    response = client.post(
        "/v1/jobs",
        headers={"X-API-Key": settings.api_key},
        files={"file": ("a.wav", io.BytesIO(b"fake bytes"), "audio/wav")},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    mock_enqueue.assert_called_once_with(body["job_id"], "transcribe", None)

    session = get_session()
    job = session.get(Job, body["job_id"])
    session.close()
    assert job is not None
    assert job.status == JobStatus.QUEUED

    _cleanup_job(body["job_id"], job.storage_key)


def test_submit_job_rejects_empty_file():
    response = client.post(
        "/v1/jobs",
        headers={"X-API-Key": settings.api_key},
        files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
    )
    assert response.status_code == 400


def test_get_job_returns_404_for_unknown_id():
    response = client.get("/v1/jobs/does-not-exist", headers={"X-API-Key": settings.api_key})
    assert response.status_code == 404


def test_get_job_returns_done_status_and_transcript():
    init_db()
    job_id = "test-fixed-job-id"
    session = get_session()
    session.add(
        Job(
            id=job_id,
            original_filename="x.wav",
            storage_key=f"{job_id}/original.wav",
            status=JobStatus.DONE,
            transcript=FAKE_TRANSCRIPT,
        )
    )
    session.commit()
    session.close()

    response = client.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": settings.api_key})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "done"
    assert body["transcript"] == FAKE_TRANSCRIPT

    _cleanup_job(job_id)


@patch("app.routers.transcribe.run_pipeline", return_value={**FAKE_TRANSCRIPT, "task": "translate", "language": "ur"})
def test_transcribe_sync_passes_translate_options_to_pipeline(mock_run_pipeline):
    response = client.post(
        "/v1/transcribe",
        headers={"X-API-Key": settings.api_key},
        files={"file": ("urdu.wav", io.BytesIO(b"fake"), "audio/wav")},
        data={"task": "translate", "language": "ur"},
    )

    assert response.status_code == 200
    assert response.json()["task"] == "translate"
    _, task, language = mock_run_pipeline.call_args.args
    assert (task, language) == ("translate", "ur")


@patch("app.routers.transcribe.run_pipeline")
def test_transcribe_rejects_unknown_task_before_running_pipeline(mock_run_pipeline):
    response = client.post(
        "/v1/transcribe",
        headers={"X-API-Key": settings.api_key},
        files={"file": ("a.wav", io.BytesIO(b"fake"), "audio/wav")},
        data={"task": "summarize"},
    )

    assert response.status_code == 422
    mock_run_pipeline.assert_not_called()


@patch("app.routers.jobs.enqueue_transcription")
def test_submit_job_forwards_translate_options_to_queue(mock_enqueue):
    response = client.post(
        "/v1/jobs",
        headers={"X-API-Key": settings.api_key},
        files={"file": ("urdu.wav", io.BytesIO(b"fake bytes"), "audio/wav")},
        data={"task": "translate", "language": "ur"},
    )

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    mock_enqueue.assert_called_once_with(job_id, "translate", "ur")

    session = get_session()
    job = session.get(Job, job_id)
    session.close()
    _cleanup_job(job_id, job.storage_key)


@patch("app.routers.jobs.enqueue_transcription")
def test_submit_job_rejects_bad_language_without_storing_anything(mock_enqueue):
    response = client.post(
        "/v1/jobs",
        headers={"X-API-Key": settings.api_key},
        files={"file": ("a.wav", io.BytesIO(b"fake bytes"), "audio/wav")},
        data={"language": "klingon"},
    )

    assert response.status_code == 422
    mock_enqueue.assert_not_called()
