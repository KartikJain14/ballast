"""
Ballast API + frontend server.

    python run.py app      (or: uvicorn api.main:app --port 8000)

Serves the dashboard at / and the analysis endpoints under /api. Analysis
runs in a background thread per job so the frontend can poll real,
stage-by-stage progress (not a fabricated timer) while a CSV is processed.
"""
from __future__ import annotations

import io
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.pipeline import run_pipeline  # noqa: E402
from api.ai_summary import generate_summary  # noqa: E402
from engine.data import RAW_COLUMNS, DEFAULT_CSV_PATH  # noqa: E402

app = FastAPI(title="Ballast API")

JOBS: dict[str, dict] = {}
UPLOAD_DIR = ROOT / ".cache" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def index():
    return FileResponse(ROOT / "ui" / "dashboard.html")


@app.get("/api/sample-info")
def sample_info():
    return {"has_default": DEFAULT_CSV_PATH.exists(), "filename": "treasury_4tier_dataset.csv (sample)"}


def _start_job(csv_path: Path) -> str:
    job_id = str(uuid.uuid4())
    job = {"status": "running", "stage": "validate", "percent": 0}
    JOBS[job_id] = job
    t = threading.Thread(target=run_pipeline, args=(csv_path, job), daemon=True)
    t.start()
    return job_id


@app.post("/api/run-sample")
def run_sample():
    if not DEFAULT_CSV_PATH.exists():
        raise HTTPException(404, "Sample dataset not found on server.")
    job_id = _start_job(DEFAULT_CSV_PATH)
    return {"job_id": job_id}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > 60 * 1024 * 1024:
        raise HTTPException(413, "File too large (60MB limit).")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise HTTPException(
            400,
            f"Missing required columns: {sorted(missing)}. "
            f"Expected the treasury_4tier_dataset schema: {RAW_COLUMNS}",
        )

    job_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{job_id}.csv"
    dest.write_bytes(raw)
    job_id2 = _start_job(dest)
    return {"job_id": job_id2}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id.")
    return {
        "status": job.get("status", "running"),
        "stage": job.get("stage"),
        "stage_label": job.get("stage_label"),
        "percent": job.get("percent", 0),
        "error": job.get("error"),
    }


@app.get("/api/results/{job_id}")
def results(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id.")
    if job.get("status") != "done":
        raise HTTPException(409, "Job not finished yet.")
    return JSONResponse({
        "validation": job.get("validation"),
        "flows": job.get("flows"),
        "risk": job.get("risk"),
        "scoreboard": job.get("scoreboard"),
        "best_static_buffer": job.get("best_static_buffer"),
        "regret": job.get("regret"),
        "stress": job.get("stress"),
        "stress_cap": job.get("stress_cap"),
    })


@app.get("/api/summarize/{job_id}")
def summarize(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id.")
    if job.get("status") != "done":
        raise HTTPException(409, "Job not finished yet.")
    data = {
        "scoreboard": job["scoreboard"], "risk": job["risk"],
        "regret": job["regret"], "stress": job["stress"],
    }
    return generate_summary(data)
