"""
Ballast API + frontend server.

    python run.py app      (or: uvicorn api.main:app --port 8000)

Serves the dashboard at / and the analysis endpoints under /api. Analysis
runs in a background thread per job so the frontend can poll real,
stage-by-stage progress (not a fabricated timer) while a CSV is processed.

The built-in reference datasets (the sample + a few synthetic variants) are
precomputed once at server startup and cached to disk, so selecting one in
the UI is instant rather than re-running a ~40s simulation every time --
only a freshly uploaded CSV triggers a live run.
"""
from __future__ import annotations

import io
import json
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.pipeline import run_pipeline  # noqa: E402
from api.ai_summary import generate_summary, generate_summary_stream  # noqa: E402
from engine.data import RAW_COLUMNS, DEFAULT_CSV_PATH  # noqa: E402

app = FastAPI(title="Ballast API")

JOBS: dict[str, dict] = {}
UPLOAD_DIR = ROOT / ".cache" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PRECOMPUTE_DIR = ROOT / ".cache" / "precomputed"
PRECOMPUTE_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_DATASETS = {
    "sample": {
        "path": DEFAULT_CSV_PATH,
        "name": "Reference Dataset",
        "description": "The original one-year, 8,760-hour treasury dataset.",
    },
    "growing": {
        "path": ROOT / "data" / "synthetic_growing_bank.csv",
        "name": "Growing Bank",
        "description": "Same shape, deposits trending up ~40%/year instead of down.",
    },
    "volatile": {
        "path": ROOT / "data" / "synthetic_volatile_bank.csv",
        "name": "Volatile Bank",
        "description": "Same average flow, ~2.5x the hour-to-hour noise.",
    },
    "bank_run": {
        "path": ROOT / "data" / "synthetic_bank_run_aftermath.csv",
        "name": "Bank-Run Aftermath",
        "description": "An SVB-calibrated run baked into the middle of the year.",
    },
}


@app.get("/")
def index():
    return FileResponse(ROOT / "ui" / "dashboard.html")


@app.get("/api/datasets")
def list_datasets():
    return [
        {
            "key": k, "name": v["name"], "description": v["description"],
            "available": v["path"].exists(), "cached": _cache_path(k).exists(),
        }
        for k, v in REFERENCE_DATASETS.items()
    ]


def _cache_path(key: str) -> Path:
    return PRECOMPUTE_DIR / f"{key}.json"


def _done_job_from_cache(key: str) -> dict | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            data = json.load(f)
        job = {"status": "done", "stage": "done", "stage_label": "Finalizing report", "percent": 100}
        job.update(data)
        return job
    except Exception:
        return None


def _precompute_and_cache(key: str, csv_path: Path):
    job = {"status": "running", "stage": "validate", "percent": 0}
    run_pipeline(csv_path, job)
    if job.get("status") == "done":
        with open(_cache_path(key), "w") as f:
            json.dump({k: v for k, v in job.items() if k not in ("status", "stage", "stage_label", "percent")}, f)


def _warm_all_sequentially():
    # One worker thread processing datasets one at a time, NOT one thread per
    # dataset: this is CPU-bound pandas/numpy work, and Python threads don't
    # parallelize that (GIL contention) -- 4 concurrent threads measured
    # SLOWER to finish even the first dataset than running them in sequence.
    # "sample" is listed first so it (the one most likely to be clicked first)
    # becomes instant soonest.
    for key, meta in REFERENCE_DATASETS.items():
        if not meta["path"].exists() or _cache_path(key).exists():
            continue
        _precompute_and_cache(key, meta["path"])


@app.on_event("startup")
def _warm_reference_datasets():
    threading.Thread(target=_warm_all_sequentially, daemon=True).start()


def _start_job(csv_path: Path) -> str:
    job_id = str(uuid.uuid4())
    job = {"status": "running", "stage": "validate", "percent": 0}
    JOBS[job_id] = job
    t = threading.Thread(target=run_pipeline, args=(csv_path, job), daemon=True)
    t.start()
    return job_id


@app.post("/api/run-dataset/{key}")
def run_dataset(key: str):
    meta = REFERENCE_DATASETS.get(key)
    if meta is None:
        raise HTTPException(404, f"Unknown reference dataset '{key}'.")
    if not meta["path"].exists():
        raise HTTPException(404, f"Dataset file for '{key}' not found on server.")

    cached = _done_job_from_cache(key)
    if cached is not None:
        job_id = str(uuid.uuid4())
        JOBS[job_id] = cached
        return {"job_id": job_id, "cached": True}

    job_id = _start_job(meta["path"])
    # Opportunistically cache this run too, so the NEXT request for the same
    # reference dataset is instant even if startup warming hadn't finished yet.
    threading.Thread(target=_watch_and_cache, args=(job_id, key), daemon=True).start()
    return {"job_id": job_id, "cached": False}


def _watch_and_cache(job_id: str, key: str):
    import time
    while True:
        job = JOBS.get(job_id)
        if job is None:
            return
        if job.get("status") == "done":
            with open(_cache_path(key), "w") as f:
                json.dump({k: v for k, v in job.items() if k not in ("status", "stage", "stage_label", "percent")}, f)
            return
        if job.get("status") == "error":
            return
        time.sleep(1)


@app.post("/api/run-sample")
def run_sample():
    """Kept for backward compatibility; equivalent to /api/run-dataset/sample."""
    return run_dataset("sample")


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
    """Returns whatever sections of the analysis are ready RIGHT NOW, even
    mid-run -- the pipeline fills in job['flows'], job['risk'], etc. as each
    stage completes, so the frontend can render each chart the moment its
    real data exists instead of waiting for the entire ~40s run to finish."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id.")
    return JSONResponse({
        "status": job.get("status", "running"),
        "stage": job.get("stage"), "percent": job.get("percent", 0),
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


@app.get("/api/summarize-stream/{job_id}")
def summarize_stream(job_id: str):
    """Chunked plain-text stream for the frontend's progressive reveal effect.
    Deliberately a synchronous (not async def) endpoint -- it uses a blocking
    generator (real Gemini SSE reads, or a paced template fallback), and
    FastAPI runs sync path operations in a worker thread automatically, so
    this doesn't block the event loop."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id.")
    if job.get("status") != "done":
        raise HTTPException(409, "Job not finished yet.")
    data = {
        "scoreboard": job["scoreboard"], "risk": job["risk"],
        "regret": job["regret"], "stress": job["stress"],
    }
    return StreamingResponse(generate_summary_stream(data), media_type="text/plain")
