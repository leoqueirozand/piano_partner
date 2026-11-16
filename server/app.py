"""
Slice 1: backend skeleton -- accepts a recorded performance, creates its
own timestamped folder under performances/, saves the raw WAV.
Slice 2: runs both transcription models against that WAV and writes
their MIDI/JSON outputs into the same folder.
Slice 3: runs the evaluator against those outputs and a reference JSON,
writes feedback.json, and returns the structured report.

Run with:
    .venv/bin/uvicorn server.app:app --reload --port 8787
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from server.evaluate import evaluate
from server.transcribe import get_audio_duration, run_transkun

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERFORMANCES_DIR = PROJECT_ROOT / "performances"

# The browser's file picker only ever gives us a filename, never a real
# path -- the server resolves that name against its own known reference
# locations until a proper reference registry ("the database") exists.
REFERENCE_SEARCH_DIRS = [
    PROJECT_ROOT / "reference-data",
    PROJECT_ROOT / "experiments" / "transcription-smoke-test",
]


def resolve_reference_path(name: str) -> Path:
    safe_name = Path(name).name  # defend against a path-y filename, e.g. "../../etc/passwd"
    for d in REFERENCE_SEARCH_DIRS:
        candidate = d / safe_name
        if candidate.is_file():
            return candidate
    raise HTTPException(404, f"reference '{name}' not found in {[str(d) for d in REFERENCE_SEARCH_DIRS]}")

app = FastAPI()

# The page is opened via a plain static file server (or file://) on a
# different origin than this API -- allow any localhost origin rather
# than pin one, since the dev server's port isn't fixed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def slugify(name: str) -> str:
    """Reference filename -> filesystem-safe slug, e.g.
    'bach-3bars-reference.json' -> 'bach-3bars-reference'."""
    stem = Path(name).stem
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", stem).strip("-")
    if not slug:
        raise HTTPException(400, "reference name has no usable characters")
    return slug


@app.post("/performances")
async def submit_performance(audio: UploadFile = File(...), reference: str = Form(...)):
    reference_path = resolve_reference_path(reference)  # fail fast, before saving anything

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder_name = f"{timestamp}__{slugify(reference)}"
    folder = PERFORMANCES_DIR / folder_name
    folder.mkdir(parents=True, exist_ok=False)

    recording_path = folder / "recording.wav"
    data = await audio.read()
    recording_path.write_bytes(data)

    # CPU-bound model inference -- run_in_threadpool just keeps the
    # event loop free for /health etc. during a submit; this server
    # only ever serves one performer, so there's no real concurrency
    # benefit beyond that.
    duration_sec = await run_in_threadpool(get_audio_duration, recording_path)
    transkun_notes = await run_in_threadpool(run_transkun, recording_path, folder)

    report = await run_in_threadpool(
        evaluate, str(reference_path), str(folder / "transkun.json"), folder_name, duration_sec)
    (folder / "feedback.json").write_text(json.dumps(report, indent=2))

    return {
        "folder": folder_name,
        "recording_bytes": len(data),
        "reference": reference,
        "transkun_note_count": len(transkun_notes),
        "feedback": report,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
