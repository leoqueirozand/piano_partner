"""
Slice 1: backend skeleton -- accepts a recorded performance, creates its
own timestamped folder under performances/, saves the raw WAV.
Slice 2: runs Transkun against that WAV and writes its MIDI/JSON
outputs into the same folder (see server/transcribe.py).
Slice 3: runs the evaluator against those outputs and a reference JSON,
writes feedback.json, and returns the structured report.
Slice 4: the reference JSON is now a whole-piece parse -- start_measure/
end_measure (optional form fields) pick the bar range this submission
is judged against, sliced out at request time (see
server/reference_range.py) instead of needing a separate offline parse
per range.
Slice 5: persistence to performances/ is opt-in per submission (a `save`
form field, default false) -- otherwise everything happens in a scratch
temp dir that's deleted before the response is returned, so casual
practice reps don't silently fill the disk.

Run with:
    .venv/bin/uvicorn server.app:app --reload --port 8787
"""
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from server.evaluate import evaluate
from server.reference_range import slice_reference
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
async def submit_performance(
    audio: UploadFile = File(...),
    reference: str = Form(...),
    start_measure: int | None = Form(None),
    end_measure: int | None = Form(None),
    save: bool = Form(False),
):
    reference_path = resolve_reference_path(reference)  # fail fast, before saving anything

    # The reference file is a whole-piece parse (reference-prep/make_reference.py) --
    # start_measure/end_measure pick the bar range this specific submission is
    # judged against; omitting both evaluates the whole piece. See
    # server/reference_range.py's module docstring.
    full_reference = json.loads(reference_path.read_text())
    try:
        sliced_reference = slice_reference(full_reference, start_measure, end_measure)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder_name = f"{timestamp}__{slugify(reference)}"

    # Every submission needs somewhere on disk to run Transkun/the evaluator
    # against -- but by default that's a scratch temp dir, deleted once the
    # response is built, so a casual practice rep doesn't silently persist a
    # WAV + model output forever. `save` opts this one submission into
    # performances/ instead; it's a per-request flag, not a server setting,
    # because the decision ("keep this take") only makes sense per attempt.
    with tempfile.TemporaryDirectory(prefix=f"{folder_name}__") as scratch:
        folder = Path(scratch)

        (folder / "reference-sliced.json").write_text(json.dumps(sliced_reference, indent=2))

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
            evaluate, str(folder / "reference-sliced.json"), str(folder / "transkun.json"), folder_name, duration_sec)
        (folder / "feedback.json").write_text(json.dumps(report, indent=2))

        if save:
            PERFORMANCES_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copytree(folder, PERFORMANCES_DIR / folder_name)

    return {
        "folder": folder_name,
        "saved": save,
        "recording_bytes": len(data),
        "reference": reference,
        "start_measure": sliced_reference["start_measure"],
        "end_measure": sliced_reference["end_measure"],
        "transkun_note_count": len(transkun_notes),
        "feedback": report,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
