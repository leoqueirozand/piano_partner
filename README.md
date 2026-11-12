# Piano Partner — core evaluator prototype

Proves out the central mechanic of the design: capture a performance,
align it against a sheet-music reference, extract a structured
evaluation diff. Deliberately **not** attempting full polyphonic pitch
transcription — onset timing, dynamics, and chroma-based alignment are
the reliable signals under test; predominant pitch contour is included
as medium-confidence, not ground truth.

## 1. Generate a reference template (offline, one-time per piece/section)

```
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python reference-prep/make_reference.py scores/bach-minuet-g.mxl reference-data/bach-minuet-g.json \
  --part 0 --start-measure 1 --end-measure 8 --tempo 120
```

One venv for the whole project (`.venv/` at the project root, Python 3.11 —
newer interpreters don't have wheels yet for the transcription models'
deps): `requirements.txt` covers `music21` (reference generation), the
ByteDance and Basic Pitch transcription models, and the FastAPI backend.
This is already set up — you can also just run:

```
.venv/bin/python reference-prep/make_reference.py scores/bach-minuet-g.mxl reference-data/bach-minuet-g.json \
  --part 0 --start-measure 1 --end-measure 8 --tempo 120
```

`--part 0` is the first part in the score (right hand / top staff for this piece).
`--tempo` overrides the score's own tempo mark; omit it to use the score's
metronome mark if present, else a 120bpm fallback.

## 2. Record performance takes

Drop `.wav`/`.mp3` files into `recordings/`. For the Bach excerpt, aim for a
few contrasting takes (clean, deliberately rushed, deliberately dragged) so
the diff has real signal to show.

## 3. Run the prototype page

No build step — serve the folder and open it:

```
npx serve .
```

(or `python3 -m http.server`), then open the page, pick the reference JSON
and a recording, hit **Analyze**.

## Project structure

- `scores/` — source MusicXML (`.musicxml` / `.mxl`), public-domain only.
- `reference-prep/` — offline Python/music21 step, MusicXML → reference JSON. Not part of the browser app.
- `reference-data/` — generated reference JSON templates (output of the above).
- `recordings/` — test performance audio files.
- `js/` — the browser-side analysis pipeline (Essentia.js onset/pitch/chroma extraction, DTW alignment, diff builder, visualization).
- `index.html` — the prototype page itself.

## Success criteria

- **Tier 1 (must-have, Bach excerpt):** onset timing / tempo-stability numbers visibly track real timing errors across contrasting takes; melody pitch contour roughly follows the expected note sequence.
- **Tier 2 (stretch, Schumann Op. 13 Etude XI excerpt):** same pipeline produces a non-garbage signal on a complex, polyphonic, rubato-heavy piece — even if noisier. A clean failure here is itself a useful result, not a project failure.
