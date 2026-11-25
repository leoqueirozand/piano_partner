"""
Slices a measure range out of a full-score reference (the output of
reference-prep/make_reference.py, one JSON per whole piece, absolute
onset times from the start of the piece) at request time -- so one
offline parse of a score serves any bar selection a performer picks,
instead of needing a fresh offline parse per range.

The result has the exact same shape make_reference.py used to emit
directly for a single baked-in range (source_file, part_indices,
start_measure, end_measure, tempo_bpm, onsets_sec, notes, beats), so
evaluate.py, scoring.py, and the browser-side prototype (js/diff.js)
need no changes to consume it.
"""


def _tempo_at_measure(tempo_marks, measure):
    applicable = [t for t in tempo_marks if t["measure"] <= measure]
    return (applicable[-1] if applicable else tempo_marks[0])["bpm"]


def slice_reference(full_reference, start_measure=None, end_measure=None):
    beats = full_reference.get("beats") or []
    if not beats:
        raise ValueError("full reference has no beat grid to slice against")

    all_measures = [b["measure"] for b in beats]
    piece_lo, piece_hi = min(all_measures), max(all_measures)
    lo = start_measure if start_measure is not None else piece_lo
    hi = end_measure if end_measure is not None else piece_hi
    if lo > hi:
        raise ValueError(f"start_measure ({lo}) is after end_measure ({hi})")

    sel_beats = [b for b in beats if lo <= b["measure"] <= hi]
    if not sel_beats:
        raise ValueError(
            f"no measures {lo}-{hi} found in reference (piece has measures {piece_lo}-{piece_hi})"
        )
    sel_notes = [n for n in full_reference["notes"] if lo <= n["measure"] <= hi]

    zero_sec = min(b["onset_sec"] for b in sel_beats)

    def rezero(item):
        out = dict(item)
        out["onset_sec"] = round(item["onset_sec"] - zero_sec, 4)
        return out

    sliced_notes = sorted((rezero(n) for n in sel_notes), key=lambda n: n["onset_sec"])
    sliced_beats = sorted((rezero(b) for b in sel_beats), key=lambda b: b["onset_sec"])

    tempo_marks = full_reference.get("tempo_marks") or [{"measure": piece_lo, "bpm": full_reference.get("tempo_bpm", 120.0)}]

    return {
        "source_file": full_reference.get("source_file"),
        "part_indices": full_reference.get("part_indices"),
        "start_measure": lo,
        "end_measure": hi,
        "tempo_bpm": _tempo_at_measure(tempo_marks, lo),
        "onsets_sec": [n["onset_sec"] for n in sliced_notes],
        "notes": sliced_notes,
        "beats": sliced_beats,
    }
