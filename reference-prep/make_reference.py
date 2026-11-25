#!/usr/bin/env python3
"""
Offline, one-time step: MusicXML -> full-score reference JSON.

Reads a MusicXML (or compressed .mxl) file and emits a JSON file
describing the expected note sequence (onset time, duration, pitches,
chroma vector per note) for the WHOLE piece, with absolute onset times
from the start of the piece -- not just one bar range. A measure range
is no longer picked here: server/reference_range.py slices a range out
of this full reference at request time, so one parse serves any bar
selection instead of needing a fresh offline parse per range (see
server/reference_range.py's module docstring).

Tempo is tracked per written MetronomeMark (piecewise, by quarter-length
offset), not as a single global BPM -- some scores (e.g. the Orfeo ed
Euridice excerpt) notate a tempo change or written-out rit/accel as a
sequence of marks partway through, and onset_sec for notes after that
point must integrate each segment's own tempo rather than one BPM for
the entire piece. The scalar "tempo_bpm" field is kept for the older
browser-side prototype (js/diff.js) as a display-only initial tempo;
it is not used to compute onset_sec.

Usage:
    python make_reference.py <input.musicxml|.mxl> <output.json> \
        [--parts 0,1] [--tempo 120]
"""
import argparse
import json
import sys

from music21 import converter, chord, note, tempo as m21_tempo

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def chroma_vector(pitch_classes):
    vec = [0] * 12
    for pc in pitch_classes:
        vec[pc] = 1
    return vec


def build_tempo_segments(score, override):
    """[(offset_ql, bpm), ...], sorted, covering the whole piece from
    offset 0 -- each segment's bpm holds until the next mark's offset.
    An explicit --tempo override replaces every notated mark with one
    constant tempo for the whole piece."""
    if override:
        return [(0.0, override)]
    marks = score.flatten().getElementsByClass(m21_tempo.MetronomeMark)
    segs = sorted({
        (round(m.getOffsetInHierarchy(score), 6), m.number)
        for m in marks if m.number is not None
    })
    if not segs:
        return [(0.0, 120.0)]  # fallback default (no numeric tempo mark found in score)
    if segs[0][0] > 0.0:
        segs = [(0.0, segs[0][1])] + segs
    return segs


def ql_to_sec(ql, segs):
    """Integrates quarter-length offset `ql` into seconds across
    possibly several tempo segments, rather than one flat bpm."""
    sec = 0.0
    for i, (off, bpm) in enumerate(segs):
        next_off = segs[i + 1][0] if i + 1 < len(segs) else None
        seg_hi = min(ql, next_off) if next_off is not None else ql
        if seg_hi <= off:
            break
        sec += (seg_hi - off) * (60.0 / bpm)
        if next_off is None or ql <= next_off:
            break
    return sec


def measure_for_offset(measures, offset_ql):
    """Which measure a raw score offset (quarter-length, from piece
    start) falls in -- used to label each tempo mark with a measure
    number for the output JSON (debugging/UI only; slicing itself
    works in ql/sec, not measure numbers)."""
    best = measures[0].number
    for m in measures:
        if m.offset <= offset_ql:
            best = m.number
        else:
            break
    return best


def extract_notes(part, segs):
    """Flatten a part into a list of (onset_sec, duration_sec,
    midi_pitches, measure_number, beat), using absolute offsets from
    the start of the piece (no measure-range restriction)."""
    measures = list(part.getElementsByClass("Measure"))

    events = []
    for m in measures:
        # recurse() descends into Voice sub-streams -- some scores (this Bach
        # prelude included) notate multiple voices within one staff/part as
        # separate Voice streams inside each measure, which plain
        # notesAndRests does not see.
        for el in m.recurse().notesAndRests:
            if el.isRest:
                continue
            onset_ql = m.offset + el.getOffsetInHierarchy(m)
            duration_ql = el.quarterLength
            if isinstance(el, chord.Chord):
                midis = [p.midi for p in el.pitches]
            elif isinstance(el, note.Note):
                midis = [el.pitch.midi]
            else:
                continue
            onset_sec = ql_to_sec(onset_ql, segs)
            offset_sec = ql_to_sec(onset_ql + duration_ql, segs)
            # el.beat is music21's own beat position within the measure
            # (1-indexed, resolved against the measure's actual time
            # signature -- including compound meters -- so this is exact,
            # not guessed from tempo_bpm + an assumed meter).
            events.append((onset_sec, offset_sec - onset_sec, midis, m.number, float(el.beat)))
    events.sort(key=lambda e: e[0])
    return events


def extract_beat_grid(part, segs):
    """Every beat cell (measure, beat) in the whole piece, with its
    absolute onset/duration in seconds -- including beats with no note
    in them (rests), which a note-derived grid would silently skip.
    Needed so a disputed region or extra note landing in a rest can
    still be mapped to the beat it broke, not just beats that happen to
    contain a reference note. It's also what a measure-range slice
    (server/reference_range.py) uses to find the exact onset_sec of the
    first beat of the requested range, to re-zero against."""
    measures = list(part.getElementsByClass("Measure"))

    cells = []
    for m in measures:
        ts = m.timeSignature or m.getContextByClass("TimeSignature")
        offsets = ts.getBeatOffsets()
        bar_ql = ts.barDuration.quarterLength
        for i, off_ql in enumerate(offsets):
            end_ql = offsets[i + 1] if i + 1 < len(offsets) else bar_ql
            onset_ql = m.offset + off_ql
            end_ql_abs = m.offset + end_ql
            onset_sec = ql_to_sec(onset_ql, segs)
            offset_sec = ql_to_sec(end_ql_abs, segs)
            cells.append({
                "measure": m.number,
                "beat": i + 1,
                "onset_sec": onset_sec,
                "duration_sec": offset_sec - onset_sec,
            })
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--parts", type=str, default="0", help="Comma-separated part indices to merge (e.g. '0,1' for both hands), or 'all'")
    ap.add_argument("--tempo", type=float, default=None, help="Override tempo in BPM for the whole piece (ignores notated tempo marks)")
    args = ap.parse_args()

    score = converter.parse(args.input)
    parts = score.parts
    if not parts:
        print("No parts found in score", file=sys.stderr)
        sys.exit(1)

    if args.parts == "all":
        part_indices = list(range(len(parts)))
    else:
        part_indices = [int(p) for p in args.parts.split(",")]
    for pi in part_indices:
        if pi >= len(parts):
            print(f"Part index {pi} out of range (score has {len(parts)} parts)", file=sys.stderr)
            sys.exit(1)

    segs = build_tempo_segments(score, args.tempo)

    # Merge events from all selected parts into one flat, time-sorted timeline.
    # Each part keeps its own note events (not chord-combined across parts) --
    # a real performance produces separate note-on events per pitch even when
    # two hands land on the same beat, so the reference should mirror that.
    events = []
    for pi in part_indices:
        events.extend(extract_notes(parts[pi], segs))
    events.sort(key=lambda e: e[0])

    notes_out = []
    for onset_sec, duration_sec, midis, measure_number, beat in events:
        pitch_classes = sorted(set(m % 12 for m in midis))
        notes_out.append({
            "onset_sec": round(onset_sec, 4),
            "duration_sec": round(duration_sec, 4),
            "midi_pitches": midis,
            "pitch_names": [PITCH_CLASSES[pc] for pc in pitch_classes],
            "chroma": chroma_vector(pitch_classes),
            "measure": measure_number,
            "beat": beat,
        })

    # Beat grid (incl. rest-only beats) is only well-defined from a single
    # part's time signature -- all parts of a score share the same measure
    # structure, so the first selected part's grid is the grid.
    beat_cells = extract_beat_grid(parts[part_indices[0]], segs)
    beats_out = [{
        "measure": c["measure"],
        "beat": c["beat"],
        "onset_sec": round(c["onset_sec"], 4),
        "duration_sec": round(c["duration_sec"], 4),
    } for c in beat_cells]

    measures = list(parts[part_indices[0]].getElementsByClass("Measure"))
    tempo_marks_out = [{
        "measure": measure_for_offset(measures, off_ql),
        "offset_ql": off_ql,
        "bpm": bpm,
    } for off_ql, bpm in segs]

    reference = {
        "source_file": args.input,
        "part_indices": part_indices,
        "start_measure": measures[0].number if measures else None,
        "end_measure": measures[-1].number if measures else None,
        "tempo_bpm": segs[0][1],
        "tempo_marks": tempo_marks_out,
        "onsets_sec": [n["onset_sec"] for n in notes_out],
        "notes": notes_out,
        "beats": beats_out,
    }

    with open(args.output, "w") as f:
        json.dump(reference, f, indent=2)

    print(f"Wrote {len(notes_out)} notes ({measures[0].number}-{measures[-1].number if measures else '?'}) "
          f"to {args.output} (initial tempo={segs[0][1]} bpm, {len(segs)} tempo segment(s))")


if __name__ == "__main__":
    main()
