#!/usr/bin/env python3
"""
Offline, one-time step: MusicXML -> reference JSON template.

Reads a MusicXML (or compressed .mxl) file, picks one part and an optional
measure range, and emits a JSON file describing the expected note sequence
(onset time, duration, pitches, chroma vector per note) at a given tempo.
This JSON is what the browser-side prototype aligns a real performance
against (DTW on onset/chroma) -- music21 never runs in the browser.

Usage:
    python make_reference.py <input.musicxml|.mxl> <output.json> \
        [--parts 0,1] [--start-measure 1] [--end-measure 8] [--tempo 120]
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


def select_measures(part, start_measure, end_measure):
    measures = part.getElementsByClass("Measure")
    if start_measure is not None or end_measure is not None:
        lo = start_measure if start_measure is not None else measures[0].number
        hi = end_measure if end_measure is not None else measures[-1].number
        return [m for m in measures if lo <= m.number <= hi]
    return list(measures)


def extract_notes(part, start_measure, end_measure):
    """Flatten a part into a list of (offset_quarterLength, duration_quarterLength,
    midi_pitches, measure_number, beat)."""
    selected = select_measures(part, start_measure, end_measure)

    events = []
    base_offset = selected[0].offset if selected else 0.0
    for m in selected:
        # recurse() descends into Voice sub-streams -- some scores (this Bach
        # prelude included) notate multiple voices within one staff/part as
        # separate Voice streams inside each measure, which plain
        # notesAndRests does not see.
        for el in m.recurse().notesAndRests:
            if el.isRest:
                continue
            onset_ql = (m.offset + el.getOffsetInHierarchy(m)) - base_offset
            duration_ql = el.quarterLength
            if isinstance(el, chord.Chord):
                midis = [p.midi for p in el.pitches]
            elif isinstance(el, note.Note):
                midis = [el.pitch.midi]
            else:
                continue
            # el.beat is music21's own beat position within the measure
            # (1-indexed, resolved against the measure's actual time
            # signature -- including compound meters -- so this is exact,
            # not guessed from tempo_bpm + an assumed meter).
            events.append((onset_ql, duration_ql, midis, m.number, float(el.beat)))
    events.sort(key=lambda e: e[0])
    return events


def extract_beat_grid(part, start_measure, end_measure):
    """Every beat cell (measure, beat) in the selected range, with its
    quarterLength offset/duration -- including beats with no note in them
    (rests), which a note-derived grid would silently skip. Needed so a
    disputed region or extra note landing in a rest can still be mapped to
    the beat it broke, not just beats that happen to contain a reference
    note."""
    selected = select_measures(part, start_measure, end_measure)
    base_offset = selected[0].offset if selected else 0.0

    cells = []
    for m in selected:
        ts = m.timeSignature or m.getContextByClass("TimeSignature")
        offsets = ts.getBeatOffsets()
        bar_ql = ts.barDuration.quarterLength
        for i, off_ql in enumerate(offsets):
            end_ql = offsets[i + 1] if i + 1 < len(offsets) else bar_ql
            cells.append({
                "measure": m.number,
                "beat": i + 1,
                "onset_ql": (m.offset + off_ql) - base_offset,
                "duration_ql": end_ql - off_ql,
            })
    return cells


def find_tempo_bpm(score, override):
    if override:
        return override
    marks = score.flatten().getElementsByClass(m21_tempo.MetronomeMark)
    for mark in marks:
        if mark.number is not None:
            return mark.number
    return 120.0  # fallback default (no numeric tempo mark found in score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--parts", type=str, default="0", help="Comma-separated part indices to merge (e.g. '0,1' for both hands), or 'all'")
    ap.add_argument("--start-measure", type=int, default=None)
    ap.add_argument("--end-measure", type=int, default=None)
    ap.add_argument("--tempo", type=float, default=None, help="Override tempo in BPM (quarter note = 1 beat)")
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

    bpm = find_tempo_bpm(score, args.tempo)
    sec_per_quarter = 60.0 / bpm

    # Merge events from all selected parts into one flat, time-sorted timeline.
    # Each part keeps its own note events (not chord-combined across parts) --
    # a real performance produces separate note-on events per pitch even when
    # two hands land on the same beat, so the reference should mirror that.
    events = []
    for pi in part_indices:
        events.extend(extract_notes(parts[pi], args.start_measure, args.end_measure))
    events.sort(key=lambda e: e[0])

    notes_out = []
    for onset_ql, duration_ql, midis, measure_number, beat in events:
        pitch_classes = sorted(set(m % 12 for m in midis))
        notes_out.append({
            "onset_sec": round(onset_ql * sec_per_quarter, 4),
            "duration_sec": round(duration_ql * sec_per_quarter, 4),
            "midi_pitches": midis,
            "pitch_names": [PITCH_CLASSES[pc] for pc in pitch_classes],
            "chroma": chroma_vector(pitch_classes),
            "measure": measure_number,
            "beat": beat,
        })

    # Beat grid (incl. rest-only beats) is only well-defined from a single
    # part's time signature -- all parts of a score share the same measure
    # structure, so the first selected part's grid is the grid.
    beat_cells = extract_beat_grid(parts[part_indices[0]], args.start_measure, args.end_measure)
    beats_out = [{
        "measure": c["measure"],
        "beat": c["beat"],
        "onset_sec": round(c["onset_ql"] * sec_per_quarter, 4),
        "duration_sec": round(c["duration_ql"] * sec_per_quarter, 4),
    } for c in beat_cells]

    reference = {
        "source_file": args.input,
        "part_indices": part_indices,
        "start_measure": args.start_measure,
        "end_measure": args.end_measure,
        "tempo_bpm": bpm,
        "onsets_sec": [n["onset_sec"] for n in notes_out],
        "notes": notes_out,
        "beats": beats_out,
    }

    with open(args.output, "w") as f:
        json.dump(reference, f, indent=2)

    print(f"Wrote {len(notes_out)} notes to {args.output} (tempo={bpm} bpm)")


if __name__ == "__main__":
    main()
