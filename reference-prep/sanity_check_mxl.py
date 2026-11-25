"""
Structural sanity check for an OMR-generated MXL (e.g. from Audiveris),
before trusting it as a make_reference.py input. This reports STRUCTURE
only (part/measure/note counts, pitch range, time/key signatures,
anomalies like empty measures or wildly short/long notes) -- never the
actual note-by-note content -- since that's enough to catch the most
common OMR failure modes (missed/duplicated measures, voices merged or
dropped, garbage pitches from misread accidentals) without requiring a
full manual read-through against the PDF.

Usage: .venv/bin/python reference-prep/sanity_check_mxl.py <path.mxl>
"""
import sys

import music21


def main(path):
    score = music21.converter.parse(path)
    parts = score.parts
    print(f"Parts: {len(parts)}")

    for i, part in enumerate(parts):
        measures = part.getElementsByClass("Measure")
        notes = list(part.recurse().notesAndRests)
        pitched = [n for n in notes if n.isNote or n.isChord]
        rests = [n for n in notes if n.isRest]

        all_pitches = []
        for n in pitched:
            if n.isChord:
                all_pitches.extend(n.pitches)
            else:
                all_pitches.append(n.pitch)

        print(f"\n-- Part {i} ({part.partName or 'unnamed'}) --")
        print(f"  Measures: {len(measures)}")
        print(f"  Notes/chords: {len(pitched)}, Rests: {len(rests)}")
        if all_pitches:
            lowest = min(all_pitches, key=lambda p: p.ps)
            highest = max(all_pitches, key=lambda p: p.ps)
            print(f"  Pitch range: {lowest.nameWithOctave} to {highest.nameWithOctave}")
            # Flag pitches outside a sane 88-key piano range (A0-C8) --
            # a common OMR failure is a misread clef/ottava producing a
            # pitch a register (or more) off from where it should be.
            out_of_range = [p for p in all_pitches if not (21 <= p.midi <= 108)]
            if out_of_range:
                print(f"  ANOMALY: {len(out_of_range)} pitch(es) outside piano range (A0-C8)")

        empty_measures = [m for m in measures if not list(m.recurse().notesAndRests)]
        if empty_measures:
            nums = [m.number for m in empty_measures]
            print(f"  ANOMALY: {len(empty_measures)} empty measure(s): {nums}")

        # Duration outliers often mean a misread note value (e.g. a
        # dotted eighth misread as a whole note).
        durations = [n.quarterLength for n in notes if n.quarterLength > 0]
        if durations:
            very_long = [d for d in durations if d > 8]
            if very_long:
                print(f"  ANOMALY: {len(very_long)} note(s)/rest(s) longer than 8 quarter-lengths (possible misread)")

    ts_list = score.recurse().getElementsByClass("TimeSignature")
    print(f"\nTime signatures found: {len(ts_list)}")
    seen_ts = []
    for ts in ts_list:
        key = str(ts)
        if not seen_ts or seen_ts[-1] != key:
            m = ts.getContextByClass("Measure")
            print(f"  {key} at measure {m.number if m else '?'}")
            seen_ts.append(key)

    ks_list = score.recurse().getElementsByClass("KeySignature")
    print(f"\nKey signatures found: {len(ks_list)}")
    seen_ks = []
    for ks in ks_list:
        key = str(ks)
        if not seen_ks or seen_ks[-1] != key:
            m = ks.getContextByClass("Measure")
            print(f"  {key} at measure {m.number if m else '?'}")
            seen_ks.append(key)

    tempo_marks = score.recurse().getElementsByClass("MetronomeMark")
    print(f"\nTempo marks found: {len(tempo_marks)}")
    for tm in tempo_marks:
        m = tm.getContextByClass("Measure")
        print(f"  {tm.number} bpm at measure {m.number if m else '?'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path.mxl>")
        sys.exit(1)
    main(sys.argv[1])
