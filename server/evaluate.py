"""
Part 3: unify Part 1 (pitch-sequence identity, align_sequence.py), Part 2
(local tempo/timing deviation, local_tempo.py), and the Step 7 noise/
resonance extra-note classification (score.py) into one evaluator pass.

For each reference note, identity is resolved from Transkun's
pitch-sequence alignment (Part 1): matched/transposed (direct),
substituted (flagged as a real wrong note), or deleted (missing). A
still-missing or substituted note is reported as such -- this
evaluator's whole purpose is to catch exactly these cases, so they are
never silently dropped.

This evaluator ran two independent models (ByteDance primary, Basic
Pitch as a recovery/corroboration pass) until Transkun replaced both --
see experiments/transcription-smoke-test/README.md, "Experiment 3", and
server/transcribe.py's module docstring for why. The old second-model
machinery (recovered-via-Basic-Pitch identity tier, cross-model
corroboration in region/extra-note reporting) is gone along with it,
not hidden behind a flag -- a direct cross-check showed the secondary
model's disagreements were false alarms far more often than genuine
catches, so there was nothing worth preserving to fall back to. If a
confidence signal is needed later, it'll need a different heuristic
than "the other model disagreed" (see transcribe.py).

Every performed note NOT used as a primary match is run through the
Step 7 classification signals (duration, velocity, harmonic interval to
a concurrently-sounding confirmed note, pedal-sustain bleed) to tell
likely noise/resonance (suppress) from a genuinely unexplained extra
note (flag, don't suppress).

Finally, local tempo/timing deviation (Part 2) is computed over the
confirmed-identity reference-note timeline -- consecutive matched notes
only, skipping over missing/wrong-note reference points rather than
bridging them with a guess.

Usage: python evaluate.py <reference.json> <notes.json> [label]
"""
import json
import statistics
import sys
from pathlib import Path

from server.align_sequence import align
from server.regions import build_regions, classify_region, region_time_range
from server.scoring import build_beat_cells, beat_for_ref_time, estimate_ref_time, estimate_perf_time, compute_scores

MAX_MERGE_GAP_SEC = 0.05
MAX_SUBSTANTIAL_NOTE_DUR_SEC = 0.15

MIN_REF_IOI_SEC = 0.05
TIGHT_DEVIATION_FRACTION = 0.12
EXCESSIVE_DEVIATION_FRACTION = 0.30

EXTRA_NOTE_SHORT_DUR_SEC = 0.08
EXTRA_NOTE_LOW_VELOCITY = 70
HARMONIC_INTERVALS_SEMITONES = (12, 19, 24)
HARMONIC_RESONANCE_MAX_DUR_SEC = 0.15
RESONANCE_LOOKAHEAD_SEC = 0.1

BOUNDARY_NOISE_WINDOW_SEC = 1.0  # pre/post-roll noise (mic pickup before/after playing) clusters here

OCTAVE_CONFUSION_SEMITONES = (12, 24)
OCTAVE_SEARCH_WINDOW_SEC = 0.25


def merge_adjacent_same_pitch(notes):
    notes = sorted(notes, key=lambda n: n["onset"])
    last_by_pitch = {}
    merged = []
    for n in notes:
        prev = last_by_pitch.get(n["pitch"])
        prev_dur = (prev["offset"] - prev["onset"]) if prev is not None else None
        n_dur = n["offset"] - n["onset"]
        looks_like_fragment = prev is not None and (
            prev_dur < MAX_SUBSTANTIAL_NOTE_DUR_SEC or n_dur < MAX_SUBSTANTIAL_NOTE_DUR_SEC
        )
        if prev is not None and looks_like_fragment and n["onset"] - prev["offset"] <= MAX_MERGE_GAP_SEC:
            prev["offset"] = n["offset"]
        else:
            new_note = dict(n)
            merged.append(new_note)
            last_by_pitch[n["pitch"]] = new_note
    return merged


def load_notes(path):
    with open(path) as f:
        notes = json.load(f)
    notes = merge_adjacent_same_pitch(notes)
    notes.sort(key=lambda n: n["onset"])
    for n in notes:
        n["consumed"] = False
    return notes


def load_pedal_events(notes_path):
    """Sustain-pedal on/off windows, from the sibling file transcribe.py
    writes next to transkun.json (see README) -- optional: older
    performance folders recorded before this existed simply have no
    sibling file, and get [] (no pedal-aware suppression), not an
    error."""
    pedal_path = Path(notes_path).parent / "transkun-pedal.json"
    if not pedal_path.is_file():
        return []
    with open(pedal_path) as f:
        return json.load(f)


def load_reference(path):
    with open(path) as f:
        reference = json.load(f)
    ref_events = []
    for n in reference["notes"]:
        for p in sorted(n["midi_pitches"]):
            ref_events.append({"pitch": p, "onset": n["onset_sec"], "measure": n["measure"], "beat": n["beat"]})
    return reference, ref_events


def identity_pass(ref_pitches, perf_notes):
    """Returns ({ref_index: (status, perf_index)}, events) where status is
    one of matched/substituted/missing, from one model's pitch-sequence
    alignment. The raw events are kept too, for region-grouping (Part 4)."""
    perf_pitches = [n["pitch"] for n in perf_notes]
    events, _ = align(ref_pitches, perf_pitches)
    by_ref = {}
    for op, ri, pi in events:
        if op in ("match", "trans"):
            by_ref[ri] = ("matched", pi)
        elif op == "sub":
            by_ref[ri] = ("substituted", pi)
        elif op == "del":
            by_ref[ri] = ("missing", None)
        # 'ins' has no ref index -- handled separately as an unused performed note.
    return by_ref, events


def _is_probable_noise(note):
    """Context-free version of classify_extra's noise check (no
    harmonic-resonance lookup needed here) -- used to keep known noise
    artifacts from polluting region-grouping (Part 4) as if they were a
    performer adding notes."""
    duration = note["offset"] - note["onset"]
    velocity = note.get("velocity", note.get("amplitude", 0) * 127)
    return duration < EXTRA_NOTE_SHORT_DUR_SEC and velocity < EXTRA_NOTE_LOW_VELOCITY


def _is_boundary_noise(note, duration_sec):
    """A note in the first/last BOUNDARY_NOISE_WINDOW_SEC of the recording
    -- the pre/post-roll mic pickup case (performer talking or fumbling
    before/after playing, see README) rather than a real opening/closing
    note. There's no second model left to corroborate against, so any
    note in this window is treated as boundary noise by default."""
    if duration_sec is None:
        return False
    return note["onset"] < BOUNDARY_NOISE_WINDOW_SEC or note["onset"] > duration_sec - BOUNDARY_NOISE_WINDOW_SEC


def _is_pedal_sustain_bleed(note, pedal_events):
    """A quiet onset landing while the sustain pedal was ALREADY down (not
    just engaged at the same moment) -- the decay/resonance-bleed
    signature described in README: a real, deliberately-played note under
    pedal is still struck at a normal velocity, but a note's own
    prolonged decay or a bled-through overtone only looks quiet. Requires
    strict containment (pedal window starts before AND ends after this
    note's onset) -- a note landing right as the pedal goes down is
    ordinary playing, not bleed."""
    velocity = note.get("velocity", note.get("amplitude", 0) * 127)
    if velocity >= EXTRA_NOTE_LOW_VELOCITY:
        return False
    return any(p["onset"] < note["onset"] < p["offset"] for p in pedal_events)


def events_without_noise(events, perf_notes, duration_sec, pedal_events=()):
    """Drops 'ins' ops for performed notes that look like noise before
    region-grouping -- a cluster of noise artifacts next to an otherwise
    clean passage should not be reported as "notes added"."""
    return [
        (op, ri, pi) for op, ri, pi in events
        if op != "ins" or not (
            _is_probable_noise(perf_notes[pi]) or _is_boundary_noise(perf_notes[pi], duration_sec)
            or _is_pedal_sustain_bleed(perf_notes[pi], pedal_events)
        )
    ]


PITCH_MARGIN_SEMITONES = 12  # one octave beyond the reference's own pitch range
DURATION_TOLERANCE_SEC = 0.05  # small buffer for float/segment rounding at the clip boundary


def filter_implausible_notes(notes, model_name, ref_pitches, duration_sec):
    """Drops detected notes that can't plausibly be real performance
    content, before they ever reach identity_pass/region-grouping --
    two patterns found in practice (see README):
      - a recurring ByteDance bass-register artifact at pitches 12-16+
        semitones below anything actually in the piece (pedal noise or
        room rumble, not a performer going a full octave-plus out of
        range);
      - a tail-segment hallucination whose reported offset fell AFTER
        the recording's own end -- a note cannot end after the audio
        stops, so that's a sanity check, not a heuristic.
    Filtered notes are never silently lost -- the caller puts them in
    the report (feedback.json's "filtered_notes") so an over-aggressive
    filter would be visible, not just trusted."""
    floor = min(ref_pitches) - PITCH_MARGIN_SEMITONES
    ceiling = max(ref_pitches) + PITCH_MARGIN_SEMITONES
    kept, filtered = [], []
    for n in notes:
        if n["pitch"] < floor:
            reason = (f"pitch {n['pitch']} is more than {PITCH_MARGIN_SEMITONES} semitones "
                       f"below the reference's lowest note ({min(ref_pitches)})")
        elif n["pitch"] > ceiling:
            reason = (f"pitch {n['pitch']} is more than {PITCH_MARGIN_SEMITONES} semitones "
                       f"above the reference's highest note ({max(ref_pitches)})")
        elif duration_sec is not None and n["offset"] > duration_sec + DURATION_TOLERANCE_SEC:
            reason = f"offset {n['offset']:.3f}s exceeds the recording's own length ({duration_sec:.3f}s)"
        else:
            kept.append(n)
            continue
        filtered.append({**n, "model": model_name, "reason": reason})
    return kept, filtered


def robust_median_ratio(ratios):
    med = statistics.median(ratios)
    mad = statistics.median([abs(r - med) for r in ratios]) or 1e-6
    kept = [r for r in ratios if abs(r - med) <= 3 * mad]
    return statistics.median(kept) if kept else med


def classify_timing(frac):
    if frac <= TIGHT_DEVIATION_FRACTION:
        return "tight"
    elif frac <= EXCESSIVE_DEVIATION_FRACTION:
        return "noticeable"
    return "excessive"


def classify_extra(note, confirmed_notes, duration_sec=None, pedal_events=()):
    duration = note["offset"] - note["onset"]
    velocity = note.get("velocity", note.get("amplitude", 0) * 127)
    is_short = duration < EXTRA_NOTE_SHORT_DUR_SEC
    is_quiet = velocity < EXTRA_NOTE_LOW_VELOCITY

    concurrent = [
        c for c in confirmed_notes
        if c is not note and (c["onset"] - RESONANCE_LOOKAHEAD_SEC) <= note["onset"] <= c["offset"]
    ]
    harmonic_match = None
    for c in concurrent:
        if (c["pitch"] - note["pitch"]) in HARMONIC_INTERVALS_SEMITONES:
            harmonic_match = c
            break

    if _is_boundary_noise(note, duration_sec):
        edge = "start" if note["onset"] < BOUNDARY_NOISE_WINDOW_SEC else "end"
        return "low", "likely pre/post-roll noise (suppressed)", \
            f"t={note['onset']:.3f}s is within {BOUNDARY_NOISE_WINDOW_SEC:.0f}s of the recording's {edge}"
    if is_quiet and duration < HARMONIC_RESONANCE_MAX_DUR_SEC and harmonic_match is not None:
        return "low", "likely harmonic resonance (suppressed)", \
            f"dur={duration*1000:.0f}ms, {harmonic_match['pitch']-note['pitch']} semitones below concurrent midi={harmonic_match['pitch']}"
    if _is_pedal_sustain_bleed(note, pedal_events):
        pedal_window = next(p for p in pedal_events if p["onset"] < note["onset"] < p["offset"])
        return "low", "likely pedal-sustain bleed (suppressed)", \
            f"dur={duration*1000:.0f}ms, velocity/amplitude={velocity:.0f}, under pedal held {note['onset']-pedal_window['onset']:.3f}s (down since t={pedal_window['onset']:.3f}s)"
    if is_short and is_quiet:
        return "low", "likely noise artifact (suppressed)", \
            f"dur={duration*1000:.0f}ms, velocity/amplitude={velocity:.0f}"
    return "medium", "unexplained extra note (flagged, not suppressed)", \
        f"dur={duration*1000:.0f}ms, velocity/amplitude={velocity:.0f}"


def evaluate(ref_path, notes_path, label, duration_sec=None):
    reference, ref_events = load_reference(ref_path)
    ref_pitches = [n["pitch"] for n in ref_events]
    notes = load_notes(notes_path)
    pedal_events = load_pedal_events(notes_path)

    notes, filtered_notes = filter_implausible_notes(notes, "Transkun", ref_pitches, duration_sec)

    by_ref, events = identity_pass(ref_pitches, notes)

    # --- Part 4: group adjacent disputed (del/ins) ops into regions, so a
    # passage with a distorted note count (not just a single wrong note)
    # gets one macro-level verdict instead of forced, over-precise 1:1
    # note pairing.
    regions = build_regions(events_without_noise(events, notes, duration_sec, pedal_events))
    reported_regions = []  # (region, region_label) for regions worth a macro verdict
    ref_idx_to_region_num = {}
    for region in regions:
        region_label, is_trivial = classify_region(region)
        if is_trivial:
            continue
        reported_regions.append((region, region_label))
        region_num = len(reported_regions)
        for ri in region["dels"]:
            ref_idx_to_region_num[ri] = region_num

    results = []
    for i, ref in enumerate(ref_events):
        status, pi = by_ref.get(i, ("missing", None))
        if status == "matched":
            confidence, verdict = "high", "matched"
            perf_time = notes[pi]["onset"]
            notes[pi]["consumed"] = True
            evidence = f"Transkun t={perf_time:.3f}s"
        elif status == "substituted":
            confidence, verdict = "high", "wrong note"
            perf_time = notes[pi]["onset"]
            notes[pi]["consumed"] = True
            evidence = f"Transkun played midi={notes[pi]['pitch']} instead of {ref['pitch']} at t={perf_time:.3f}s"
        elif i in ref_idx_to_region_num:
            region_num = ref_idx_to_region_num[i]
            confidence, verdict = "low", f"in disputed region #{region_num}"
            perf_time = None
            evidence = f"part of a passage-level mismatch -- see region #{region_num} below, not a single-note verdict"
        else:
            confidence, verdict = "low", "missing"
            perf_time = None
            evidence = "Transkun's alignment didn't find this note"
        results.append({
            "ref_index": i, "ref_onset": ref["onset"], "pitch": ref["pitch"],
            "measure": ref["measure"], "beat": ref["beat"],
            "verdict": verdict, "confidence": confidence, "perf_time": perf_time, "evidence": evidence,
        })

    confirmed = [
        (r["ref_onset"], r["perf_time"]) for r in results
        if r["perf_time"] is not None and r["verdict"] not in ("missing",) and "wrong note" not in r["verdict"]
    ]

    # --- octave-confusability recheck ---
    # A "missing" reference note might actually have been played, just
    # detected an octave off by Transkun's own pitch decision -- a
    # reproducible failure mode found in the older ByteDance-based
    # pipeline (see README) and worth guarding against even though
    # Transkun didn't exhibit it on any recording tested so far. When a
    # reference note is "missing" but an unconsumed performed note sits
    # near its estimated performance time at exactly one or two octaves
    # away, that's plausible evidence of the SAME note, octave-misread,
    # rather than a genuinely missing one -- without this, it would
    # otherwise be double-penalized, as both a missing note AND
    # (separately, in the extra-notes section) a confusing unexplained
    # extra note.
    for r in results:
        if r["verdict"] != "missing":
            continue
        est_perf_t = estimate_perf_time(r["ref_onset"], confirmed)
        if est_perf_t is None:
            continue
        candidate = next(
            (n for n in notes if not n["consumed"]
             and abs(n["pitch"] - r["pitch"]) in OCTAVE_CONFUSION_SEMITONES
             and abs(n["onset"] - est_perf_t) <= OCTAVE_SEARCH_WINDOW_SEC),
            None,
        )
        if candidate is not None:
            candidate["consumed"] = True
            r["verdict"] = "matched (octave-uncertain)"
            r["confidence"] = "medium"
            r["perf_time"] = candidate["onset"]
            r["evidence"] = (
                f"Transkun detected midi={candidate['pitch']} "
                f"({abs(candidate['pitch'] - r['pitch'])} semitones from reference) at "
                f"t={candidate['onset']:.3f}s -- plausible octave slip, not a missing note"
            )

    # --- Part 2, over the fused per-reference-note timeline ---
    steps = []
    for (ref_t, perf_t), (ref_t2, perf_t2) in zip(confirmed, confirmed[1:]):
        ref_ioi = ref_t2 - ref_t
        if ref_ioi < MIN_REF_IOI_SEC:
            continue
        perf_ioi = perf_t2 - perf_t
        steps.append({"ref_to": ref_t2, "ratio": perf_ioi / ref_ioi})

    baseline = robust_median_ratio([s["ratio"] for s in steps]) if steps else None
    timing_by_ref_onset = {}
    if baseline:
        for s in steps:
            frac = abs(s["ratio"] - baseline) / baseline
            timing_by_ref_onset[s["ref_to"]] = (classify_timing(frac), frac)

    for r in results:
        t = timing_by_ref_onset.get(r["ref_onset"])
        r["timing_tier"], r["timing_dev_frac"] = t if t else (None, None)

    # --- extra-note classification (performed notes not used as a primary match) ---
    confirmed_notes = [n for n in notes if n["consumed"]]
    extra_notes = [n for n in notes if not n["consumed"]]

    extra_results = []
    for n in extra_notes:
        conf, verdict, evidence = classify_extra(n, confirmed_notes, duration_sec, pedal_events)
        extra_results.append({"model": "Transkun", "onset": n["onset"], "pitch": n["pitch"],
                               "confidence": conf, "verdict": verdict, "evidence": evidence})

    # --- Part 5: beat-level scores (see scoring.py for the full rules) ---
    note_score, rhythm_score, overall, scored_beats = compute_scores(reference, results, confirmed, extra_results)

    report = build_report(
        label, reference, results, reported_regions, ref_events, notes,
        extra_results, confirmed, note_score, rhythm_score, overall, scored_beats, filtered_notes)

    print_report(report)
    return report


def build_report(label, reference, results, reported_regions, ref_events, notes,
                  extra_results, confirmed, note_score, rhythm_score, overall, scored_beats, filtered_notes):
    """Everything evaluate() computes, as plain JSON-serializable data --
    the structured counterpart to print_report()'s human-readable text.
    Both are built from the exact same inputs so the UI and the CLI
    report can never silently disagree."""
    n = len(results)
    identity_summary = {
        "total": n,
        "matched": sum(1 for r in results if r["verdict"] == "matched"),
        "octave_uncertain": sum(1 for r in results if "octave-uncertain" in r["verdict"]),
        "wrong_note": sum(1 for r in results if "wrong note" in r["verdict"]),
        "missing": sum(1 for r in results if r["verdict"] == "missing"),
        "in_disputed_region": sum(1 for r in results if r["verdict"].startswith("in disputed region")),
    }

    tiers = [r["timing_tier"] for r in results if r["timing_tier"]]
    timing_summary = {
        "tight": tiers.count("tight"), "noticeable": tiers.count("noticeable"),
        "excessive": tiers.count("excessive"), "measured": len(tiers),
    }

    regions_out = []
    if reported_regions:
        beat_cells = build_beat_cells(reference)
        for region_num, (region, region_label) in enumerate(reported_regions, start=1):
            ref_t0, ref_t1 = (
                (ref_events[region["dels"][0]]["onset"], ref_events[region["dels"][-1]]["onset"])
                if region["dels"] else region_time_range(region, ref_events, notes)
            )
            if region["dels"]:
                beats_touched = sorted({(ref_events[ri]["measure"], ref_events[ri]["beat"]) for ri in region["dels"]})
            else:
                estimated = [estimate_ref_time(notes[pi]["onset"], confirmed) for pi in region["ins"]]
                beats_touched = sorted({
                    b for t in estimated if t is not None
                    for b in [beat_for_ref_time(beat_cells, t)] if b is not None
                })
            del_pitches = [ref_events[ri]["pitch"] for ri in region["dels"]]
            ins_pitches = [notes[pi]["pitch"] for pi in region["ins"]]
            regions_out.append({
                "region_num": region_num, "label": region_label,
                "beats_touched": [{"measure": m, "beat": b} for m, b in beats_touched],
                "ref_time_range": [ref_t0, ref_t1],
                "del_pitches": del_pitches, "ins_pitches": ins_pitches,
            })

    zeroed = sorted(k for k, v in scored_beats.items() if v == 0.0)
    scores = {
        "note_accuracy": note_score, "rhythm_accuracy": rhythm_score, "overall": overall,
        "scored_beats": [{"measure": m, "beat": b, "value": v} for (m, b), v in sorted(scored_beats.items())],
        "zeroed_beats": [{"measure": m, "beat": b} for m, b in zeroed],
    }

    return {
        "label": label,
        "notes": results,
        "identity_summary": identity_summary,
        "timing_summary": timing_summary,
        "regions": regions_out,
        "extra_notes": sorted(extra_results, key=lambda e: e["onset"]),
        "filtered_notes": filtered_notes,
        "scores": scores,
    }


def print_report(report):
    print(f"=== {report['label']} ===")
    if report["filtered_notes"]:
        print("Filtered before scoring (implausible pitch or past the recording's end -- see README):")
        for f in report["filtered_notes"]:
            print(f"  [{f['model']}] t={f['onset']:.3f}s midi={f['pitch']}  {f['reason']}")
        print()
    print(f"{'m:beat':>8} {'ref_t':>6} {'pitch':>5}  {'verdict':<36} {'conf':<6} {'timing':<11} evidence")
    for r in report["notes"]:
        timing_str = f"{r['timing_tier']} ({r['timing_dev_frac']*100:.0f}%)" if r["timing_tier"] else "-"
        beat_str = f"{r['measure']}:{r['beat']:g}"
        print(f"{beat_str:>8} {r['ref_onset']:6.3f} {r['pitch']:5d}  {r['verdict']:<36} {r['confidence']:<6} {timing_str:<16} {r['evidence']}")

    s, t = report["identity_summary"], report["timing_summary"]
    print()
    print(f"Identity summary: {s['total']} reference notes -- matched={s['matched']} "
          f"octave_uncertain={s['octave_uncertain']} wrong_note={s['wrong_note']} missing={s['missing']} in_disputed_region={s['in_disputed_region']}")
    print(f"Timing summary: tight={t['tight']} noticeable={t['noticeable']} excessive={t['excessive']} (of {t['measured']} measured)")

    print()
    if report["regions"]:
        print("Disputed regions (identity could not be cleanly recovered -- see README Part 4):")
        for region in report["regions"]:
            beats_touched = region["beats_touched"]
            beats_str = ", ".join(f"{b['measure']}:{b['beat']:g}" for b in beats_touched) if beats_touched else \
                "(outside the evaluated range -- likely continued past the excerpt)"
            ref_t0, ref_t1 = region["ref_time_range"]
            print(f"  Region #{region['region_num']}: beats [{beats_str}] (ref t={ref_t0:.3f}s-{ref_t1:.3f}s) -- {region['label']}")
            print(f"    {len(region['del_pitches'])} ref notes unaccounted for (pitches {region['del_pitches']}), "
                  f"{len(region['ins_pitches'])} extra performed notes (pitches {region['ins_pitches']})")
        print()

    print("Extra notes (not used as a primary match):")
    for e in report["extra_notes"]:
        print(f"  [{e['model']}] t={e['onset']:.3f}s midi={e['pitch']}  {e['verdict']:<40} conf={e['confidence']:<11} {e['evidence']}")
    print()

    sc = report["scores"]
    zeroed = sc["zeroed_beats"]
    print("Scores:")
    print(f"  Note accuracy:   {sc['note_accuracy']:.1f}%  ({len(sc['scored_beats'])} scored beats, {len(zeroed)} at zero)"
          if sc["note_accuracy"] is not None else "  Note accuracy:   n/a (no scored beats)")
    print(f"  Rhythm accuracy: {sc['rhythm_accuracy']:.1f}%" if sc["rhythm_accuracy"] is not None else "  Rhythm accuracy: n/a (no measured timing steps)")
    print(f"  Overall (70/30): {sc['overall']:.1f}%" if sc["overall"] is not None else "  Overall:         n/a")
    if zeroed:
        zeroed_str = ", ".join(f"{b['measure']}:{b['beat']:g}" for b in zeroed)
        print(f"  Zero-value beats: {zeroed_str}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "clip")
