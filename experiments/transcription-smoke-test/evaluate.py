"""
Part 3: unify Part 1 (pitch-sequence identity, align_sequence.py), Part 2
(local tempo/timing deviation, local_tempo.py), and the Step 7 noise/
resonance/cross-model extra-note classification (score.py) into one
evaluator pass.

For each reference note:
  - identity is resolved primarily from ByteDance's pitch-sequence
    alignment (Part 1): matched/transposed (direct), substituted
    (flagged as a real wrong note), or deleted (missing from ByteDance).
  - a ByteDance "deleted" ref note is checked against Basic Pitch's own,
    INDEPENDENTLY computed alignment; if Basic Pitch matched/transposed
    it, recover it at reduced confidence (same idea as the earlier
    recovered-via-Basic-Pitch tier, now identity-based instead of
    time-window-based).
  - still-missing or substituted notes are reported as such -- this
    evaluator's whole purpose is to catch exactly these cases, so they
    are never silently dropped.

Every performed note NOT used as a primary or recovery match (either
model's "inserted" notes once recoveries are accounted for) is run
through the Step 7 classification signals (duration, velocity, harmonic
interval to a concurrently-sounding confirmed note, cross-model
corroboration) to tell likely noise/resonance (suppress) from a
genuinely unexplained extra note (flag, don't suppress).

Finally, local tempo/timing deviation (Part 2) is computed over the
FUSED per-reference-note timeline (ByteDance primary, Basic Pitch
filling the identity gaps) -- consecutive confirmed-identity notes only,
skipping over missing/wrong-note reference points rather than bridging
them with a guess.

Usage: python evaluate.py <reference.json> <bytedance.json> <basicpitch.json> [label]
"""
import json
import statistics
import sys

from align_sequence import align
from regions import build_regions, classify_region, corroborating_regions, region_time_range
from scoring import build_beat_cells, beat_for_ref_time, estimate_ref_time, estimate_perf_time, compute_scores

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
CROSS_MODEL_WINDOW_SEC = 0.1

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
    """Context-free version of classify_extra's noise check (no cross-model
    or harmonic-resonance lookup needed here) -- used to keep known noise
    artifacts from polluting region-grouping (Part 4) as if they were a
    performer adding notes."""
    duration = note["offset"] - note["onset"]
    velocity = note.get("velocity", note.get("amplitude", 0) * 127)
    return duration < EXTRA_NOTE_SHORT_DUR_SEC and velocity < EXTRA_NOTE_LOW_VELOCITY


def events_without_noise(events, perf_notes):
    """Drops 'ins' ops for performed notes that look like noise before
    region-grouping -- a cluster of noise artifacts next to an otherwise
    clean passage should not be reported as "notes added"."""
    return [(op, ri, pi) for op, ri, pi in events if op != "ins" or not _is_probable_noise(perf_notes[pi])]


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


def classify_extra(note, other_model_notes, confirmed_notes, source_label, other_label):
    corroborated = [
        c for c in other_model_notes
        if c["pitch"] == note["pitch"] and abs(c["onset"] - note["onset"]) <= CROSS_MODEL_WINDOW_SEC
    ]
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

    if corroborated:
        return "medium-high", "extra note (seen by both models)", \
            f"{other_label} also reports midi={note['pitch']} at t={corroborated[0]['onset']:.3f}s"
    if is_quiet and duration < HARMONIC_RESONANCE_MAX_DUR_SEC and harmonic_match is not None:
        return "low", "likely harmonic resonance (suppressed)", \
            f"dur={duration*1000:.0f}ms, {harmonic_match['pitch']-note['pitch']} semitones below concurrent midi={harmonic_match['pitch']}"
    if is_short and is_quiet:
        return "low", "likely noise artifact (suppressed)", \
            f"dur={duration*1000:.0f}ms, velocity/amplitude={velocity:.0f}, no corroboration"
    return "medium", "unexplained extra note (flagged, not suppressed)", \
        f"dur={duration*1000:.0f}ms, velocity/amplitude={velocity:.0f}, only in {source_label}"


def evaluate(ref_path, bd_path, bp_path, label):
    reference, ref_events = load_reference(ref_path)
    ref_pitches = [n["pitch"] for n in ref_events]
    bd_notes = load_notes(bd_path)
    bp_notes = load_notes(bp_path)

    bd_by_ref, bd_events = identity_pass(ref_pitches, bd_notes)
    bp_by_ref, bp_events = identity_pass(ref_pitches, bp_notes)

    # --- Part 4: group adjacent disputed (del/ins) ops into regions, so a
    # passage with a distorted note count (not just a single wrong note)
    # gets one macro-level verdict instead of forced, over-precise 1:1
    # note pairing. ByteDance is primary, same as the per-note identity
    # pass above; Basic Pitch's own regions are only used to corroborate.
    bd_regions = build_regions(events_without_noise(bd_events, bd_notes))
    bp_regions = build_regions(events_without_noise(bp_events, bp_notes))
    reported_regions = []  # (region, region_label) for regions worth a macro verdict
    ref_idx_to_region_num = {}
    for region in bd_regions:
        region_label, is_trivial = classify_region(region)
        if is_trivial:
            continue
        reported_regions.append((region, region_label))
        region_num = len(reported_regions)
        for ri in region["dels"]:
            ref_idx_to_region_num[ri] = region_num

    results = []
    for i, ref in enumerate(ref_events):
        bd_status, bd_pi = bd_by_ref.get(i, ("missing", None))
        if bd_status == "matched":
            confidence, verdict = "high", "matched"
            perf_time = bd_notes[bd_pi]["onset"]
            bd_notes[bd_pi]["consumed"] = True
            evidence = f"ByteDance t={perf_time:.3f}s"
        elif bd_status == "substituted":
            confidence, verdict = "high", "wrong note"
            perf_time = bd_notes[bd_pi]["onset"]
            bd_notes[bd_pi]["consumed"] = True
            evidence = f"ByteDance played midi={bd_notes[bd_pi]['pitch']} instead of {ref['pitch']} at t={perf_time:.3f}s"
        else:  # missing in ByteDance -- check Basic Pitch's independent alignment
            bp_status, bp_pi = bp_by_ref.get(i, ("missing", None))
            if bp_status == "matched":
                confidence, verdict = "medium", "matched (recovered via Basic Pitch)"
                perf_time = bp_notes[bp_pi]["onset"]
                bp_notes[bp_pi]["consumed"] = True
                evidence = f"Basic Pitch t={perf_time:.3f}s; ByteDance missed"
            elif bp_status == "substituted":
                confidence, verdict = "medium", "wrong note (Basic Pitch)"
                perf_time = bp_notes[bp_pi]["onset"]
                bp_notes[bp_pi]["consumed"] = True
                evidence = f"Basic Pitch played midi={bp_notes[bp_pi]['pitch']} instead of {ref['pitch']} at t={perf_time:.3f}s"
            elif i in ref_idx_to_region_num:
                region_num = ref_idx_to_region_num[i]
                confidence, verdict = "low", f"in disputed region #{region_num}"
                perf_time = None
                evidence = f"part of a passage-level mismatch -- see region #{region_num} below, not a single-note verdict"
            else:
                confidence, verdict = "low", "missing"
                perf_time = None
                evidence = "neither model's alignment found this note"
        results.append({
            "ref_index": i, "ref_onset": ref["onset"], "pitch": ref["pitch"],
            "measure": ref["measure"], "beat": ref["beat"],
            "verdict": verdict, "confidence": confidence, "perf_time": perf_time, "evidence": evidence,
        })

    # A model's own matched/substituted note at a ref index it *wasn't*
    # chosen to resolve is still corroboration of that same reference
    # note's identity -- not a separate event -- so mark it consumed too.
    # Without this, e.g. Basic Pitch's own correct detection of a note
    # ByteDance already matched would wrongly show up below as a "real
    # extra/wrong note seen by both models."
    for i, ref in enumerate(ref_events):
        bd_status, bd_pi = bd_by_ref.get(i, ("missing", None))
        if bd_status in ("matched", "substituted") and bd_pi is not None:
            bd_notes[bd_pi]["consumed"] = True
        bp_status, bp_pi = bp_by_ref.get(i, ("missing", None))
        if bp_status in ("matched", "substituted") and bp_pi is not None:
            bp_notes[bp_pi]["consumed"] = True

    confirmed = [
        (r["ref_onset"], r["perf_time"]) for r in results
        if r["perf_time"] is not None and r["verdict"] not in ("missing",) and "wrong note" not in r["verdict"]
    ]

    # --- octave-confusability recheck ---
    # A "missing" reference note might actually have been played, just
    # detected an octave off by a model's own pitch decision. Found in
    # practice (see README): a note BOTH models independently reported as
    # a clean "matched" turned out, per an independent pYIN pitch check on
    # the raw audio, to likely be an octave slip -- that specific case
    # can't be caught here (both models agreed; there was no competing
    # signal in either model's own output to react to). This recheck
    # covers the detectable half: when a reference note is "missing" but
    # an unconsumed performed note sits near its estimated performance
    # time at exactly one or two octaves away, that's plausible evidence
    # of the SAME note, octave-misread, rather than a genuinely missing
    # one -- without this, it would otherwise be double-penalized, as
    # both a missing note AND (separately, in the extra-notes section) a
    # confusing unexplained extra note.
    for r in results:
        if r["verdict"] != "missing":
            continue
        est_perf_t = estimate_perf_time(r["ref_onset"], confirmed)
        if est_perf_t is None:
            continue
        for notes, model_name in ((bd_notes, "ByteDance"), (bp_notes, "Basic Pitch")):
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
                    f"{model_name} detected midi={candidate['pitch']} "
                    f"({abs(candidate['pitch'] - r['pitch'])} semitones from reference) at "
                    f"t={candidate['onset']:.3f}s -- plausible octave slip, not a missing note"
                )
                break

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

    # --- extra-note classification (performed notes not used as a primary or recovery match) ---
    confirmed_bd_notes = [n for n in bd_notes if n["consumed"]]
    bd_extra = [n for n in bd_notes if not n["consumed"]]
    bp_extra = [n for n in bp_notes if not n["consumed"]]

    extra_results = []
    for n in bd_extra:
        conf, verdict, evidence = classify_extra(n, bp_notes, confirmed_bd_notes, "ByteDance", "Basic Pitch")
        extra_results.append({"model": "ByteDance", "onset": n["onset"], "pitch": n["pitch"],
                               "confidence": conf, "verdict": verdict, "evidence": evidence})
    for n in bp_extra:
        conf, verdict, evidence = classify_extra(n, bd_notes, confirmed_bd_notes, "Basic Pitch", "ByteDance")
        extra_results.append({"model": "Basic Pitch", "onset": n["onset"], "pitch": n["pitch"],
                               "confidence": conf, "verdict": verdict, "evidence": evidence})

    # --- Part 5: beat-level scores (see scoring.py for the full rules) ---
    note_score, rhythm_score, overall, scored_beats = compute_scores(
        reference, ref_events, results, reported_regions, bd_notes, confirmed, extra_results)

    report = build_report(
        label, reference, results, reported_regions, ref_events, bd_notes, bp_notes, bp_regions,
        extra_results, confirmed, note_score, rhythm_score, overall, scored_beats)

    print_report(report)
    return report


def build_report(label, reference, results, reported_regions, ref_events, bd_notes, bp_notes, bp_regions,
                  extra_results, confirmed, note_score, rhythm_score, overall, scored_beats):
    """Everything evaluate() computes, as plain JSON-serializable data --
    the structured counterpart to print_report()'s human-readable text.
    Both are built from the exact same inputs so the UI and the CLI
    report can never silently disagree."""
    n = len(results)
    identity_summary = {
        "total": n,
        "matched": sum(1 for r in results if r["verdict"] == "matched"),
        "recovered": sum(1 for r in results if "recovered" in r["verdict"]),
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
                if region["dels"] else region_time_range(region, ref_events, bd_notes)
            )
            if region["dels"]:
                beats_touched = sorted({(ref_events[ri]["measure"], ref_events[ri]["beat"]) for ri in region["dels"]})
            else:
                estimated = [estimate_ref_time(bd_notes[pi]["onset"], confirmed) for pi in region["ins"]]
                beats_touched = sorted({
                    b for t in estimated if t is not None
                    for b in [beat_for_ref_time(beat_cells, t)] if b is not None
                })
            del_pitches = [ref_events[ri]["pitch"] for ri in region["dels"]]
            ins_pitches = [bd_notes[pi]["pitch"] for pi in region["ins"]]
            corroborated = corroborating_regions(region, bp_regions, ref_events, bd_notes, bp_notes)
            shared_dels = sorted(set(region["dels"]) & {ri for r in corroborated for ri in r["dels"]}) if corroborated else []
            regions_out.append({
                "region_num": region_num, "label": region_label,
                "beats_touched": [{"measure": m, "beat": b} for m, b in beats_touched],
                "ref_time_range": [ref_t0, ref_t1],
                "del_pitches": del_pitches, "ins_pitches": ins_pitches,
                "corroborated_by_basic_pitch": bool(corroborated),
                "corroboration_detail": (
                    f"independently also lost track of ref note(s) at index {shared_dels}" if shared_dels
                    else "has its own disputed region in the same time span" if corroborated
                    else "only ByteDance struggled with this span"
                ),
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
        "scores": scores,
    }


def print_report(report):
    print(f"=== {report['label']} ===")
    print(f"{'m:beat':>8} {'ref_t':>6} {'pitch':>5}  {'verdict':<36} {'conf':<6} {'timing':<11} evidence")
    for r in report["notes"]:
        timing_str = f"{r['timing_tier']} ({r['timing_dev_frac']*100:.0f}%)" if r["timing_tier"] else "-"
        beat_str = f"{r['measure']}:{r['beat']:g}"
        print(f"{beat_str:>8} {r['ref_onset']:6.3f} {r['pitch']:5d}  {r['verdict']:<36} {r['confidence']:<6} {timing_str:<16} {r['evidence']}")

    s, t = report["identity_summary"], report["timing_summary"]
    print()
    print(f"Identity summary: {s['total']} reference notes -- matched={s['matched']} recovered={s['recovered']} "
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
            print(f"    ByteDance: {len(region['del_pitches'])} ref notes unaccounted for (pitches {region['del_pitches']}), "
                  f"{len(region['ins_pitches'])} extra performed notes (pitches {region['ins_pitches']})")
            prefix = "yes --" if region["corroborated_by_basic_pitch"] else "no --"
            print(f"    corroborated by Basic Pitch: {prefix} {region['corroboration_detail']}")
        print()

    print("Extra notes (not used as a primary or recovery match):")
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
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "clip")
