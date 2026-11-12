"""
Part 2: rhythm-instability measurement, built on top of Part 1's
pitch-sequence identity matching (align_sequence.py).

Part 1 proved note identity can be recovered independent of timing, even
under gross tempo instability. This step uses that now-trusted
correspondence to measure timing itself: for each pair of CONSECUTIVE
identity-confirmed notes (match or transpose -- never del/ins, those
aren't real correspondences), compare the inter-onset interval (IOI) in
the reference to the IOI in the performance.

Deliberately local, not global: Step 8's global linear tempo fit (one
slope+offset for the whole excerpt) cannot represent real tempo
instability -- by definition, if the tempo is unstable, no single straight
line fits it. Here, each step's "expected" timing is anchored only on the
previous confirmed note's ACTUAL performed time, not on a global
projection, so one unstable region cannot corrupt the matching or
deviation measurement anywhere else in the excerpt.

A step's local tempo ratio is perf_IOI / ref_IOI (performance seconds per
reference/score second, just like Step 8's global "scale", but computed
one note-to-note transition at a time). The player's own overall chosen
tempo is estimated as the ROBUST (median, outlier-trimmed) average of all
step ratios -- not asked for up front, consistent with the project's
scope: evaluate consistency with the player's own tempo, not against any
fixed bpm.

Usage: python local_tempo.py <reference.json> <transcribed-notes.json> [label]
"""
import statistics
import sys

from align_sequence import align, load_perf_notes, load_ref_notes

MIN_REF_IOI_SEC = 0.05  # treat reference notes closer together than this as "simultaneous" (a chord) -- no meaningful inter-onset interval to measure
TIGHT_DEVIATION_FRACTION = 0.12      # fraction of the player's own beat -- ordinary human timing precision
EXCESSIVE_DEVIATION_FRACTION = 0.30   # likely a real rhythm mistake, not just imprecision


def robust_median_ratio(ratios):
    med = statistics.median(ratios)
    mad = statistics.median([abs(r - med) for r in ratios]) or 1e-6
    kept = [r for r in ratios if abs(r - med) <= 3 * mad]
    return statistics.median(kept) if kept else med


def classify(frac):
    if frac <= TIGHT_DEVIATION_FRACTION:
        return "tight"
    elif frac <= EXCESSIVE_DEVIATION_FRACTION:
        return "noticeable"
    else:
        return "excessive"


def report(label, ref_path, perf_path):
    ref_notes = load_ref_notes(ref_path)   # [(pitch, onset_sec), ...]
    perf_notes = load_perf_notes(perf_path)
    ref_pitches = [p for p, _ in ref_notes]
    perf_pitches = [p for p, _ in perf_notes]

    events, _ = align(ref_pitches, perf_pitches)
    # Keep only real identity correspondences (match/trans), in reference order.
    pairs = sorted(
        [(ri, pi) for op, ri, pi in events if op in ("match", "trans")],
        key=lambda x: x[0],
    )

    steps = []
    for (ref_i, perf_i), (ref_j, perf_j) in zip(pairs, pairs[1:]):
        ref_ioi = ref_notes[ref_j][1] - ref_notes[ref_i][1]
        if ref_ioi < MIN_REF_IOI_SEC:
            continue  # simultaneous (chord) in the score -- no IOI to compare
        perf_ioi = perf_notes[perf_j][1] - perf_notes[perf_i][1]
        ratio = perf_ioi / ref_ioi
        steps.append({
            "ref_from": ref_notes[ref_i][1], "ref_to": ref_notes[ref_j][1],
            "perf_from": perf_notes[perf_i][1], "perf_to": perf_notes[perf_j][1],
            "pitch_from": ref_pitches[ref_i], "pitch_to": ref_pitches[ref_j],
            "ref_ioi": ref_ioi, "perf_ioi": perf_ioi, "ratio": ratio,
        })

    if not steps:
        print(f"=== {label}: no usable (non-simultaneous) consecutive pairs found ===\n")
        return

    baseline_ratio = robust_median_ratio([s["ratio"] for s in steps])

    print(f"=== {label} ===")
    print(f"{len(pairs)} identity-confirmed notes -> {len(steps)} usable (non-simultaneous) timing steps")
    print(f"Player's overall tempo ratio (robust median): {baseline_ratio:.3f}x "
          f"(i.e. {1/baseline_ratio:.3f} reference-seconds pass per performance-second)")
    print()

    print(f"{'ref_t':>6} {'->':>2} {'ref_t':>6}  {'pitch':>5} {'->':>2} {'pitch':>5}  "
          f"{'ref_ioi':>7} {'perf_ioi':>8} {'ratio':>6} {'dev%':>6}  tier")
    tiers = []
    for s in steps:
        dev_frac = abs(s["ratio"] - baseline_ratio) / baseline_ratio
        tier = classify(dev_frac)
        tiers.append(tier)
        print(f"{s['ref_from']:6.3f} {'->':>2} {s['ref_to']:6.3f}  {s['pitch_from']:5d} {'->':>2} {s['pitch_to']:5d}  "
              f"{s['ref_ioi']:7.3f} {s['perf_ioi']:8.3f} {s['ratio']:6.3f} {dev_frac*100:5.0f}%  {tier}")

    print()
    tight = tiers.count("tight")
    noticeable = tiers.count("noticeable")
    excessive = tiers.count("excessive")
    print(f"Rhythm consistency: tight={tight} noticeable={noticeable} excessive={excessive} (of {len(steps)} steps)")
    if excessive:
        print("  -> steps with EXCESSIVE deviation from the player's own established tempo (likely rhythm mistakes):")
        for s in steps:
            dev_frac = abs(s["ratio"] - baseline_ratio) / baseline_ratio
            if classify(dev_frac) == "excessive":
                print(f"     ref {s['ref_from']:.3f}s->{ s['ref_to']:.3f}s (pitch {s['pitch_from']}->{s['pitch_to']}): "
                      f"ratio={s['ratio']:.3f} vs baseline {baseline_ratio:.3f} ({dev_frac*100:.0f}% off)")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    report(sys.argv[3] if len(sys.argv) > 3 else "clip", sys.argv[1], sys.argv[2])
