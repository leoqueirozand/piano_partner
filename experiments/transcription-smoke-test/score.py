"""
Prototype: reference-aware, confidence-tiered scoring of a performance,
fusing two independent transcription models (ByteDance primary,
Basic Pitch secondary) against a MusicXML-derived reference.

Not production code -- proving out the matching/tiering logic on one
real recording before building it into the app pipeline.
"""
import json
import statistics

REFERENCE_PATH = "bach-3bars-reference.json"
BYTEDANCE_PATH = "bach-clip-bytedance.json"
BASICPITCH_PATH = "bach-clip-basicpitch.json"

TIME_WINDOW_SEC = 0.25  # how far a transcribed note can be from the expected (scaled) reference time and still count as "the same event"

# Signals for classifying an unmatched/"extra" transcribed note (Steps 1-2's
# noise/resonance problem) -- separate thresholds from the reference-matching
# side above.
EXTRA_NOTE_SHORT_DUR_SEC = 0.08
EXTRA_NOTE_LOW_VELOCITY = 70
HARMONIC_INTERVALS_SEMITONES = (12, 19, 24)  # octave, octave+fifth, 2 octaves
HARMONIC_RESONANCE_MAX_DUR_SEC = 0.15  # more lenient than EXTRA_NOTE_SHORT_DUR_SEC -- resonance tails run a bit longer than pure onset-detector noise (observed: 116ms)
RESONANCE_LOOKAHEAD_SEC = 0.1  # a resonance can be detected slightly *before* the real note's own onset lands, not just during its sustain

# Collapses a sustained note that a model's onset detector split into several
# same-pitch events back-to-back. Already applied to the ByteDance output
# when bach-clip-bytedance.json was generated (Step 5); Basic Pitch's raw
# output has the same failure mode and was never run through this, so it's
# applied here instead.
MAX_MERGE_GAP_SEC = 0.05

# A genuine split-onset artifact looks like one note followed by a short
# trailing fragment. Two substantial notes of the same pitch landing
# back-to-back with a tiny gap are two real distinct notes, not an
# artifact -- merging them would silently erase real timing information.
MAX_SUBSTANTIAL_NOTE_DUR_SEC = 0.15


def merge_adjacent_same_pitch(notes):
    # Compares against the last note of the *same pitch* seen so far, not
    # just the previous note in time overall -- a different pitch sounding
    # in between (e.g. the other hand) must not block a same-pitch merge.
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


with open(REFERENCE_PATH) as f:
    reference = json.load(f)
with open(BYTEDANCE_PATH) as f:
    bytedance = json.load(f)
with open(BASICPITCH_PATH) as f:
    basicpitch = json.load(f)

basicpitch_before_merge = len(basicpitch)
basicpitch = merge_adjacent_same_pitch(basicpitch)
if len(basicpitch) != basicpitch_before_merge:
    print(f"Merged Basic Pitch same-pitch split onsets (gap <= {MAX_MERGE_GAP_SEC*1000:.0f}ms): {basicpitch_before_merge} -> {len(basicpitch)}")

for n in bytedance:
    n["consumed"] = False
for n in basicpitch:
    n["consumed"] = False

# Flatten reference chords into one (onset, pitch) event per note, since a
# real performance produces one note-on per pitch even for notes written
# together as a chord/two-hand coincidence.
ref_events = []
for n in reference["notes"]:
    for pitch in n["midi_pitches"]:
        ref_events.append({"onset": n["onset_sec"], "duration": n["duration_sec"], "pitch": pitch})
ref_events.sort(key=lambda e: e["onset"])

# --- Tempo normalization ---
# Players choose their own tempo; we don't ask for one up front and we are
# NOT modeling deliberate expressive tempo change (rubato/accelerando) --
# this evaluator isn't judging musicality, only whether the section was
# played correctly. The assumption is a single, roughly constant tempo
# chosen by the player for this excerpt. "Excessive deviation" from that
# chosen tempo is itself a signal worth surfacing (suggests the player
# doesn't know the piece's rhythm), not something to explain away.
#
# Two passes: (1) a rough anchor-based scale (first/last note only, wide
# matching window) just to get enough matches to work with, then (2) a
# robust linear refit over those matches (median-based outlier trimming,
# so a few wrong matches -- or a real rhythm mistake -- can't drag the
# whole-excerpt tempo estimate off course), then re-match everything
# against the refined, tighter estimate.
ROUGH_TIME_WINDOW_SEC = 0.25
REFINED_TIME_WINDOW_SEC = 0.12

# Deviation tiers are expressed as a fraction of the PERFORMER'S OWN beat
# (quarter note) duration at their fitted tempo, not absolute milliseconds
# -- the same absolute timing error matters more at a fast tempo
# (sixteenth notes) than a slow one.
TIGHT_DEVIATION_FRACTION = 0.12       # ordinary human timing precision
EXCESSIVE_DEVIATION_FRACTION = 0.30   # likely a rhythm mistake, not just imprecision

ref_start = ref_events[0]["onset"]
ref_end = max(e["onset"] + e["duration"] for e in ref_events)
perf_start_rough = min(n["onset"] for n in bytedance)
perf_end_rough = max(n["offset"] for n in bytedance)
rough_scale = (perf_end_rough - perf_start_rough) / (ref_end - ref_start)


def rough_to_perf_time(ref_onset):
    return perf_start_rough + (ref_onset - ref_start) * rough_scale


def fit_linear(xs, ys):
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs) or 1e-9
    slope = num / den
    intercept = mean_y - slope * mean_x
    return slope, intercept


def robust_tempo_fit(pairs):
    """perf_onset ~= intercept + slope*ref_onset, least squares with
    iterative median-absolute-deviation outlier trimming."""
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    slope, intercept = fit_linear(xs, ys)
    for _ in range(3):
        residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        med = statistics.median(residuals)
        mad = statistics.median([abs(r - med) for r in residuals]) or 1e-6
        kept = [(x, y) for x, y, r in zip(xs, ys, residuals) if abs(r - med) <= 3 * mad]
        if len(kept) == len(xs) or len(kept) < 4:
            break
        xs = [k[0] for k in kept]
        ys = [k[1] for k in kept]
        slope, intercept = fit_linear(xs, ys)
    return slope, intercept, len(xs)


def find_candidates(events, expected_time, pitch=None, pitch_class_only=False, unconsumed_only=False, window_sec=TIME_WINDOW_SEC):
    out = []
    for n in events:
        if unconsumed_only and n["consumed"]:
            continue
        dt = n["onset"] - expected_time
        if abs(dt) > window_sec:
            continue
        if pitch is not None:
            if pitch_class_only:
                if n["pitch"] % 12 != pitch % 12:
                    continue
            elif n["pitch"] != pitch:
                continue
        out.append((abs(dt), n))
    out.sort(key=lambda x: x[0])
    return [n for _, n in out]


def reset_consumed():
    for n in bytedance:
        n["consumed"] = False
    for n in basicpitch:
        n["consumed"] = False


def match_references(to_perf_time_fn, window_sec):
    results = []
    high_confidence_pairs = []  # (ref_onset, perf_onset) for ByteDance-exact matches only -- used to refit tempo
    for ref in ref_events:
        expected_t = to_perf_time_fn(ref["onset"])
        pitch = ref["pitch"]

        bd_exact = find_candidates(bytedance, expected_t, pitch, unconsumed_only=True, window_sec=window_sec)
        bp_exact = find_candidates(basicpitch, expected_t, pitch, unconsumed_only=True, window_sec=window_sec)
        bd_octave = find_candidates(bytedance, expected_t, pitch, pitch_class_only=True, unconsumed_only=True, window_sec=window_sec)
        bp_octave = find_candidates(basicpitch, expected_t, pitch, pitch_class_only=True, unconsumed_only=True, window_sec=window_sec)

        timing_error = None
        if bd_exact:
            # Primary model confirms pitch and timing directly.
            confidence = "high"
            verdict = "matched"
            evidence = f"ByteDance exact match at t={bd_exact[0]['onset']:.3f}s"
            bd_exact[0]["consumed"] = True
            if bp_exact:
                bp_exact[0]["consumed"] = True
            timing_error = bd_exact[0]["onset"] - expected_t
            high_confidence_pairs.append((ref["onset"], bd_exact[0]["onset"]))
        elif bp_exact:
            # Primary model missed it (e.g. masked by a simultaneous louder
            # note), secondary model recovers it -- report as real, but at
            # reduced confidence since only one model saw it.
            confidence = "medium"
            verdict = "matched (recovered via Basic Pitch)"
            evidence = f"Basic Pitch exact match at t={bp_exact[0]['onset']:.3f}s, amp={bp_exact[0].get('amplitude', '?')}; ByteDance missed"
            bp_exact[0]["consumed"] = True
            timing_error = bp_exact[0]["onset"] - expected_t
        elif bd_octave or bp_octave:
            # Right pitch class, wrong octave, in at least one model -- flag as
            # an octave-error candidate rather than a flat miss or flat match.
            confidence = "medium"
            verdict = "octave mismatch"
            src = bd_octave[0] if bd_octave else bp_octave[0]
            which = "ByteDance" if bd_octave else "Basic Pitch"
            evidence = f"{which} found pitch class match at wrong octave: midi={src['pitch']} (expected {pitch}) t={src['onset']:.3f}s"
            src["consumed"] = True
            timing_error = src["onset"] - expected_t
        else:
            confidence = "low"
            verdict = "missing"
            evidence = "neither model found this pitch near the expected time"

        results.append({
            "ref_onset": ref["onset"],
            "expected_perf_time": round(expected_t, 3),
            "expected_pitch": pitch,
            "verdict": verdict,
            "confidence": confidence,
            "evidence": evidence,
            "timing_error": timing_error,
        })
    return results, high_confidence_pairs


def classify_timing(timing_error, beat_sec):
    if timing_error is None:
        return None, None
    frac = abs(timing_error) / beat_sec
    if frac <= TIGHT_DEVIATION_FRACTION:
        return "tight", frac
    elif frac <= EXCESSIVE_DEVIATION_FRACTION:
        return "noticeable", frac
    else:
        return "excessive", frac


# Pass 1: rough anchor-based scale, wide window, just to collect enough
# ByteDance-exact matches to fit a tempo from.
reset_consumed()
_, high_conf_pairs = match_references(rough_to_perf_time, ROUGH_TIME_WINDOW_SEC)

if len(high_conf_pairs) >= 4:
    slope, intercept, used_n = robust_tempo_fit(high_conf_pairs)
else:
    slope, intercept, used_n = rough_scale, perf_start_rough - ref_start * rough_scale, len(high_conf_pairs)


def to_perf_time(ref_onset):
    return intercept + slope * ref_onset


beat_sec_ref = 60.0 / reference["tempo_bpm"]  # one quarter note, in score/reference time
beat_sec_perf = slope * beat_sec_ref          # the performer's own beat duration, in real seconds

print(f"Rough anchor-based scale: {rough_scale:.3f}x (first/last note only)")
print(f"Robust tempo fit: scale={slope:.3f}x, offset={intercept:.3f}s "
      f"({used_n}/{len(high_conf_pairs)} high-confidence matches kept after outlier trimming)")
print(f"Performer's beat (quarter note): {beat_sec_perf*1000:.0f}ms "
      f"-- score tempo {reference['tempo_bpm']:.0f} bpm, performer's effective tempo {60/beat_sec_perf:.1f} bpm")
print()

# Pass 2: re-match everything against the refined tempo estimate, with a
# tighter window now that we trust the estimate more.
reset_consumed()
results, _ = match_references(to_perf_time, REFINED_TIME_WINDOW_SEC)

print(f"{'ref_t':>6} {'perf_t':>7} {'pitch':>5}  {'verdict':<28} {'conf':<6} {'timing':>8} {'tier':<11} evidence")
for r in results:
    tier, frac = classify_timing(r["timing_error"], beat_sec_perf)
    timing_str = f"{r['timing_error']*1000:+.0f}ms" if r["timing_error"] is not None else "n/a"
    print(f"{r['ref_onset']:6.3f} {r['expected_perf_time']:7.3f} {r['expected_pitch']:5d}  {r['verdict']:<28} {r['confidence']:<6} {timing_str:>8} {(tier or '-'):<11} {r['evidence']}")

print()
n = len(results)
matched = sum(1 for r in results if r["verdict"] == "matched")
recovered = sum(1 for r in results if "recovered" in r["verdict"])
octave = sum(1 for r in results if r["verdict"] == "octave mismatch")
missing = sum(1 for r in results if r["verdict"] == "missing")
print(f"Summary: {n} reference notes -- matched={matched} recovered={recovered} octave_mismatch={octave} missing={missing}")

timing_tiers = [classify_timing(r["timing_error"], beat_sec_perf)[0] for r in results if r["timing_error"] is not None]
tight = timing_tiers.count("tight")
noticeable = timing_tiers.count("noticeable")
excessive = timing_tiers.count("excessive")
print(f"Rhythm consistency ({len(timing_tiers)} notes with a timing measurement): "
      f"tight={tight} noticeable={noticeable} excessive={excessive}")
if excessive:
    print("  -> notes with 'excessive' timing deviation (possible rhythm mistake):")
    for r in results:
        tier, frac = classify_timing(r["timing_error"], beat_sec_perf)
        if tier == "excessive":
            print(f"     ref_t={r['ref_onset']:.3f}s pitch={r['expected_pitch']} "
                  f"timing_error={r['timing_error']*1000:+.0f}ms ({frac*100:.0f}% of beat)")


# --- Step 2 (symmetric check): classify transcribed notes that weren't
# claimed by any reference match. These are the candidates for "real
# extra/wrong note" vs. the noise/resonance artifacts from Steps 1-2. ---
CROSS_MODEL_WINDOW_SEC = 0.1

confirmed_notes = [n for n in bytedance if n["consumed"]]  # ByteDance notes that matched a reference pitch -- context for the harmonic-interval check


def classify_extra(note, other_model_events, source_label, other_label):
    corroborated = find_candidates(other_model_events, note["onset"], note["pitch"])
    corroborated = [c for c in corroborated if abs(c["onset"] - note["onset"]) <= CROSS_MODEL_WINDOW_SEC]

    duration = note["offset"] - note["onset"]
    velocity = note.get("velocity", note.get("amplitude", 0) * 127)
    is_short = duration < EXTRA_NOTE_SHORT_DUR_SEC
    is_quiet = velocity < EXTRA_NOTE_LOW_VELOCITY

    # A resonance can be picked up slightly before the real note's own
    # detected onset, not just while it's already sounding -- so the window
    # extends a bit earlier than c.onset, not just [c.onset, c.offset].
    concurrent = [
        c for c in confirmed_notes
        if c is not note
        and (c["onset"] - RESONANCE_LOOKAHEAD_SEC) <= note["onset"] <= c["offset"]
    ]
    harmonic_match = None
    for c in concurrent:
        if (c["pitch"] - note["pitch"]) in HARMONIC_INTERVALS_SEMITONES:
            harmonic_match = c
            break

    if corroborated:
        return "medium-high", "extra note (seen by both models)", \
            f"{other_label} also reports midi={note['pitch']} at t={corroborated[0]['onset']:.3f}s -- likely a real extra/wrong note, not an artifact of one model"
    if is_quiet and duration < HARMONIC_RESONANCE_MAX_DUR_SEC and harmonic_match is not None:
        return "low", "likely harmonic resonance (suppressed)", \
            f"dur={duration*1000:.0f}ms, interval {harmonic_match['pitch']-note['pitch']} semitones below concurrently-sounding midi={harmonic_match['pitch']} (t={harmonic_match['onset']:.3f}-{harmonic_match['offset']:.3f}s)"
    if is_short and is_quiet:
        return "low", "likely noise artifact (suppressed)", \
            f"dur={duration*1000:.0f}ms, velocity/amplitude={velocity:.0f}, no corroboration, no harmonic explanation"
    return "medium", "unexplained extra note (flagged, not suppressed)", \
        f"dur={duration*1000:.0f}ms, velocity/amplitude={velocity:.0f}, only in {source_label}, no harmonic explanation -- could be a real mistake, reported rather than guessed away"


print("\n--- Extra notes: ByteDance notes not claimed by any reference match ---")
bd_extra = [n for n in bytedance if not n["consumed"]]
for note in sorted(bd_extra, key=lambda n: n["onset"]):
    confidence, verdict, evidence = classify_extra(note, basicpitch, "ByteDance", "Basic Pitch")
    print(f"  t={note['onset']:.3f}s midi={note['pitch']}  {verdict:<40} conf={confidence:<11} {evidence}")

print("\n--- Extra notes: Basic Pitch notes not claimed by any reference match or ByteDance corroboration ---")
bp_extra = [n for n in basicpitch if not n["consumed"]]
for note in sorted(bp_extra, key=lambda n: n["onset"]):
    confidence, verdict, evidence = classify_extra(note, bytedance, "Basic Pitch", "ByteDance")
    print(f"  t={note['onset']:.3f}s midi={note['pitch']}  {verdict:<40} conf={confidence:<11} {evidence}")
