"""
Part 5: beat-level note-accuracy and rhythm-accuracy scores.

The per-reference-note and per-region detail evaluate.py already produces
is the right level for diagnosis, but not for a player glancing at a
number across attempts. This reduces that detail to two comparable
percentages (note accuracy, rhythm accuracy) plus a weighted overall
score, using a beat (not a note) as the unit of credit -- a beat with
four sixteenths is worth exactly as much as a beat with one quarter note,
per the project's framing: steer the player, don't reward them for
picking a busy passage.

Scoring rules (agreed with the user):
  - A reference note that is matched (directly, or recovered via Basic
    Pitch -- that's a detection-confidence distinction, not a performance
    one) counts as full credit for its share of its beat.
  - A reference note that is missing or wrong, but NOT part of a
    disputed region, counts as zero credit for its share only -- one bad
    note in a four-note beat costs 25%, not the whole beat.
  - A beat touched by a disputed region (Part 4) is worth ZERO entirely,
    not a fraction -- the region exists precisely because we can't say
    how many of its notes were actually wrong, so partial credit would
    overstate our confidence.
  - A beat touched by an isolated extra note (not part of any region,
    but still an unexplained/corroborated extra -- not suppressed noise)
    is ALSO zeroed entirely, same as a disputed region: there's no fixed
    "how many extra notes were expected" to normalize a fraction against,
    and the player did add something the score doesn't have.
  - Rest beats (no reference note at all) are not scored UNLESS a
    disputed region or an extra note lands in them -- then they are
    zeroed like any other touched beat. A clean rest stays unscored.
  - A reference note reclassified "matched (octave-uncertain)" (see
    evaluate.py's octave-recheck pass, right after identity resolution)
    counts as OCTAVE_UNCERTAIN_CREDIT (0.5), not full and not zero --
    we have evidence a note was played roughly there, roughly in the
    right pitch class, but can't tell which octave, so neither extreme
    is honest.

A region's or an extra note's beat is known directly when it has a
deleted reference note to anchor to (region["dels"]). When it doesn't
(a pure-insertion region, or a lone extra note with no deletion nearby),
there is no reference-time anchor of its own -- its only timestamp is in
PERFORMANCE time, which can't be compared to the reference's beat grid
directly once tempo is unstable. That time is estimated by interpolating
between the two nearest CONFIRMED (matched/recovered) notes bracketing
it, using the same local-tempo idea Part 2's IOI-ratio measurement
already relies on.

Rhythm accuracy reuses Part 2's per-step timing tiers directly: tight
counts full credit, noticeable half, excessive zero.
"""

TIMING_TIER_WEIGHTS = {"tight": 1.0, "noticeable": 0.5, "excessive": 0.0}

# A note's full verdict string -> credit for its share of its beat. Any
# verdict not listed (missing, wrong note, wrong note (Basic Pitch), ...)
# defaults to 0. "matched (octave-uncertain)" is deliberately NOT full
# credit -- we don't actually know the octave was right -- but also not
# zero, since we don't know it was wrong either; see
# OCTAVE_CONFUSION_SEMITONES below for how that verdict gets assigned.
OCTAVE_UNCERTAIN_CREDIT = 0.5
VERDICT_CREDIT = {
    "matched": 1.0,
    "matched (recovered via Basic Pitch)": 1.0,
    "matched (octave-uncertain)": OCTAVE_UNCERTAIN_CREDIT,
}

# Only ByteDance's own extra-note classification affects scoring -- it's
# the primary identity source throughout this evaluator. A Basic-Pitch-
# only "unexplained extra" with no ByteDance corroboration is one
# secondary model's unconfirmed noise; scoring on it would make the score
# noisier, not more accurate, so it stays a flagged item in the report
# without zeroing anything.
SCORED_EXTRA_VERDICTS = ("unexplained extra note (flagged, not suppressed)", "extra note (seen by both models)")


def build_beat_cells(reference):
    """Beat cells from the reference's own beat grid (includes rest-only
    beats), each with its reference-time interval."""
    return [
        {"measure": b["measure"], "beat": b["beat"],
         "start": b["onset_sec"], "end": b["onset_sec"] + b["duration_sec"]}
        for b in reference["beats"]
    ]


BOUNDARY_EPS_SEC = 0.05  # tolerate float/duration edge effects at the very start/end of the evaluated range


def beat_for_ref_time(cells, t):
    """Returns the (measure, beat) cell containing reference-time t, or
    None if t falls clearly outside the evaluated range entirely (e.g. a
    performer continuing past the excerpt's last measure). Silently
    clamping an out-of-range estimate onto the nearest boundary beat would
    misattribute content that isn't even part of the scored section to
    that beat -- confirmed in practice: a performer's continuation past
    the reference's last beat interpolated to times ~6.8s against a
    6.75s-long reference, and got folded into the last beat as if it were
    a mistake inside the excerpt."""
    for c in cells:
        if c["start"] <= t < c["end"]:
            return (c["measure"], c["beat"])
    if cells[0]["start"] - BOUNDARY_EPS_SEC <= t < cells[0]["start"]:
        return (cells[0]["measure"], cells[0]["beat"])
    if cells[-1]["end"] <= t <= cells[-1]["end"] + BOUNDARY_EPS_SEC:
        return (cells[-1]["measure"], cells[-1]["beat"])
    return None


def estimate_ref_time(perf_time, confirmed_pairs):
    """Interpolates a reference-time estimate for a PERFORMANCE timestamp
    that has no identity-confirmed note of its own, by bracketing it
    between the two nearest confirmed (ref_time, perf_time) pairs and
    assuming locally-linear tempo between them -- the same assumption
    Part 2's IOI-ratio measurement makes, just solved for the opposite
    unknown. Returns None if there aren't at least two confirmed pairs to
    bracket with."""
    if len(confirmed_pairs) < 2:
        return None
    if perf_time <= confirmed_pairs[0][1]:
        (r0, p0), (r1, p1) = confirmed_pairs[0], confirmed_pairs[1]
    elif perf_time >= confirmed_pairs[-1][1]:
        (r0, p0), (r1, p1) = confirmed_pairs[-2], confirmed_pairs[-1]
    else:
        (r0, p0), (r1, p1) = confirmed_pairs[0], confirmed_pairs[1]
        for i in range(len(confirmed_pairs) - 1):
            if confirmed_pairs[i][1] <= perf_time <= confirmed_pairs[i + 1][1]:
                (r0, p0), (r1, p1) = confirmed_pairs[i], confirmed_pairs[i + 1]
                break
    if p1 == p0:
        return r0
    return r0 + (perf_time - p0) * (r1 - r0) / (p1 - p0)


def estimate_perf_time(ref_time, confirmed_pairs):
    """The reverse of estimate_ref_time: estimates a PERFORMANCE timestamp
    for a reference time that has no confirmed note of its own (used to
    know roughly where in the recording to look for an octave-confusable
    candidate note for a "missing" reference note). Same bracketing logic,
    with the pair roles swapped."""
    return estimate_ref_time(ref_time, [(p, r) for r, p in confirmed_pairs])


def compute_note_score(reference, ref_events, results, reported_regions, bd_notes, confirmed_pairs, extra_results):
    cells = build_beat_cells(reference)
    beat_note_counts = {}  # (measure, beat) -> [good, total], from clean (non-region) reference notes
    zeroed_beats = set()

    for r in results:
        if r["verdict"].startswith("in disputed region"):
            continue  # zeroed below via the region's own span, not counted proportionally
        # NOTE: this must be the coarse beat CELL (from the grid), not the
        # note's own fractional "beat" label (e.g. 1.5 for an eighth after
        # beat 1) -- using the fractional label directly would bucket each
        # distinct sub-beat onset as its own "beat", defeating the entire
        # point of scoring by beat rather than by note.
        key = beat_for_ref_time(cells, r["ref_onset"])
        credit = VERDICT_CREDIT.get(r["verdict"], 0.0)
        bucket = beat_note_counts.setdefault(key, [0.0, 0])
        bucket[0] += credit
        bucket[1] += 1

    for region, _label in reported_regions:
        if region["dels"]:
            ref_times = [ref_events[ri]["onset"] for ri in region["dels"]]
            affected = {beat_for_ref_time(cells, t) for t in ref_times}
        else:
            perf_times = [bd_notes[pi]["onset"] for pi in region["ins"]]
            estimated = [estimate_ref_time(t, confirmed_pairs) for t in perf_times]
            affected = {beat_for_ref_time(cells, t) for t in estimated if t is not None}
        zeroed_beats |= {b for b in affected if b is not None}

    for e in extra_results:
        if e["model"] != "ByteDance" or e["verdict"] not in SCORED_EXTRA_VERDICTS:
            continue
        estimated = estimate_ref_time(e["onset"], confirmed_pairs)
        if estimated is not None:
            beat = beat_for_ref_time(cells, estimated)
            if beat is not None:
                zeroed_beats.add(beat)

    scored_beats = {}
    for key, (good, total) in beat_note_counts.items():
        scored_beats[key] = 0.0 if key in zeroed_beats else good / total
    for key in zeroed_beats:
        scored_beats.setdefault(key, 0.0)  # a rest beat touched only by a region/extra note

    if not scored_beats:
        return None, scored_beats
    return 100.0 * sum(scored_beats.values()) / len(scored_beats), scored_beats


def compute_rhythm_score(results):
    tiers = [r["timing_tier"] for r in results if r["timing_tier"]]
    if not tiers:
        return None
    return 100.0 * sum(TIMING_TIER_WEIGHTS[t] for t in tiers) / len(tiers)


def compute_scores(reference, ref_events, results, reported_regions, bd_notes, confirmed_pairs, extra_results,
                    note_weight=0.7, rhythm_weight=0.3):
    note_score, scored_beats = compute_note_score(
        reference, ref_events, results, reported_regions, bd_notes, confirmed_pairs, extra_results)
    rhythm_score = compute_rhythm_score(results)
    parts, weights = [], []
    if note_score is not None:
        parts.append(note_score * note_weight)
        weights.append(note_weight)
    if rhythm_score is not None:
        parts.append(rhythm_score * rhythm_weight)
        weights.append(rhythm_weight)
    overall = sum(parts) / sum(weights) if weights else None
    return note_score, rhythm_score, overall, scored_beats
