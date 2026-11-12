"""
Part 4: region-level verdicts for passages where identity breaks down.

Part 1 (align_sequence.py) keeps SUBSTITUTE_COST deliberately high so
noise/misses don't get cheaply absorbed as fake substitutions. The
consequence (confirmed mathematically, not just empirically -- see README
Step 12 postscript): the DP can NEVER choose "substitute" over "delete +
insert", since the combined cost of the latter (2) is always lower than
the former (3). So a real wrong note always shows up as one deletion
(missing ref note) and one insertion (extra performed note), never as a
direct substitution op.

That's recoverable with a simple 1:1 pairing when the mistake is exactly
one wrong note. It is NOT recoverable that way when the performer plays a
different NUMBER of notes in a passage (e.g. four sixteenths become six,
with distorted pitches) -- there is no single "this note became that
note" correspondence to find, and forcing one onto the data would invent
precision we don't have.

Instead of pairing individual missing/extra notes, this module groups
ADJACENT disputed ops (del/ins) into regions -- maximal runs of del/ins,
broken wherever a match or transpose confirms the alignment is back on
track -- and gives each region ONE shape-based verdict:
  del=1, ins=0        -> isolated missing note (not "disputed" -- handled
                          as a plain miss elsewhere, unchanged from before)
  del=0, ins=1        -> isolated extra note (ditto, handled elsewhere)
  del=1, ins=1        -> wrong note (clean one-for-one substitution)
  del=ins, del>1       -> wrong notes (passage): same note count, wrong pitches
  ins>del              -> notes added / rhythm subdivided (e.g. 4 -> 6 notes)
  del>ins              -> notes dropped from a passage
"""


def build_regions(events):
    """Groups consecutive del/ins ops into maximal runs. A match or
    transpose confirms identity is back on track and closes the current
    region. Returns a list of {"dels": [ref_idx, ...], "ins": [perf_idx, ...]}
    in the order the ops were encountered (i.e. roughly chronological)."""
    regions = []
    current = None
    for op, ri, pi in events:
        if op in ("del", "ins"):
            if current is None:
                current = {"dels": [], "ins": []}
            (current["dels"] if op == "del" else current["ins"]).append(ri if op == "del" else pi)
        elif current is not None:
            regions.append(current)
            current = None
    if current is not None:
        regions.append(current)
    return regions


def classify_region(region):
    """Returns (label, is_trivial). Trivial regions (one isolated miss or
    one isolated extra note) are left to the existing per-note handling --
    only non-trivial regions get the macro-level "disputed passage" report."""
    d, i = len(region["dels"]), len(region["ins"])
    if d == 1 and i == 0:
        return "isolated missing note", True
    if d == 0 and i == 1:
        return "isolated extra note", True
    if d == 1 and i == 1:
        return "wrong note", False
    if d == i:  # and d > 1, since d == i == 0 can't happen (region requires >=1 op)
        return "wrong notes (passage)", False
    if i > d:
        return "notes added / rhythm subdivided", False
    return "notes dropped from a passage", False


def region_time_range(region, ref_events, perf_notes):
    times = [ref_events[ri]["onset"] for ri in region["dels"]] + \
            [perf_notes[pi]["onset"] for pi in region["ins"]]
    return min(times), max(times)


def _ranges_overlap(a, b, tolerance_sec):
    return a[0] <= b[1] + tolerance_sec and b[0] <= a[1] + tolerance_sec


def corroborating_regions(region, other_regions, ref_events, perf_notes, other_perf_notes, tolerance_sec=0.3):
    """Other model's regions that line up with this one -- either they
    share deleted reference indices (both models independently lost track
    of the same reference notes, the strongest signal), or, when this
    region has no deletions to share, their time ranges overlap."""
    my_dels = set(region["dels"])
    if my_dels:
        shared = [r for r in other_regions if my_dels & set(r["dels"])]
        if shared:
            return shared
    my_range = region_time_range(region, ref_events, perf_notes)
    return [
        r for r in other_regions
        if _ranges_overlap(my_range, region_time_range(r, ref_events, other_perf_notes), tolerance_sec)
    ]
