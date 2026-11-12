"""
Part 1 of the sequence-first rework: establish note IDENTITY from pitch
ORDER alone, with no absolute timing involved at all. This is deliberately
separate from, and comes before, any timing/tempo analysis -- the goal here
is only to prove we can recover the correct reference<->performance note
correspondence even when the recording's tempo is grossly unstable, which
the time-window approach in score.py cannot do (see README Step 8).

Alignment is an edit-distance DP (Optimal String Alignment variant of
Damerau-Levenshtein: insert / delete / substitute / adjacent-transpose),
over PITCH SEQUENCES ONLY (no times). Costs are chosen so the aligner
prefers "insert + delete" (a separate extra note and a separate missing
note) over "substitute" (claiming one pitch turned into another) --
substitution should be reserved for an actual wrong note, not absorbing
noise/misses cheaply. Adjacent transposition is cheap because the two
hands landing a near-simultaneous chord in slightly different order is
expected and NOT a mistake (seen directly in the unstable clip's data).

Usage: python align_sequence.py <reference.json> <transcribed-notes.json> [label]
"""
import json
import sys

DELETE_COST = 1      # reference note missing from the performance
INSERT_COST = 1       # extra/noise note in the performance, not in the reference
SUBSTITUTE_COST = 3    # claims pitch A became pitch B -- expensive on purpose
TRANSPOSE_COST = 1     # adjacent two-note swap (near-simultaneous chord notes, order ambiguous)


def load_ref_notes(path):
    """Returns [(pitch, onset_sec), ...] in the same order used for alignment."""
    with open(path) as f:
        reference = json.load(f)
    out = []
    for n in reference["notes"]:
        for p in sorted(n["midi_pitches"]):
            out.append((p, n["onset_sec"]))
    return out


def load_perf_notes(path):
    """Returns [(pitch, onset_sec), ...] sorted by onset."""
    with open(path) as f:
        notes = json.load(f)
    notes = sorted(notes, key=lambda n: n["onset"])
    return [(n["pitch"], n["onset"]) for n in notes]


def load_ref_pitches(path):
    return [p for p, _ in load_ref_notes(path)]


def load_perf_pitches(path):
    return [p for p, _ in load_perf_notes(path)]


def align(ref, perf):
    n, m = len(ref), len(perf)
    # d[i][j] = min edit cost aligning ref[:i] with perf[:j]
    d = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i * DELETE_COST
        op[i][0] = "del"
    for j in range(1, m + 1):
        d[0][j] = j * INSERT_COST
        op[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_cost = 0 if ref[i - 1] == perf[j - 1] else SUBSTITUTE_COST
            best = d[i - 1][j - 1] + match_cost
            best_op = "match" if match_cost == 0 else "sub"

            if d[i - 1][j] + DELETE_COST < best:
                best = d[i - 1][j] + DELETE_COST
                best_op = "del"
            if d[i][j - 1] + INSERT_COST < best:
                best = d[i][j - 1] + INSERT_COST
                best_op = "ins"
            if (i > 1 and j > 1 and ref[i - 1] == perf[j - 2] and ref[i - 2] == perf[j - 1]):
                trans = d[i - 2][j - 2] + TRANSPOSE_COST
                if trans < best:
                    best = trans
                    best_op = "trans"

            d[i][j] = best
            op[i][j] = best_op

    # Backtrack
    events = []
    i, j = n, m
    while i > 0 or j > 0:
        o = op[i][j]
        if o == "match" or o == "sub":
            events.append((o, i - 1, j - 1))
            i, j = i - 1, j - 1
        elif o == "trans":
            # ref[i-1] == perf[j-2] and ref[i-2] == perf[j-1] (that's the
            # condition that made this a transposition) -- the pairing is
            # crossed, not same-index.
            events.append(("trans", i - 1, j - 2))
            events.append(("trans", i - 2, j - 1))
            i, j = i - 2, j - 2
        elif o == "del":
            events.append(("del", i - 1, None))
            i -= 1
        elif o == "ins":
            events.append(("ins", None, j - 1))
            j -= 1
    events.reverse()
    return events, d[n][m]


def report(label, ref_path, perf_path):
    ref = load_ref_pitches(ref_path)
    perf = load_perf_pitches(perf_path)
    events, cost = align(ref, perf)

    counts = {"match": 0, "sub": 0, "del": 0, "ins": 0, "trans": 0}
    for o, _, _ in events:
        counts[o] += 1

    print(f"=== {label} ===")
    print(f"reference notes: {len(ref)}, performed notes (after duration-filter+merge): {len(perf)}")
    print(f"total edit cost: {cost}")
    print(f"match={counts['match']} transposed={counts['trans']} substituted(!!)={counts['sub']} "
          f"deleted(missing)={counts['del']} inserted(extra)={counts['ins']}")

    if counts["sub"]:
        print("  SUBSTITUTIONS (claimed wrong note -- should be zero for these ground-truth clips):")
        for o, ri, pi in events:
            if o == "sub":
                print(f"    ref[{ri}]={ref[ri]} -> perf[{pi}]={perf[pi]}")

    print("  deleted (reference note with no corresponding performed note):")
    for o, ri, pi in events:
        if o == "del":
            print(f"    ref[{ri}]={ref[ri]}")

    print("  inserted (performed note with no corresponding reference note):")
    for o, ri, pi in events:
        if o == "ins":
            print(f"    perf[{pi}]={perf[pi]}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    report(sys.argv[3] if len(sys.argv) > 3 else "clip", sys.argv[1], sys.argv[2])
