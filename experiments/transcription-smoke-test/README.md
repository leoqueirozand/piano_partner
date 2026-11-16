# Neural audio-to-MIDI transcription — exploration notes

Exploring whether server-side neural transcription (inspired by
Songscription.ai) can replace the browser-side onset-detection +
chroma-DTW pipeline in `js/`, now that server-side processing is
acceptable (uploaded audio handled in-memory only, never persisted —
"volatile" per the existing privacy posture, extended to the server).

## Why

The current pipeline infers notes indirectly: SuperFlux onset
detection + a predominant-pitch tracker + chroma vectors, then DTW
against a reference. It's fragile (see the SuperFlux merge-threshold,
double-peak, and DTW skip-penalty bugs fixed earlier in this project)
and fundamentally monophonic — it approximates "the pitch active at
each onset," not real polyphonic note content.

A neural transcription model outputs actual discrete notes (pitch,
onset, offset, velocity) directly from audio, including polyphony.
That would let the alignment step match real MIDI pitches against the
reference instead of fuzzy chroma, which should be both more accurate
and far less finicky to tune.

## Candidates considered

| | ByteDance `piano_transcription_inference` | Spotify Basic Pitch | MuScriptor |
|---|---|---|---|
| Piano-specialized | Yes (trained on MAESTRO) | No (general-purpose) | No (general multi-instrument) |
| Reported accuracy | Onset F1 96.72% on MAESTRO | Not piano-specific; "marginally below specialized SOTA" | Onset F1 ~52–53% (different/harder benchmark, not directly comparable) |
| Output | onset, offset, pitch, velocity, **sustain pedal** | onset, offset, pitch, pitch-bend | onset, offset, pitch, instrument tag |
| License | Code: Apache 2.0 (confirmed from repo README). Checkpoint: downloaded from Zenodo, no separate license statement found on the checkpoint itself — **not yet fully verified for commercial use**, since MAESTRO (the training data) is CC BY-NC-SA 4.0. | Apache 2.0 (clean, permissive) | Code: MIT. Weights: **CC BY-NC 4.0 — explicitly non-commercial**, plus an indemnification clause. |

Decision: start with **ByteDance's model** as the accuracy benchmark,
since it's piano-specialized and highest-reported-accuracy. If
precision isn't good enough, or the license question becomes a
blocker, fall back to **Basic Pitch** (cleanly licensed, likely a step
down in piano-specific accuracy) — having ByteDance's numbers first
gives us something concrete to compare Basic Pitch against.

**Open item:** verify whether the ByteDance checkpoint can be used
commercially before shipping anything built on it — the code license
is clear (Apache 2.0) but the weights' provenance (trained on a
non-commercial dataset) hasn't been resolved from official sources.

## Step 1 — smoke test (done)

Ran `piano_transcription_inference` directly (no app integration) on
a 15s clip of `recordings/minueto-estavel.m4a`, CPU-only inference.

Setup notes:
- Needs Python ≤3.11-ish for current torch wheels (3.14, the system
  default here, is too new). Used a `python3.11 -m venv`.
- The library's checkpoint auto-downloader shells out to `wget`,
  which isn't installed on this Mac — fetched the ~165MB checkpoint
  from the Zenodo URL in its source directly with `curl` instead.
  One-time step; cached at `~/piano_transcription_inference_data/`.
- CPU inference: 7.2s for 15s of audio — faster than real-time, no
  GPU needed even for this rough test.

Result: **72 note events**, written to `test-clip-unfiltered.mid`.
Listened back — the full melodic performance came through correctly.
But it also added a cluster of **17 spurious low notes (MIDI 36/38,
C2/D2)** that are not present in the original recording.

## Step 2 — duration-based filter (done)

All 17 spurious notes had implausibly short durations (5–26ms, vs.
real piano notes which ring out far longer). Filtering out any note
shorter than 50ms (`MIN_NOTE_DURATION_SEC` in `smoke_test.py`) removed
all 17 artifacts and **zero legitimate notes** on this clip — kept
55/72, dropped exactly the 17 that were already flagged as suspicious
by ear.

**Threshold revised to 30ms** after testing a second clip
(`minueto-estavel-com-erro.m4a`, see below) — 50ms turned out to also
catch a real note (a 48ms G4). Across both test clips, every confirmed
artifact tops out at 15ms, and the only real note affected was the
48ms G4, so 30ms cleanly separates the two without reintroducing any
previously-confirmed artifact. One edge case remains unresolved by
duration alone: an 8ms MIDI 72 blip on the second clip sits in the
same duration range as genuine artifacts, but lines up with the same
octave-resonance pattern described below (12 semitones below a note
sustaining at that exact moment) — likely resonance, not a missed
note, but indistinguishable from one by duration alone.

Output: `test-clip-filtered.mid`, for listening comparison against
`test-clip-unfiltered.mid`.

This is a cheap, deterministic first-pass denoise. It's a blunt
heuristic threshold, though — it works here because the artifact
happened to be both low-pitched and very short, but nothing guarantees
every false positive will look like that.

**3 low notes still survived the filter** (MIDI 36/38/40, 53–57ms
each — just over the 50ms cutoff). Listening back, these aren't
noise: they're the piano's own sympathetic resonance/overtones from
real struck notes. The data confirms it — e.g. the MIDI 38 note at
t=3.607s lands exactly as MIDI 57 (A3) starts ringing, 19 semitones
up (an octave + a fifth, the 3rd harmonic); the MIDI 36 note at
t=6.603s falls inside the sustain of MIDI 60 (C4), exactly 2 octaves
up. These have ordinary-looking duration and velocity — acoustically
indistinguishable at the signal level from a genuinely struck quiet
note. This sets a hard ceiling on what duration/pitch/velocity
heuristics (A/B/C) can ever catch: real harmonic content isn't a
recording defect, so no amount of signal-level tuning will fully
separate it from genuine notes. Only knowing what's *expected* (the
reference score) can make that distinction — which is the core
argument for building the reference-aware filter (D) next, rather
than continuing to chase signal-level thresholds.

## Follow-up: reference-aware filtering (not yet implemented)

Because this app always has a known reference score to align against
(unlike generic transcription use cases), a more robust long-term
filter is: **after DTW-aligning transcribed notes against the
reference, discard any transcribed note that doesn't land near any
reference note in both time and pitch** — rather than relying on a
generic duration/pitch heuristic that has to be guessed at per
recording.

This needs to be built carefully, not bolted on casually: the entire
point of this tool is to **accurately assess the performance**, so the
filter must not end up silently discarding real mistakes (wrong notes,
extra notes, rhythmic errors) just because they don't match the
reference — that would defeat the purpose of the evaluator. The
duration-based filter (Step 2) is safe because it targets a signal
(implausible note duration) that's unrelated to performance
correctness. A reference-aware filter has to be designed to
distinguish "likely transcription noise" from "a real performance
mistake we need to report," which is a much subtler problem — e.g. by
reasoning about amplitude/confidence, consistency with surrounding
context, or how far off-reference the note is, rather than discarding
anything merely unmatched.

Deferred until the duration filter's limits become clear (e.g. on
louder/noisier recordings, or ones with legitimate very-short notes).

## Step 3 — second clip, with deliberate wrong notes (done)

Ran the same pipeline on `recordings/minueto-estavel-com-erro.m4a`
(16.5s, full clip) using `transcribe.py` (generalized version of
`smoke_test.py` — takes `<audio path> <output prefix>` as args).

Result: 77 raw notes, 51 after the (then-50ms) duration filter. The
melody came through correctly, including a clean octave-down shift on
the final phrase (MIDI 60/59/57 instead of the expected 69/71/69
register) — confirmed by ear as a real performance mistake, correctly
preserved by the transcription rather than smoothed over. This is a
good sign for the tool's actual purpose: it's not just transcribing
"a" performance, it's faithfully capturing *this* performance,
mistakes included.

This clip is also what surfaced the 48ms real G4 note being caught by
the 50ms filter, leading to the threshold revision described above.

Files: `erro-clip.wav`, `erro-clip-unfiltered.mid`,
`erro-clip-filtered.mid`.

## Step 4 — octave errors, and comparing against Basic Pitch (done)

Separately from the noise/resonance artifacts above, ByteDance's
model also makes **octave errors on real, confidently-detected notes**
— reporting the right pitch class but the wrong octave. Confirmed
ground-truth case: a B3 in the real performance (`erro-clip.wav`,
~t=5.48s) is transcribed as B4 (MIDI 71 instead of 59), with a long
duration (1.5s) and ordinary velocity — not a borderline/quiet note,
the model is just confidently wrong.

**Tried: lowshelf bass boost before transcription** (+9dB below
150Hz, via `ffmpeg`'s `bass` filter as a stand-in for a browser-side
`BiquadFilterNode`, which is what an actual implementation would use).
Hypothesis was a weak/masked fundamental (plausible if a recording
device's mic rolls off bass). Result: **no effect on the confirmed
B3→B4 case** (still reported as MIDI 71, if anything with a slightly
longer duration), and a net loss of real notes elsewhere — 6 real
notes vanished on `test-clip.wav` and 3 on `erro-clip.wav` after
boosting, with zero new/corrected notes gained on either clip.
**Conclusion: this preprocessing approach is not viable** — the
error isn't caused by a weak fundamental being masked, so boosting it
has nothing to fix, and it actively degrades other, already-correct
transcriptions. Not pursuing further (no further gain/frequency
sweeps planned).

**Tried: Spotify Basic Pitch on the same clip, as a second opinion.**
On the confirmed B3 ground-truth note, Basic Pitch gets it right
(MIDI 59, amplitude 0.587 — a confident detection), where ByteDance
is wrong. But across the rest of the clip, Basic Pitch has its own
issues: it makes the *same kind* of octave mistake on a different
note (the opening D5, reported an octave low / collided with an
accompaniment note), and adds extra spurious notes including one very
high outlier (MIDI 91, ~370ms) not present in the recording. Overall
verdict (by ear): **ByteDance is still the more precise transcription
of the two**, despite missing this specific note.

This means neither model is unconditionally trustworthy on octave
placement, but they don't make *the same* mistakes — which is useful.

**Decision (documented, not implemented yet): Option 3 — ByteDance as
primary, Basic Pitch as a targeted arbitrator.** Keep ByteDance as the
main transcription model (better overall precision), but run Basic
Pitch as a second opinion specifically to arbitrate suspected octave
disagreements (e.g. flagged by the reference-aware filter below, or
by comparing the two models' outputs directly) rather than running it
unconditionally on everything. Rejected alternatives: switching to
Basic Pitch as primary (worse overall by ear, despite winning this one
case); always running both models in full (doubles inference cost for
marginal benefit, since most of the clip doesn't need arbitration).

Deferred: this needs the alignment/scoring pipeline in place first to
know *when* arbitration is actually needed, so it's being deferred
alongside the reference-aware filter (D/E) work above, not built in
isolation right now. Priority right now is testing the core
transcription step against more complex/varied recordings.

Files: `erro-clip-basicpitch.mid` (Basic Pitch's raw output on the
same clip, for listening comparison against the two ByteDance MIDIs).

## Step 5 — first real recording, and a same-pitch onset-merge fix (done)

Added WAV persistence to the in-app recorder (`js/recorder.js` +
`index.html`, dev/testing-only — the shipped recording flow stays
volatile) so real takes could be saved and reused across tests instead
of re-recording every time. First real test: `bach_english_suite_take_1.wav`
(11.2s, first 3 bars of J.S. Bach's English Suite No. 2 Prelude,
BWV 807 — genuinely two-voice polyphonic, unlike the single-line
minuet clips above).

Result: 62 raw notes, 53 after the duration filter. By ear: pretty
faithful overall, with two issues.

**Issue 1 — a real note transcribed twice.** The opening MIDI 76 (E5)
note (t=1.057s, 133ms) was immediately followed by another MIDI 76
onset at t=1.196s (54ms) — a ~6ms gap between the first note's offset
and the second's onset, i.e. essentially one continuous sound split
into two note events by the onset detector. This is a different
failure mode than Steps 1–2 (noise/resonance): it's the model
over-segmenting a single sustained note, not adding a spurious one.

**Fixed with a same-pitch onset-merge step**, added to `transcribe.py`
right after the duration filter: collapse adjacent notes of the same
pitch into one whenever the gap between them is ≤50ms
(`MAX_MERGE_GAP_SEC`). Chosen by checking real repeated-pitch notes
across all three existing clips first — every genuine same-pitch
repeat (`test-clip`, `erro-clip`) had a gap ≥300ms, so 50ms separates
the split-onset artifact (6ms) from real repeated keystrokes with
large margin on both sides. Verified no regression: re-running all
three clips through the merge step changed only `bach-clip` (53→52
notes, exactly the one expected merge); `test-clip` and `erro-clip`
were untouched (0 merges on either).

**Issue 2 — a cluster of recurring low notes, deferred.** A fixed
pitch around D2 (MIDI 37/38/39) repeats several times (t≈2.15–6.84s,
50–64ms each, velocity 41–69), distinct from a separate group of
longer, higher-velocity low notes starting at t≈7.4s (MIDI 52/48/56/59,
75–265ms, velocity 51–87) that look like the piece's real bass voice
entering. Unlike the Step 2 harmonic-resonance case, the D2 cluster
doesn't track whatever's sounding above it, which points more toward
noise/artifact than genuine resonance — but distinguishing it safely
from the real bass notes by signal-level heuristics alone risks the
same problem as before (cutting real bass content). Not filtering
this yet; revisit once the reference-aware filter exists, or with
more data from further recordings.

Files: `bach-clip-unfiltered.mid`, `bach-clip-filtered.mid` (now
includes the onset-merge step).

## Step 6 — quiet/masked note investigation (done — dead end at the signal level)

On the same Bach clip, bar 2's last sixteenth note is a D in both
hands (an octave apart). The lower one (D4, MIDI 62) is missing from
the transcription, filtered and unfiltered. Listening to the raw WAV,
the note is really there, just played a little softer than the rest
of the performance, struck at the same instant as the louder D5 above
it.

**Investigated via raw per-frame probabilities** (`output_dict`'s
`reg_onset_output`, shape `(frames, 88)`, column = `midi_note - 21`,
`frames_per_second=100`): the onset probability at D4's expected frame
is ~0.001–0.007 — not borderline-low, essentially zero. This is a
textbook **simultaneous-onset masking** case: the model's attention is
dominated by the louder concurrent D5 onset, and it isn't picking up
the quieter note's attack at all, at any threshold.

**Tried: lowering `onset_threshold` globally** (0.3 → 0.1) across all
three test clips. No recovery of the D4 anywhere in that range, but
raw note counts ballooned 35–50% on every clip (bach 62→94, test-clip
72→100, erro-clip 77→106) — i.e. lowering the global threshold doesn't
reach masked notes, it just lets through a lot of new noise elsewhere.
Same shape of dead end as the Step 4 lowshelf-boost experiment.
**Conclusion: not a threshold problem, not fixable by preprocessing.**

**Tried (this session): checking the raw `frame_output` curve too**,
not just `reg_onset_output` — hypothesis being that masking might
specifically corrupt the attack-transient regression (`reg_onset_output`)
while the separate "is this key down" sustain signal (`frame_output`,
gated at `frame_threshold=0.1` in the real post-processor) survives.
Built `raw_probe.py` to pull both raw arrays for every reference note
not directly matched by ByteDance's thresholded output, in a ±150ms
window around the expected time. Result on the D4 case: `frame_output`
max in that window is 0.023 — also essentially zero, well under
`frame_threshold`. **No signal in either curve.** This closes the
question: ByteDance's model has a genuine blind spot here at every
stage of its own pipeline, not just at the final thresholding step —
there is nothing left to recover from ByteDance's output alone.

**Resolved via cross-model recovery instead**: Basic Pitch, run on the
same clip, independently reports this exact note (MIDI 62, t=6.633s,
amplitude 0.486 — a confident detection, not a marginal one). This is
the concrete case that motivated building the scoring prototype in
Step 7 around *always* running both models and letting Basic Pitch
recover what ByteDance structurally cannot see, rather than trying to
rescue it from ByteDance alone.

Files: `raw_probe.py` (the per-frame probe tool — reusable for
checking any other future "missing" case before assuming cross-model
recovery is the only option; on this clip it confirmed there was
nothing to recover).

## Step 7 — reference-aware, confidence-tiered scoring prototype (done)

Building on the decision (Step 4) to defer the "Basic Pitch only when
needed" optimization: the priority right now is proving the evaluator
can produce a *useful, trustworthy* assessment at all, before making
it efficient. Working principle adopted this step: **never silently
discard a mismatch.** A filter that only ever gives positive feedback,
even when the performance has real problems, is worse than useless —
every event (missing reference note, extra transcribed note,
pitch/octave mismatch) gets classified into a confidence tier instead
of being silently dropped or silently trusted.

### `make_reference.py`: Voice sub-stream bug (found and fixed)

The Bach prelude notates its two voices as `Voice` sub-streams inside
each `Measure` (common for contrapuntal keyboard music), rather than
one note/chord per part per beat. `music21`'s `m.notesAndRests` does
**not** recurse into `Voice` streams, so the original script silently
returned near-empty note lists for measures that used them (measure 2
in particular) — a correctness bug, not a missing feature, since
nothing errored, it just produced a wrong (too-short) reference.

Fixed via `m.recurse().notesAndRests` + `el.getOffsetInHierarchy(m)`
for correct measure-relative timing. Also changed `--part <int>` to
`--parts <comma-separated or "all">`, since this piece's two hands
need merging into one flat, time-sorted reference timeline (a real
performance produces one note-on per pitch even when both hands land
on the same beat — the reference should mirror that, not chord-combine
across parts). Verified: reference note count for bars 1–3 went from
12 (buggy) to 44 (correct, cross-checked against the actual recording).

### `score.py`: matching reference notes against both models

For each reference note (onset, pitch), look for a matching transcribed
note in ByteDance first, then Basic Pitch, within a time window (using
a rough global linear tempo-scale estimate anchored on first/last note
— a known placeholder for real DTW alignment, not yet replaced):

| Tier | Condition | Confidence |
|---|---|---|
| matched | ByteDance has an exact pitch+time match | high |
| matched (recovered) | only Basic Pitch has it | medium |
| octave mismatch | right pitch class, wrong octave, in either model | medium |
| missing | neither model found it near the expected time | low |

Result on the Bach clip (44 reference notes): 43 matched, 1 recovered
(the Step 6 masked D4), 0 missing, 0 octave mismatches — stable across
all the fixes below.

### Symmetric check: classifying unmatched/"extra" transcribed notes

The other direction of the same problem: a transcribed note not
claimed by any reference match could be real transcription noise
(safe to suppress), harmonic resonance (safe to suppress, same
phenomenon as Step 2), a real extra/wrong note the performer actually
played (must not be suppressed), or genuinely ambiguous (must be
flagged, not guessed away). `classify_extra()` combines several
signals rather than any single threshold:

| Tier | Condition | Confidence |
|---|---|---|
| extra note (seen by both models) | the other model independently reports the same pitch near the same time | medium-high — likely real |
| harmonic resonance (suppressed) | quiet + short duration (<150ms) + sits at a clean interval (12/19/24 semitones) below a concurrently-sounding confirmed note | low |
| noise artifact (suppressed) | quiet + short, no corroboration, no harmonic explanation | low |
| unexplained (flagged, not suppressed) | none of the above fit | medium — reported rather than guessed away |

**Two bugs found and fixed in the underlying onset-merge heuristic**
while building this (same function, used in both `transcribe.py` and
`score.py`):

1. **Pitch-tracking bug.** The original merge only compared a note
   against `merged[-1]` (the previous note in overall time order).
   When applying the merge to Basic Pitch's output, a 3-way split
   E3 note (at the end of bar 3) failed to merge at all, because a
   different-pitch note (D5, the other hand) sat between two of the
   E3 segments in time, breaking the same-pitch adjacency check. Fixed
   by tracking the last note *per pitch* via a `last_by_pitch` dict.

2. **Safety bug the fix exposed.** Once per-pitch tracking was
   correct, it found a *new* ByteDance merge candidate (two B4 notes,
   735ms + 302ms, 15ms gap) that turned out, on checking against the
   reference, to be two genuinely distinct real notes on different
   beats, not a split artifact — merging them would have silently
   erased real timing information, directly against this step's
   working principle. Fixed by adding `MAX_SUBSTANTIAL_NOTE_DUR_SEC`
   (150ms): only merge when at least one of the two segments looks
   like a short trailing fragment, not when both are substantial notes.
   Distinguishing signal: the legitimate E5 merge (Step 5) was
   133ms+54ms (one short side); the bad B4 case was 735ms+302ms (both
   substantial). Verified: main matching results (43/1/0/0 above)
   unchanged before and after both fixes; the bad B4 merge no longer
   happens; the good E5 merge still does.

**User-corrected misdiagnosis along the way:** a batch of "unexplained,
flagged" extra notes was initially (wrongly) hypothesized to be a
reference-boundary artifact (recording running past bar 3). Corrected
directly from listening: those were legitimate end-of-bar-3 notes
(RH sixteenths E4/D5/C5/B4, LH eighths G3/E3). Re-investigated and
found the real explanations were (a) ByteDance's extra E2 being an
under-detected harmonic resonance — caught by widening
`HARMONIC_RESONANCE_MAX_DUR_SEC` (0.08s → 0.15s, resonance tails run
a bit longer than pure noise) and adding `RESONANCE_LOOKAHEAD_SEC`
(a resonance can register slightly before the real note's own onset,
not only during its sustain); and (b) Basic Pitch's own un-merged
E3 split-onset artifact, resolved by applying the onset-merge heuristic
(bugs above) to Basic Pitch's output as well, not just ByteDance's.

Remaining after all fixes: 4 Basic-Pitch-only items still flagged
"unexplained, not suppressed" (t=3.287s midi=49, t=6.865s midi=49,
and the two substantial E3 segments at t=9.084s/9.653s that the
`MAX_SUBSTANTIAL_NOTE_DUR_SEC` safeguard correctly declines to merge).
Left visible at medium confidence rather than force-resolved — matches
the "never silently hide" principle, but these are not yet explained.

Files: `bach-3bars-reference.json` (regenerated, 44 notes),
`bach-clip-basicpitch.mid`/`.json` (Basic Pitch's output on the Bach
clip, generated for this comparison), `score.py` (the scoring
prototype itself — not production code, proving out the matching/
tiering logic on one real recording first).

### Known open items going into the next step

- The global linear tempo-scale estimate in `score.py` is a rough
  placeholder (anchors on first/last note only) — needs real DTW-based
  alignment before this can handle tempo drift within a performance,
  not just an overall scale factor.
- The 4 remaining "unexplained, flagged" Basic-Pitch-only extras above
  aren't yet explained.
- `score.py` currently reasons entirely over final, already-thresholded
  note lists from each model. The Step 6 raw-probe technique (reading
  `reg_onset_output`/`frame_output` directly in a reference-anchored
  window) is a different, not-yet-integrated axis: useful for marginal
  near-threshold cases, though it confirmed there was nothing recoverable
  for the one case tested (the masked D4, Step 6) since that one has
  zero signal in *either* raw curve, not just a too-high threshold.

## Step 8 — tempo normalization, attempt 1: global robust linear fit (superseded, see Step 10-11)

Players shouldn't need to declare a tempo up front, and the evaluator's
job is to detect whether a section was played *correctly* — not to judge
musicality, phrasing, rubato, or accelerando. First attempt: replace the
Step 7 first/last-note anchor with a **robust linear tempo fit**
(`perf_onset ≈ intercept + slope·ref_onset`) over all ByteDance-exact
matches, using iterative median-absolute-deviation outlier trimming so a
few wrong matches can't drag the estimate off course — then a second
matching pass against the refined estimate with a tighter window, and a
per-note timing-deviation tier (tight/noticeable/excessive) expressed as
a **fraction of the performer's own beat**, not absolute milliseconds
(the same timing error matters more at a fast tempo than a slow one).

Validated on `bach_english_suite_take_1.wav` (the consistent-tempo take):
all 44 reference notes came back "tight" (max deviation 99ms, ~9.5% of
the fitted ~1037ms beat), with the fit correctly identifying the
performer's own effective tempo (57.8 bpm) rather than the score's
engraved 80 bpm. Zero false "excessive" flags on a genuinely clean
performance.

**This approach was only tested against a tempo-consistent take at this
point** — see Step 9 for what happened when it hit a real
tempo-unstable one, which is the harder and more important case.

## Step 9 — testing against a deliberately unstable recording exposes the flaw

Tested the Step 8 approach against `bach_english_suite_unstable.wav`
(same first 3 bars, deliberately recorded with gross tempo/rhythm
mistakes but — per direct confirmation — no wrong notes). Result:
only 8 matched, 1 recovered, 4 octave mismatches, **31 missing**, plus
a flood of "extra note (seen by both models)" flags. Taken at face
value, this would mean the performance was riddled with wrong notes —
directly contradicted by ground truth.

**Diagnosis: compared raw pitch sequences, ignoring all timing.** After
setting aside the known recurring low-pitch noise cluster (MIDI 37–41)
and one spurious outlier, ByteDance's detected pitch sequence was
**almost a perfect match to the reference, in the same order** — the
only discrepancies were a few adjacent two-hand chord notes swapped in
order (whichever hand landed a few ms first) and one genuinely
undetected note. **The notes were correct all along; only the matching
architecture was wrong.**

Root cause: Step 7/8's matcher only knows where to look for a note
because of a **single global linear tempo fit** (one slope + offset for
the whole excerpt). That's fine when the tempo is actually constant, but
by definition cannot represent a performance with *real* local tempo
instability — no single straight line fits a tempo that genuinely
speeds up, stalls, and lurches within the excerpt. Once the projected
"expected time" drifts far enough from the tighter matching window,
correctly-played notes get reported as "missing" and show up elsewhere
as unexplained "extra" notes — the evaluator wasn't wrong that something
was unstable, but it drew the wrong conclusion (looks like wrong notes)
from the right observation (timing is erratic).

**Decision: separate note *identity* from note *timing*.** They
shouldn't share the same fragile time-window mechanism. Rebuilt as two
explicit parts:

## Step 10 — Part 1: pitch-sequence identity matching (done)

`align_sequence.py`: establish the reference↔performance note
correspondence from **pitch order alone, with no absolute timing
involved**. This is an edit-distance DP — the Optimal String Alignment
variant of Damerau-Levenshtein (insert / delete / substitute / adjacent
transpose) — over the two pitch sequences. Costs are chosen deliberately:
substitution (claims pitch A became pitch B) is expensive (3), while
insertion (extra/noise note) and deletion (missing note) are cheap (1
each) and a two-element adjacent transposition is also cheap (1) — this
makes the aligner prefer "separate miss + separate extra" over "wrong
note" when noise is involved, and explicitly tolerates the two-hand
near-simultaneous chord reordering seen directly in the data, without
ever needing to look at a timestamp.

Validated against both recordings, both models — **zero substitutions
in all four runs**, confirming no wrong notes on either clip (matches
ground truth for both):

| Clip / model | matched | transposed | **substituted** | deleted (missing) | inserted (extra) |
|---|---|---|---|---|---|
| stable — ByteDance | 37 | 6 | **0** | 1 | 9 (known noise cluster) |
| stable — Basic Pitch | 38 | 6 | **0** | 0 | 5 |
| unstable — ByteDance | 35 | 8 | **0** | 1 | 8 (mostly known noise cluster) |
| unstable — Basic Pitch | 41 | 2 | **0** | 1 | 5 |

The key result: **the unstable clip's correspondence quality is
essentially as clean as the stable clip's.** Pitch-order identity
recovery genuinely doesn't care how erratic the timing is. The single
deletion on the stable/ByteDance run is `ref[25]`=MIDI 62 — the Step 6
masked-D4 case, which Basic Pitch recovers (0 deletions on that run) —
not a new problem, just confirmation that known cases still fall where
expected.

**Bug found and fixed while building this:** the backtrack for the
transposition case originally paired `(i-1,j-1)` and `(i-2,j-2)` (the
same-index positions) instead of the actual crossed pair implied by the
transposition condition (`ref[i-1]==perf[j-2]` and `ref[i-2]==perf[j-1]`
⟹ pairs are `(i-1,j-2)` and `(i-2,j-1)`). Didn't affect Part 1's counts
(which only cared about *how many* of each op type occurred), but would
have silently produced wrong onset-time correspondences for Part 2
below, which depends on exactly which performed note each reference
note maps to. Fixed before Part 2 was built on top of it.

Files: `align_sequence.py`.

## Step 11 — Part 2: local tempo-ratio deviation measurement (done)

`local_tempo.py`, built directly on Part 1's now-trusted
correspondence: for every pair of **consecutive identity-confirmed**
notes (match or transpose only — a deletion/insertion breaks the chain
and is skipped over, not bridged with a guess), compare the inter-onset
interval (IOI) in the reference to the IOI in the performance. The
ratio `perf_IOI / ref_IOI` is a **local** tempo estimate for just that
one note-to-note transition — deliberately not global, since a global
single slope+offset is exactly what Step 8/9 showed cannot represent
real instability. Each step's expected timing is anchored only on the
*previous confirmed note's actual performed time*, so instability in
one passage cannot corrupt the measurement anywhere else in the excerpt
(unlike the global fit, where one bad region could shift everything
downstream of it).

The player's own overall chosen tempo is estimated as the robust
(median, MAD-trimmed) average of all step ratios — not asked for up
front, and not compared against the score's engraved tempo, consistent
with the project's scope (evaluate consistency with the player's own
tempo, not against any fixed bpm; no modeling of legitimate rubato/
accelerando). Each step's deviation from that baseline is expressed as
a percentage and bucketed: tight (≤12%), noticeable (12–30%), excessive
(>30%) — same tier boundaries as Step 8, now applied locally instead of
globally.

Result, ByteDance, both clips:

| Clip | tight | noticeable | **excessive** |
|---|---|---|---|
| stable | 25 | 5 | **0** |
| unstable | 16 | 6 | **8** |

Zero excessive steps on the stable take (the 5 "noticeable" ones are
borderline 12–23% deviations on fast sixteenth-note transitions —
ordinary human timing variance, not a real problem). The unstable
take's 8 excessive steps aren't scattered randomly — they fall into two
distinct clusters: **ref 3.94s–4.50s** (3 consecutive steps, 40%/75%/
104% off baseline) and **ref 5.625s–6.56s** (5 consecutive steps,
38%/52%/35%/104%/103% off baseline). That clustering is itself a useful
signal: it reads as two specific passages where the performer lost the
beat, not uniform sloppiness throughout — exactly the kind of localized,
actionable feedback ("you lost the beat around bar 2, beat 4") this
evaluator should eventually surface, rather than a flat "rhythm: bad."

Files: `local_tempo.py`.

## Step 12 — Part 3: one unified evaluator pass (done)

`evaluate.py` folds Part 1 (identity), Part 2 (local timing), and the
Step 7 noise/resonance/cross-model extra-note classification into a
single pass over both models' output:

1. **Identity**, per reference note: ByteDance's pitch-sequence
   alignment (Part 1) resolves it first — matched/transposed directly,
   *substituted* (flagged as a real wrong note — this evaluator's core
   job, never silently dropped), or *deleted* (missing from ByteDance).
   A ByteDance miss is then checked against Basic Pitch's own,
   independently-computed alignment; if Basic Pitch matched it, recover
   it at medium confidence, mirroring the old recovered-via-Basic-Pitch
   tier but now identity-based instead of time-window-based.
2. **Timing**, over the resulting *fused* per-reference-note timeline
   (ByteDance primary, Basic Pitch filling identity gaps): Part 2's
   local-IOI-ratio deviation measurement, unchanged in method.
3. **Extra notes**: every performed note not used as a primary or
   recovery match goes through Step 7's classifier (duration, velocity,
   harmonic interval, cross-model corroboration) to separate likely
   noise/resonance (suppress) from a genuinely unexplained extra note
   (flag, don't suppress).

**Bug found and fixed while wiring this together:** Basic Pitch's own
*correct* detection of a note ByteDance had already matched was showing
up in the extra-notes list as "seen by both models — likely a real
extra/wrong note," because only the model actually chosen to resolve
each reference note had its matched note marked consumed; the other
model's own matching detection of that *same* note was left looking
unclaimed. Fixed by adding a pass that marks a model's matched/
substituted note as consumed whenever its OWN alignment independently
resolved that reference index too, regardless of which model ended up
providing the reported identity. This cut the stable clip's spurious
"extra" list from ~40 entries (nearly all just Basic Pitch echoing
notes ByteDance already matched) down to the 14 that are real noise/
resonance/unexplained items.

Result, both clips (identity unaffected by tempo instability, exactly
as Part 1 predicted; timing cleanly separates the two takes):

| Clip | matched | recovered | wrong note | missing | tight | noticeable | **excessive** |
|---|---|---|---|---|---|---|---|
| stable | 43 | 1 | 0 | 0 | 34 | 9 | **0** |
| unstable | 43 | 1 | 0 | 0 | 20 | 10 | **13** |

Identity is identical between the two takes (as it should be — no wrong
notes on either, per ground truth) while timing correctly flags the
unstable take's two known problem passages and nothing on the stable
one. The extra-notes list on both clips is now small and each item
carries a specific, inspectable reason (noise cluster pitch, harmonic
interval + concurrent note, or genuinely unexplained) rather than a bare
"doesn't match" — consistent with the project's working principle that
nothing gets silently dropped or silently trusted.

Files: `evaluate.py`.

### Known open items going into the next step

- Thresholds throughout (duration/velocity cutoffs, harmonic-interval
  list, the 12%/30% timing tiers, the edit-distance costs) are all
  hand-picked from observations on these two clips plus the two earlier
  minuet clips. They need testing against more varied recordings —
  different pieces, tempos, and genuinely wrong-note takes (neither
  existing recording has one) — before treating any of them as settled.
- The extra-notes classifier still runs independently of the
  edit-distance alignment's own insertion-cost reasoning; the two
  overlap conceptually (both are "is this performed note explained by
  the reference") and could eventually be merged into one scoring pass
  rather than two sequential ones.
- No recording with an actual wrong note (a real substitution) has been
  tested yet — the "wrong note" verdict path in `evaluate.py` is
  implemented and reachable, but unverified against ground truth.

## Step 13 — Part 4: region-level verdicts for passages, not just notes (done)

Two new ground-truth recordings closed the last open item above and
immediately exposed a real architectural gap:

- `bach_english_suite_wrong_notes.wav` — many deliberate wrong notes
  concentrated in the back half of the clip.
- `bach_english_suite_almost_good.wav` — faster than the earlier stable
  take, mostly clean, with one deliberate flub.

Running `evaluate.py` unchanged against `wrong_notes` gave
`wrong_note=0` — despite the clip being built specifically to contain
wrong notes. Inspecting the raw alignment showed why, and it's not a
tuning problem: **`SUBSTITUTE_COST=3` in `align_sequence.py` can never
be chosen by the DP**, because the combined cost of the alternative
(delete the reference note, cost 1; insert the performed note, cost 1 —
total 2) is always cheaper than a direct substitution (cost 3),
regardless of pitch. So a real wrong note always surfaces as one
deletion plus one insertion, never as a `sub` op. This is true by
construction of the cost table, not an artifact of these two clips — the
`wrong_note` verdict path in `evaluate.py` was effectively dead code.

That alone would be fixable by pairing a "missing" reference note with a
time-adjacent "extra" performed note. But the `wrong_notes` clip also
contains passages where the performer plays a **different number of
notes** than the score has (confirmed later by ear against the MIDI
dumps — e.g. a run of sixteenths padded out with extra notes). There is
no 1:1 "this note became that note" correspondence to recover there, and
forcing one onto the data would invent precision that isn't present. As
the user put it: *"bar 2, beats 3 and 4 cannot be matched, too many
extra/missing notes" is useful feedback even without naming which notes
exactly are wrong* — the goal is steering the player toward noticing a
bad passage, not adjudicating note-by-note blame no model can actually
support.

**Design: group, don't pair.** `regions.py` walks Part 1's alignment ops
in order and groups consecutive `del`/`ins` ops into **regions** —
maximal runs broken wherever a `match`/`trans` confirms the alignment is
back on track. Each region gets one shape-based verdict instead of
forced per-note pairing:

| Shape | Verdict |
|---|---|
| `del=1, ins=0` | isolated missing note (unchanged — handled as a plain miss, not a region) |
| `del=0, ins=1` | isolated extra note (unchanged — handled by the existing extra-note classifier, not a region) |
| `del=1, ins=1` | **wrong note** (clean one-for-one substitution) |
| `del=ins>1` | **wrong notes (passage)** — same note count, wrong pitches |
| `ins>del` | **notes added / rhythm subdivided** (e.g. 4 sixteenths become 6) |
| `del>ins` | **notes dropped from a passage** |

Each reported region's confidence is boosted by cross-model
corroboration (`corroborating_regions` in `regions.py`): since both
models align against the *same* reference index space, two models
independently losing track of the same reference notes (shared `dels`)
is strong evidence of a genuine passage mistake rather than one model's
noise. When a region has no deletions to key off (pure insertions),
corroboration falls back to time-range overlap between the two models'
own regions.

**Two bugs found while wiring this in:**
1. Known noise artifacts (short, quiet transients already suppressed
   elsewhere in the extra-notes classifier) were being counted as
   insertions during region-building, producing false "notes added"
   regions out of nothing but noise clusters. Fixed by filtering out
   insertions that look like noise (duration/velocity heuristic, no
   cross-model or harmonic check needed here) *before* grouping into
   regions (`events_without_noise` in `evaluate.py`).
2. The region-reporting loop in `evaluate.py` reused `label` as its
   unpacking variable name, shadowing the function's own `label`
   parameter — the report header printed the first region's
   classification instead of the clip's label. Caught immediately
   because the output was obviously wrong (`=== notes added / rhythm
   subdivided ===` instead of the clip name), not from any subtle
   reasoning; renamed to `region_label` everywhere.

**Result**, all four clips (stable/unstable re-run as a regression
check — numbers unchanged from Step 12, confirming the new layer is a
pure addition, not a behavior change for clips with no disputed
regions):

| Clip | matched | recovered | missing | in_disputed_region | disputed regions reported |
|---|---|---|---|---|---|
| stable | 43 | 1 | 0 | 0 | none |
| unstable | 43 | 1 | 0 | 0 | none |
| almost_good | 42 | 1 | 1 | 0 | 1 — "notes added / rhythm subdivided" at t=6.25–6.29s, corroborated by both models (plausibly the flub itself, caught at the passage level) |
| wrong_notes | 39 | 1 | 3 | 1 | 3 — including one **"wrong notes (passage)"** at t=4.88–5.06s where both models independently lost the same two reference notes |

The `wrong_notes` clip also has several extra notes in the 5.4–7.5s
window that are cross-model-corroborated (flagged, not suppressed) but
didn't form a full region on their own — still useful signal, now
clearly distinguished from noise.

Files: `regions.py` (new), `evaluate.py` (updated: `events_without_noise`,
region-building and reporting wired into the main pass).

### Known open items going into the next step

- **Region time ranges are seconds, not bar/beat.** "ref t=4.88–5.06s"
  means nothing to a player looking at their sheet music; it needs to be
  translated into "measure 2, beats 3–4" (or similar) to be actually
  usable feedback. `bach-3bars-reference.json` currently only carries
  `start_measure`/`end_measure` for the whole excerpt, not a
  measure/beat per note — this needs to be added to the reference-
  generation step (the original `make_reference.py` music21 script,
  not currently copied into this directory) rather than guessed at from
  `onset_sec` and `tempo_bpm` alone, since that requires knowing the
  time signature correctly rather than assuming one.
- **No overall score yet.** All of the above produces rich diagnostic
  detail but not a single comparable number a player could track across
  attempts. Needs a scoring design: how identity (matched/recovered/
  wrong/missing/disputed-region) and timing (tight/noticeable/excessive)
  combine into one figure, and how a disputed region (which has no
  precise per-note count of "how wrong") should be weighted relative to
  a clean per-note verdict.
- Thresholds throughout (duration/velocity cutoffs, harmonic-interval
  list, the 12%/30% timing tiers, the edit-distance costs) are still
  hand-picked from observations on four clips (two Bach takes, two
  minuet takes). Still need testing against more varied pieces/tempos
  before being treated as settled.
- The extra-notes classifier still runs independently of the
  edit-distance alignment's own insertion-cost reasoning; the two
  overlap conceptually and could eventually be merged into one scoring
  pass rather than two sequential ones.

## Step 14 — Part 5: beat-level note-accuracy and rhythm-accuracy scores (done)

Agreed design (user): count beats, not notes, in the evaluated section —
a beat with four sixteenths is worth exactly as much as a beat with one
quarter note, so the score can't be gamed by picking a busy passage. A
wrong/missing note reduces only its own fractional share of its beat; a
disputed region (Part 4) or an isolated unexplained extra note zeroes
its beat *entirely*, since both represent a case where we genuinely
can't say how many notes were actually wrong — partial credit there
would overstate confidence we don't have. Rest beats (no reference note
at all) aren't scored unless something lands in them. Rhythm accuracy
reuses Part 2's timing tiers directly (tight=full credit, noticeable=
half, excessive=zero). Overall = 70% note + 30% rhythm (tunable later).

**Bar/beat, not seconds.** `reference-prep/make_reference.py` already
had the music21 `Measure`/`beat` elements in scope while walking the
score; `el.beat`/`m.number` give the exact measure and beat position as
music21 resolves them from the score's *actual* time signature, no
guessing from `tempo_bpm` + an assumed meter needed. Added `measure`/
`beat` to every note, and a new top-level `"beats"` grid (via
`ts.getBeatOffsets()`) covering every beat cell in the range *including
rests* — needed so a disputed region or stray extra note landing in a
rest can still be scored, not silently skipped for lacking a reference
note of its own. Regenerated `bach-3bars-reference.json`; diffed old vs.
new to confirm the note data itself (onset/pitch) was unchanged, purely
additive.

**Mapping a region/extra note to a beat** is direct when it has a
deleted reference note to anchor to (exact ref index → exact beat).
When it doesn't (a pure-insertion region, or a lone extra note with no
nearby deletion), there's no reference-time anchor at all — only a
*performance*-time one, which can't be compared to the beat grid
directly once tempo is unstable. `estimate_ref_time` (`scoring.py`)
interpolates it from the two nearest confirmed (matched) notes
bracketing it, same local-tempo assumption Part 2's IOI-ratio
measurement already relies on.

**Two bugs found while validating against all four clips:**
1. The beat-bucketing key was each note's own *fractional* beat label
   (e.g. `1.5` for an eighth after beat 1) instead of the coarse beat
   cell from the grid — this bucketed almost one "beat" per note,
   defeating the entire point of scoring by beat rather than by note
   (the stable clip showed 31 "scored beats" when the reference only
   has 9). Caught by sanity-checking the scored-beat count against the
   known reference structure before trusting any score number. Fixed by
   keying on `beat_for_ref_time(cells, ref_onset)` instead.
2. `beat_for_ref_time` silently clamped any estimated time past the
   reference's last beat onto that last beat, instead of recognizing it
   as outside the evaluated range. Confirmed on real data: `almost_good`
   zeroed beat 3:3 because a disputed region's extra notes interpolated
   to ref-times ~6.82–6.87s against a reference that ends at 6.75s — the
   user had, as expected for a 3-measure excerpt, continued playing into
   the next bar after finishing it, and that continuation was being
   blamed on the excerpt's last beat. Fixed: `beat_for_ref_time` now
   returns `None` (excluded from scoring entirely) for anything clearly
   outside `[first beat start, last beat end]`, with a small epsilon
   only for float/duration edge effects at the boundary itself.

**Result**, all four clips, after both fixes:

| Clip | Note accuracy | Rhythm accuracy | Overall (70/30) | Zero-value beats |
|---|---|---|---|---|
| stable | 100.0% | 89.5% | 96.9% | none |
| unstable | 77.8% | 58.1% | 71.9% | 1:1, 1:2 |
| almost_good | 76.4% | 81.4% | 77.9% | 1:2, 1:3 |
| wrong_notes | 65.3% | 70.2% | 66.8% | 3:1, 3:2, 3:3 |

Directionally sound: `wrong_notes` zeroes exactly measure 3 (3:1–3:3) —
the back half of the clip, matching how it was deliberately recorded.
`unstable`'s two zeroed beats are known false positives (see Step 15)
rather than real mistakes, since that clip's ground truth is "notes all
correct, rhythm only." `almost_good`'s real flub (a missing note in a
two-note chord) now shows up as proportional half-credit on its beat
rather than being swamped by the now-excluded end-of-clip artifact.

Files: `scoring.py` (new), `reference-prep/make_reference.py` (updated:
`measure`/`beat` per note, `"beats"` grid), `bach-3bars-reference.json`
(regenerated), `evaluate.py` (updated: wires `compute_scores` into the
final report).

## Step 15 — octave-confusability workaround (done, partial by design)

While investigating `unstable`'s two false-positive zeroed beats (see
Step 14's result table), a user-recalled flub in `almost_good` ("the
first G#4 of bar 3") turned out to be a useful forcing case: the
evaluator reported that exact note as a clean high-confidence "matched"
(ByteDance *and* Basic Pitch both independently agreed on pitch 68 at
roughly the right time). Running `librosa.pyin` — a classical,
model-independent pitch tracker — directly on the raw audio at that
timestamp showed strong, stable energy at G#3 (one octave *below* the
reference pitch), consistent with the user's memory of an octave slip,
not the chord-note miss the evaluator had separately flagged a beat
earlier. (Caveat carried into this finding, not resolved: YIN-family
trackers are themselves prone to octave errors on piano audio with weak
fundamentals, so this is strong corroborating evidence, not proof.)

This case is **not catchable** by anything in the pipeline: both models
agreed on the same (possibly wrong) octave, so there is no competing
signal anywhere in our own data to react to — confirming it would need
a third, independent method (like the `pyin` check itself) run on every
matched note, a much larger scope than "add a flag." Decision (user):
don't chase exact agreement on hand-picked notes/clips any further --
detection error is a structural constraint of both neural transcription
and classical pitch tracking, not a threshold-tuning problem, and a
handful of clips isn't enough signal to tune thresholds against safely
regardless.

What *is* implementable and general: the other half of octave
confusion, where a reference note is "missing" but an unconsumed
performed note sits near its estimated performance time at exactly one
or two octaves away (`OCTAVE_CONFUSION_SEMITONES = (12, 24)`, searched
within `OCTAVE_SEARCH_WINDOW_SEC = 0.25` of the estimated time via
`estimate_perf_time` -- `estimate_ref_time`'s reverse). That's plausible
evidence of the *same* note, octave-misread, rather than a genuinely
missing one, and today would otherwise be double-penalized: once as
"missing," and again (separately, in the extra-notes section) as a
confusing unexplained extra. Reclassified as `"matched
(octave-uncertain)"`, medium confidence, worth `OCTAVE_UNCERTAIN_CREDIT
= 0.5` in scoring — not full credit (we don't know the octave was
right), not zero (we don't know it was wrong either).

Tested against all four clips: fires exactly once, on `almost_good`'s
*other* missing note (the E4 in a two-note chord at measure 2, beat 3),
matched against a ByteDance detection of midi 52 (two octaves below) at
t=3.880s -- not the G#4 case above (expected, per the gap just
described). That same performed note had previously been explained a
different way too (`"likely harmonic resonance ... 19 semitones below
concurrent midi=71"`, against a *different* concurrent note) -- both
explanations are physically plausible and not fully distinguishable
from the data available; the recheck pass runs first and claims it,
which is an acceptable, bounded tradeoff since the credit it grants is
partial, not full. `stable`/`unstable`/`wrong_notes` all show
`octave_uncertain=0` -- no false triggers introduced elsewhere.

Files: `scoring.py` (`estimate_perf_time`, `VERDICT_CREDIT` with
`OCTAVE_UNCERTAIN_CREDIT`), `evaluate.py` (new recheck pass between
identity resolution and Part 2/extra-note classification).

## Step 16 — implausible-note filtering: pitch range + duration sanity (done)

Once the evaluator was wired into the web app (see the project's main
`server/` backend) and run against real recordings instead of just the
four hand-picked clips, two new failure patterns showed up that Steps
7/13/15's existing noise/corroboration logic didn't catch:

- **Pre-roll noise.** A recording where the user spoke/mumbled briefly
  before playing showed ByteDance detecting a burst of note-like events
  during that noise -- some quiet enough to already hit the Step 7
  "noise artifact" suppression, but several loud enough (velocity
  70-80) to be classified `"unexplained extra note (flagged, not
  suppressed)"` instead, forming a disputed region at the very first
  beat and zeroing it even though nothing was actually played wrong
  there.
- **Tail-segment hallucination.** A different recording showed
  ByteDance reporting a note whose *offset* (8.41s) was later than the
  recording's own length (7.15s) -- physically impossible, and
  consistent with the model hallucinating a decaying note into
  zero-padding at the edge of its last processing segment.

Investigating both turned up a third, bigger thing: a recurring
ByteDance artifact at pitches 36/38/40 (C2/D2/E2) -- 12-16+ semitones
below this excerpt's actual range (52-81) -- showed up scattered
*throughout* every one of the original four test clips too, not just
the two new recordings. It had been quietly zeroing beats via the same
extra-note scoring path this whole time; nobody had looked at the
pitch values of the "noise" being flagged before, only duration and
velocity.

Fix (`filter_implausible_notes` in `evaluate.py`, applied to both
models' note lists right after loading, before `identity_pass` ever
runs -- so filtered notes never reach region-grouping, extra-note
classification, or scoring at all):
  - **Pitch floor/ceiling**: any note more than `PITCH_MARGIN_SEMITONES
    = 12` (one octave) below the reference's own lowest pitch, or above
    its highest, is dropped. Reference-relative, not a fixed global
    range, since "plausible" depends on which piece is being played.
  - **Duration sanity**: any note whose offset exceeds the recording's
    actual length (`+ DURATION_TOLERANCE_SEC = 0.05` for rounding) is
    dropped -- this one isn't a heuristic, a note literally cannot end
    after the audio stops.

Filtered notes are never silently discarded from the *report* -- they
land in `feedback.json`'s `"filtered_notes"` list (and the CLI's
printed output) with a reason string, so an over-aggressive filter
would be visible later, not just trusted. This mirrors the project's
running rule of never silently trusting a detector, applied to the
filter itself.

Validated two ways:
  - **Regression on the original four clips**: `bach-clip` and
    `wrong_notes` are byte-for-byte unchanged (no out-of-range or
    past-the-end content in either -- `wrong_notes`'s deliberately
    injected real mistakes are untouched, confirming no loss of
    detection power). `unstable` and `almost_good` both improved
    (77.8%->88.9%, 77.1%->88.2%) once the same pitch-38 artifact was
    removed from them too -- this resolves one of the two open items
    from Step 15 (`unstable`'s false-positive zeroed beats), though not
    both; see below.
  - **The two real recordings that motivated this**: pre-roll-noise
    case went 66.7%->88.9% note accuracy (one zeroed beat left, from an
    in-range extra note the filter correctly leaves alone); tail
    artifact case went 55.6%->77.8% (the duration check caught exactly
    the one physically-impossible note; two in-range, within-duration
    tail notes remain and still cost a beat -- expected, since neither
    new check is designed to touch an uncorroborated-but-plausible
    extra note at a normal pitch).

Files: `evaluate.py` (`filter_implausible_notes`,
`PITCH_MARGIN_SEMITONES`, `DURATION_TOLERANCE_SEC`, new `duration_sec`
param on `evaluate()`, defaults to `None` so the CLI usage without an
audio file keeps working), `server/transcribe.py`
(`get_audio_duration`, header-only read via `soundfile`), `server/app.py`
(computes duration once per submission, passes it through).

## Step 17 — pre/post-roll noise suppression by recording-boundary position (done)

Step 16 fixed the pitch-range and duration-sanity cases, but both real
recordings still had one leftover artifact each: a cluster of
ByteDance-only, uncorroborated notes right at the very start or very
end of the recording -- in-range, within the recording's length, so
neither of Step 16's checks touch them. This was exactly the pre-roll
mumbling case the user originally flagged, and it was showing up as a
spurious "disputed region" (notes added / rhythm subdivided,
uncorroborated by Basic Pitch) rather than being recognized as noise.

The fix, `_is_boundary_noise()` in `evaluate.py`: a performed note is
treated as pre/post-roll noise if **both** hold:
  - its onset falls in the first or last `BOUNDARY_NOISE_WINDOW_SEC`
    (1.0s) of the *recording itself* -- absolute time, not relative to
    where the reference's own first/last note falls, so it doesn't
    assume the performer started exactly on the reference's first beat;
  - the other model does NOT independently report the same pitch at
    the same time (the existing `CROSS_MODEL_WINDOW_SEC` corroboration
    check, already used elsewhere). If both models agree a note is
    there, it's real signal and is never suppressed, however close to
    the edge.

No cluster-size requirement -- a single uncorroborated note in the
window is enough, on the same reasoning as Step 16's other checks:
being transparent about a wrong call (logged, not hidden) matters more
here than being conservative about when to apply it.

Applied in two places that need to agree with each other:
  - `events_without_noise()` (used before Part 4 region-grouping) now
    also drops boundary-noise `"ins"` ops, alongside the existing
    short-and-quiet `_is_probable_noise()` check -- so a boundary-noise
    note can no longer form a "notes added" disputed region.
  - `classify_extra()` (used for the final extra-notes report) gets the
    same check, right after the corroboration check and before the
    harmonic-resonance/noise-artifact checks, so a suppressed boundary
    note still appears in `feedback.json`'s extra-notes list, labeled
    `"likely pre/post-roll noise (suppressed)"` -- never silently
    dropped from the report, same discipline as every other filter in
    this pipeline.

Both call sites needed a `duration_sec` plumbed through (`evaluate()`
already had it from Step 16); the CLI path still defaults it to `None`,
which makes `_is_boundary_noise()` always return `False` -- so this is
inert for every existing CLI-driven test, by construction, not by
coincidence.

Validated two ways:
  - **Regression on the original four clips** (CLI, no `duration_sec`):
    byte-for-byte unchanged from Step 16 -- confirms the new check
    truly does nothing without a real recording length.
  - **Both real recordings, re-run end to end**: the lone disputed
    region in each recording's report disappeared (both were exactly
    the pre/post-roll notes this step targets), and extra notes that
    used to read `"unexplained extra note (flagged, not suppressed)"`
    at t=0.08s/t=6.69s etc. now correctly read
    `"likely pre/post-roll noise (suppressed)"`. **Scores were
    unchanged in both cases** (note/rhythm/overall identical to the
    Step 16 numbers) -- these particular notes were never actually
    costing points (they fell outside every scored beat's range), so
    this step is a pure reporting-accuracy fix, not a score change.
    That's a useful finding in itself: the pre-roll case's remaining
    zeroed beat (1:1/1:3 in recording 1, 2:2 in recording 2) is coming
    from something else entirely, not from these boundary notes --
    worth re-checking against the per-note table before assuming it's
    resolved.

Files: `evaluate.py` (`_is_boundary_noise`, `BOUNDARY_NOISE_WINDOW_SEC`,
`events_without_noise()` and `classify_extra()` both gained/reused a
`duration_sec` parameter).

## Step 18 — sustain-pedal-aware noise suppression (done)

Motivation: on real-piano recordings (reverb + sustain pedal, see the
two `orfeo-euridice-4bars-reference` test recordings), several
"unexplained extra note (flagged, not suppressed)" entries had
suspiciously long durations (1.0-2.5 seconds) that none of the existing
noise checks caught -- `_is_probable_noise` requires a SHORT duration,
so a long one escapes it outright. ByteDance's own sustain-pedal
detection (`est_pedal_events`) was being computed at transcription time
but thrown away -- never written to `bytedance.json`, never seen by
the evaluator. Pedal-down periods are exactly the condition under which
a note's own decay or a bled-through overtone can look like a separate,
long-held "note" to the model.

Fix: `transcribe.py`'s `run_bytedance()` now also writes a sibling
`bytedance-pedal.json` (list of `{onset, offset}` pedal-down windows)
next to `bytedance.json`. `evaluate.py` gained `load_pedal_events()`
(returns `[]` if the sibling file doesn't exist -- fully backward
compatible with every performance folder recorded before this step,
and with the CLI's 4 classic clips, which have no pedal file and are
confirmed byte-for-byte unchanged) and `_is_pedal_sustain_bleed()`: a
note counts as pedal bleed only if it's QUIET (same
`EXTRA_NOTE_LOW_VELOCITY` bar used elsewhere) AND its onset falls
strictly inside a pedal-down window that started before it (not a note
landing exactly as the pedal goes down -- that's ordinary playing, not
bleed). Deliberately no duration requirement, unlike
`_is_probable_noise` -- that's the whole point, since the long-duration
cases are exactly what this is meant to catch. Wired into both
`events_without_noise()` (region-building) and `classify_extra()` (the
reported verdict), same pattern as Step 17's boundary-noise check.

Validation: regenerated ByteDance's transcription for
`20261112T221748Z__orfeo-euridice-4bars-reference` (non-destructively,
into a `rerun-pedal-20261113/` subfolder, since the pedal file didn't
exist when it was first recorded) and re-ran evaluation. 19 of the 50
previously-unexplained extra notes were correctly reclassified as
"likely pedal-sustain bleed (suppressed)", including the three longest
(1916ms, 1542ms, 1086ms) -- each with a specific pedal-down window and
how long the pedal had been held before the note's onset. Disputed
regions dropped from 25 to 21 as a direct result (fewer spurious
"notes added" verdicts). The 4 classic CLI clips are unaffected
(confirmed identical note/rhythm/overall percentages to Step 17's
numbers).

Note accuracy and overall score did NOT change (still 0.0% / 14.07%,
identical to before this step) -- expected, not a bug: this is a
report-accuracy fix, not a scoring fix, and the beat-zeroing saturation
issue (see Known open items) still zeroes all 12 beat cells regardless
of how many fewer, more-accurately-labeled extra notes remain. The 19
reclassified notes simply moved from one zero-trigger bucket to
another; removing them from the disputed-region count did free up 4
regions, but not enough beat cells to escape saturation on this
particular (very dense) reference.

Files: `transcribe.py` (`run_bytedance()` now also writes
`bytedance-pedal.json`), `evaluate.py` (`load_pedal_events()`,
`_is_pedal_sustain_bleed()`, `events_without_noise()` and
`classify_extra()` both gained a `pedal_events` parameter).

## Step 19 — proportional beat scoring, replacing all-or-nothing zeroing (done)

Motivation: every improvement from Steps 16-18 (and the pitch-tuning
and pedal-aware experiments logged in memory, not yet in this repo)
measurably improved the underlying transcription/report accuracy on
real recordings, but note_accuracy stayed exactly 0.0% regardless --
confirmed structural, not a transcription problem. The
`orfeo-euridice-4bars-reference` reference has 104 notes across only
12 beat cells (dense: both staves merged, with ornaments). The
original Part 5 rule zeroed a beat's ENTIRE credit the moment any
disputed region or unexplained extra note touched it. On all three
real test recordings, enough regions/extras existed (16-29 regions,
43-108 extras) to touch all 12 cells at least once, saturating the
score to 0% even though 74-88% of notes were actually matched
correctly (per `identity_summary`) -- the score could not distinguish
a good take from a bad one once touch density exceeded cell count.

Decision (asked the user explicitly, since this reverses a previously
agreed principle): make beat credit proportional instead of
all-or-nothing, even at the cost of some of the original caution ("we
can't say how many notes in a disputed region were wrong, so partial
credit overstates confidence"). Confirmed: a usable, differentiating
signal was worth more than that caution once it was costing 100% of
the signal 100% of the time on real recordings.

Design: every reference note AND every unexplained/corroborated extra
note (not suppressed as noise/resonance/pedal bleed) is now its own
one-unit "slot" in whichever beat it belongs to; a beat's score is the
average credit across all its slots. A missing/wrong/disputed-region
reference note costs only its own slot (as missing/wrong notes already
did -- disputed regions just needed to stop being special-cased to
zero the whole beat). An extra note costs its own zero-credit slot,
estimated into reference time the same way as before. This turned out
to simplify `compute_note_score()` significantly: disputed regions'
`dels` already appear once each in the per-reference-note loop (as an
"in disputed region #N" verdict, previously skipped and separately
zeroed -- now just falls through to `VERDICT_CREDIT`'s existing 0.0
default, no special-casing needed), and every region's `ins` performed
note already appears in the extra-notes list with its own verdict (a
performed note can't be both "consumed" by a region's grouping and
still show up as "not consumed" in `bd_extra` -- it's one or the
other), so counting it both via regions AND via extras would have
double-penalized it. Net result: `compute_note_score()` no longer
needs region data, `bd_notes`, or `ref_events` as inputs at all --
regions are now purely a reporting/display concept, never a second,
separate input to scoring. `compute_scores()`'s signature shrank to
`(reference, results, confirmed_pairs, extra_results, ...)`.

Validation:
- 4 classic CLI clips: `bach-clip` unchanged (100%/89.5%/96.9% -- no
  disputed regions to begin with, so nothing to be proportional
  about). The other three, which previously had 1-3 fully-zeroed
  beats, now have 0: `almostgood-clip` 88.2%->97.1% note accuracy
  (86.2%->92.4% overall), `wrongnotes-clip` 65.3%->85.6%
  (66.8%->81.0%), `unstable-clip` 88.9%->94.4% (79.7%->83.6%) -- all
  moved up, as expected, since a single bad note in an otherwise-good
  beat no longer costs the whole beat.
- The three real `orfeo-euridice-4bars-reference` recordings, re-scored
  against their existing (unmodified) transcription outputs -- no
  re-transcription needed, confirming once again that evaluator-only
  changes don't require re-running the ML models:
  | recording | matched+recovered | note_accuracy | overall |
  |---|---|---|---|
  | real piano #1 | 83/104 (80%) | 55.1% | 52.7% |
  | real piano #2 | 77/104 (74%) | 45.5% | 48.1% |
  | digital piano | 92/104 (88%) | 66.1% | 61.4% |

  All three moved from an identical, uninformative 0.0% to a
  meaningfully differentiated score that ranks in the same order as
  the underlying identity-match quality (88% > 80% > 74%) -- matching
  the user's own subjective read of the three recordings from when
  they were first submitted ("the last one is much clearer").

Files: `scoring.py` (`compute_note_score()` rewritten around per-slot
credit instead of a separate `zeroed_beats` override;
`compute_scores()` signature shrank), `evaluate.py` (call site updated
to match).

### Known open items going into the next step

- The G#4 case from Step 15 remains a known, structural blind spot: when
  both models agree on the same wrong octave, nothing in this pipeline
  can catch it without an independent pitch-verification method run on
  every matched note -- out of scope for now, documented rather than
  chased further.
- Resolved by Step 19: the binary full-zero-per-touched-beat rule that
  made note_accuracy saturate to 0.0% on any dense/real recording,
  regardless of corroboration status or how many notes were actually
  right. Superseded the item that used to be here about corroboration
  only flowing into the *label*, not the *score* -- still true in one
  sense (a corroborated "extra note (seen by both models)" still costs
  the same zero-credit slot as an uncorroborated one, since
  `SCORED_EXTRA_VERDICTS` treats them identically), but no longer the
  dominant problem, since a single such slot now only costs its own
  share of a beat rather than the whole thing. Confidence-weighting
  *within* that shared slot cost (e.g. should a corroborated extra
  note cost less than an uncorroborated one, now that both are
  fractional rather than both being an identical full zero) remains a
  real, smaller, separate refinement -- not yet designed, and lower
  priority now that the saturation problem itself is fixed.
- Scoring weights (70/30 note/rhythm, the 0.5 octave-uncertain credit)
  are reasonable starting points agreed with the user, not validated against
  a ground-truth scoring target -- there isn't one yet, since "what
  score should this performance get" is itself a judgment call, not a
  fact to test against.

## Investigation: diagnosing real-recording transcription quality (not yet acted on)

Two new real-piano recordings of `bach-3bars-reference` (no sustain
pedal used) came in sounding "garbled" to the ear -- ByteDance's
output had sections that sounded like noise, Basic Pitch's output was
accurate where it detected anything but missed most of the piece. The
question was whether this is an evaluator problem (same shape as Step
19) or a transcription-quality ceiling.

**Diagnosis:** not an evaluator bug. Both recordings have a genuine
~1.3-1.4s lead-in silence before playing starts (confirmed via 100ms
RMS-window dBFS profiling of `recording.wav`), which both models
correctly reflect in their earliest onsets -- not the source of the
"garbled" impression. The real finding: pulling every disputed region
from both recordings' reports showed 6/6 (recording 1) and 8/9
(recording 2) were **independently corroborated by Basic Pitch** --
both models lost track of the exact same reference notes at the exact
same spots. That rules out "random per-model noise" and points at two
specific, reproducible failure patterns:

1. **Simultaneous/near-simultaneous two-voice onsets.** The piece is a
   two-voice texture (melodic 16th notes + a sustained harmony note in
   a nearby register landing within ~190ms of the melody note). Both
   models tend to lose or merge notes exactly there -- a documented
   weak spot for AMT models.
2. **A reproducible octave-confusion bug.** Reference pitch 57 (A3) at
   ref time 3.0s (measure 2, beat 2.0) is replaced with pitch 69 (A4,
   exactly +12 semitones) in BOTH recordings independently. Research
   (see below) confirms this is a known, long-studied AMT failure mode:
   low piano notes radiate little energy at their true fundamental, so
   the recorded signal is dominated by the first harmonic an octave up,
   and two notes an octave apart share the same harmonic series --
   there's a genuine acoustic ambiguity to resolve, not a bug to patch.

Checked whether a general-purpose multimodal LLM (sending base64
audio to e.g. Gemini/GPT-4o-audio) could do better at this kind of
polyphonic note-level transcription -- researched rather than assumed.
Verdict: no. CMI-Bench (MIR benchmark for audio-text LLMs) found the
best general audio-LLM scored ~8% (Gmean) on pitch/key tasks vs >74%
for specialized MIR baselines; a 2025 study found Gemini 2.5's
accuracy on simple audio music tasks (chord-quality ID) drops 40-45
percentage points versus the same task given symbolic MIDI input, with
chord ID on raw audio falling below 10% in some setups. No benchmark
found shows any general multimodal LLM competitive with specialized
AMT models (ByteDance's own checkpoint scores ~97% F1 on standard piano
transcription benchmarks). Not pursued further -- wrong tool for this
specific job, confirmed by research rather than by building and
finding out.

### Experiment 1 -- ByteDance onset_threshold tuning (null result)

`PianoTranscription` exposes `onset_threshold` (default 0.3),
`offset_threshod` (typo in the upstream library), `frame_threshold`
(0.1), and `pedal_offset_threshold` (0.2) as plain settable instance
attributes (`piano_transcription_inference/inference.py`'s
`__init__`) -- not constructor kwargs, but settable on the
transcriptor object before calling `.transcribe()`, since it rebuilds
its `RegressionPostProcessor` from `self` on every call. Each of the
88 piano-key pitch classes is thresholded independently
(`utilities.py`'s `get_binarized_output_from_regression`, looped per
pitch) -- a structural reason threshold tuning can't fix cross-pitch
confusion: two different pitch channels never compete for the same
onset slot.

Tested: re-transcribed both bach recordings with `onset_threshold=0.2`
instead of 0.3 (non-destructively, into `rerun-threshold/` subfolders
inside each performance's own folder), re-scored against the
unchanged Basic Pitch output. Result: no meaningful change. All 6 of
recording 1's disputed regions reappeared with *identical*
deleted/inserted pitches, including the same octave-confusion bug at
the same spot. Recording 2 recovered one previously-missing note with
negligible score impact (note_accuracy 61.8%->61.7%, i.e. noise).
Lowering the threshold only added a few more low-confidence extra
notes (caught cleanly by the existing noise/extra-note classifier from
Steps 16-18).

Conclusion: these failures are the model confidently producing the
WRONG pitch or dropping real polyphonic content outright, not
borderline/under-threshold detections -- threshold tuning structurally
cannot fix either failure mode. Default (0.3) left unchanged; not
adopting a lower threshold.

### Experiment 2 -- WPE dereverberation preprocessing (promising, not yet adopted)

Installed `nara_wpe` (pure-Python Weighted Prediction Error
dereverberation; also pulled in `bottleneck` as a dependency).
Dereverberated recordings non-destructively into each performance's
own `rerun-dereverb/recording.wav` (STFT via `scipy.signal.stft`,
`nara_wpe.wpe.wpe_v8` with `taps=10, delay=3, iterations=3`, treating
the mono signal as a single-channel multi-frequency-bin input shaped
`(F, D=1, T)`, then `scipy.signal.istft` back to a WAV), then
re-transcribed with both models and re-scored against the reference.

Tested on 5 real-piano recordings total -- both new `bach-3bars`
clips, plus all three `orfeo-euridice-4bars` recordings from the
start of this session (chosen because they were described as the
worst reverb cases: "a lot of reverb and harmonics... sustain pedal
mixing notes"):

| recording | note_accuracy (orig -> dereverbed) | rhythm_accuracy | overall | regions | extra notes |
|---|---|---|---|---|---|
| bach #1 (20261113T220042Z) | 65.8% -> 65.8%* | n/a -> n/a | 70.5% -> 73.0% | 6 -> 6 | n/a |
| bach #2 (20261113T220228Z) | 61.8% -> 70.7% | n/a -> n/a | 67.8% -> 74.9% | 9 -> 6 | n/a |
| orfeo real piano #1 (20261112T221748Z) | 55.1% -> 54.9% | 46.9% -> 62.8% | 52.7% -> 57.3% | 25 -> 23 | 70 -> 94 |
| orfeo real piano #2 (20261112T222141Z) | 45.5% -> 50.3% | 54.3% -> 56.6% | 48.1% -> 52.2% | 29 -> 22 | 108 -> 124 |
| orfeo digital piano (20261112T222724Z) | 66.1% -> 67.6% | 50.5% -> 52.5% | 61.4% -> 63.1% | 15 -> 12 | 92 -> 93 |

*bach #1's note_accuracy barely moved, but matched notes rose 25->29
while recovered-via-Basic-Pitch dropped 5->2 and missing rose 3->4 --
ByteDance resolved more notes directly, at a slightly different
mix, netting out roughly flat on note identity specifically.

Findings:
- **Overall score improved on all 5 recordings tested, including the
  cleanest one** (the digital piano, which has little real room
  reverb) -- no regression observed on the case expected to benefit
  least, which is a good sign this isn't overfit to heavy reverb.
- The subjective quality ranking identified by ear at the very start of
  this session (digital clearest > real piano #1 > real piano #2) is
  preserved both before and after dereverberation.
- The improvement's source differs by recording: orfeo real piano #1's
  gain is almost entirely **rhythm accuracy** (46.9%->62.8%) --
  dereverberation sharpens smeared attack transients into more precise
  onset timing without changing which notes get identified. Real piano
  #2 and both bach recordings improve on **both** note identity
  (fewer disputed regions: 29->22, 9->6) and rhythm.
- The octave-confusion bug (pitch 57->69 in the bach recordings)
  persists **unchanged** after dereverberation in both recordings it
  appears in -- confirms it's a harmonic-ambiguity problem independent
  of reverb, exactly as the diagnosis above predicted.
- **The real cost: extra/noise note counts rose substantially on all
  three orfeo recordings** (70->94, 108->124, 92->93). WPE
  dereverberation is an aggressive, model-free signal-processing step
  the transcription models never saw in training, and it introduces
  some spurious content even as it cleans up the real signal. The
  existing noise/extra-note classification (Steps 16-18) absorbed
  most of this without hurting the score net, but it's a real
  tradeoff, not a free improvement.

**Status: not wired into `transcribe.py` yet (explicit user decision --
"we will not wire it into our main transcription process yet").**
Before adopting it as a real pipeline stage, still need to check it
doesn't regress the clean/synthetic CLI regression clips (`bach-clip`,
etc. -- likely studio-clean audio where an aggressive dereverb pass
could be pure downside with nothing to fix) and get a clearer read on
whether the extra-note cost is worth the rhythm/identity gain across a
wider sample. Logged here and in memory
(`bytedance_threshold_tuning_null_result.md` for Experiment 1; a
dereverberation memory to follow) so this doesn't need re-deriving.

Next direction the user wants to explore instead/in addition: other
specialized AMT models -- **Aria-AMT** and **Transkun V2** -- as
alternatives or an ensemble member alongside ByteDance's
`piano_transcription_inference` and Basic Pitch.

### Experiment 3 -- Transkun V2 as a ByteDance replacement (large, consistent win; not yet wired in)

Researched both candidates before testing anything. **Aria-AMT**
(EleutherAI, Apache-2.0): requires a git install (`pip install -e .`,
needs Python 3.11), is GPU-oriented (its `-compile`/`-q8` speedup
flags reference GPU-only features, no CPU timing mentioned anywhere),
and its README publishes no benchmark numbers at all -- couldn't even
compare it on paper. **Transkun V2** (Yujia Yan, ISMIR 2024, MIT):
one-line `pip install transkun`, CPU is the default device, and its
published MAESTRO numbers already beat ByteDance's own reported
figures on the same clean benchmark (98.4% note onset F1 vs
ByteDance's 96.7%). Picked Transkun V2 to test first on that basis.
Neither model's README reports any real-room/reverberant-recording
benchmark -- same gap ByteDance's own numbers have.

Installed via `pip install transkun` (pulls in its own torch/
torchaudio/tensorboard/pandas stack -- large but self-contained, no
native/system dependency beyond Python). Ran via its CLI:
`transkun <recording.wav> <out.mid>` (default bundled checkpoint, CPU,
no extra flags needed) -- non-destructively, into each performance's
own `rerun-transkun/transkun.json` (converted from the output MIDI via
`pretty_midi`, same `{onset, offset, pitch, velocity}` shape as
`bytedance.json`). ~8-14s per clip on CPU, comparable to the existing
models. Scored by substituting `transkun.json` for `bytedance.json` as
the primary identity source in `evaluate()`, keeping the existing
(unchanged) `basicpitch.json` as the secondary/recovery source --
Transkun's own pedal output isn't produced by this CLI invocation, so
`load_pedal_events()` correctly fell back to `[]` (no sibling
`*-pedal.json` file) rather than erroring.

Results, same 5 recordings as Experiment 2, compared against the
ByteDance+Basic Pitch baseline:

| recording | note_accuracy | rhythm_accuracy | overall | regions | extra notes |
|---|---|---|---|---|---|
| bach #1 | 65.8% -> **100%** | 81.6% -> 88.4% | 70.5% -> **96.5%** | 6 -> **0** | n/a -> **0** |
| bach #2 | 61.8% -> **98.6%** | 82.1% -> 83.7% | 67.8% -> **94.1%** | 9 -> **0** | n/a -> 2 |
| orfeo real #1 | 55.1% -> **78.2%** | 46.9% -> **79.3%** | 52.7% -> **78.5%** | 25 -> **6** | 70 -> **12** |
| orfeo real #2 | 45.5% -> **80.9%** | 54.3% -> 77.6% | 48.1% -> **79.9%** | 29 -> **3** | 108 -> **20** |
| orfeo digital | 66.1% -> **86.5%** | 50.5% -> 69.2% | 61.4% -> **81.3%** | 15 -> **4** | 92 -> 50 |

Findings:
- **A large, consistent win on every single recording tested** -- not
  a mixed result like the dereverberation experiment. Note accuracy
  improved 13-37 percentage points, disputed regions dropped 3-10x,
  and both bach recordings (no sustain pedal involved) came back with
  **zero disputed regions** -- including the one with the reproducible
  octave-confusion bug (ref pitch 57 -> detected 69 at ref time 3.0s,
  present in BOTH bach recordings under ByteDance, persisted even
  through dereverberation in Experiment 2). Transkun V2 simply didn't
  make that mistake in either recording.
- The real-piano recordings' rhythm accuracy improved sharply too
  (orfeo real #1: 46.9%->79.3%), suggesting Transkun's onset timing is
  meaningfully more precise on reverberant recordings, not just its
  pitch identity.
- **Ranking caveat:** under ByteDance, orfeo real #1 (55.1%) scored
  higher than real #2 (45.5%); under Transkun that order flips, barely
  (78.2% vs 80.9%) -- the two are close enough now that which one
  "sounds worse" may not be a stable signal either model can be
  trusted to reproduce exactly. The digital piano recording stays
  clearly highest under both models, which is the more important
  signal to preserve.
- **Real caveat, not yet resolved:** Transkun's CLI doesn't produce
  sustain-pedal events the way ByteDance's `est_pedal_events` does, so
  Step 18's pedal-bleed suppression can't run when Transkun is primary
  -- some genuine pedal-sustain bleed is likely landing in the
  "unexplained extra note" bucket instead of being correctly
  classified and suppressed. This is the most likely explanation for
  the digital piano recording's comparatively high extra-note count
  (50, split ~even between "unexplained extra note" and "extra note
  seen by both models") despite its otherwise-strong 86.5% note
  accuracy. If Transkun is ever wired in as the primary model, ByteDance
  would likely still need to run alongside it just to supply pedal data
  -- an argument for an ensemble rather than a straight swap.

**Status: not wired into `transcribe.py` yet.** This result is strong
enough to be worth pursuing, but still needs: (1) the same CLI
regression-clip check already pending for Experiment 2 (confirm no
regression on clean/synthetic `bach-clip` etc.); (2) a decision on the
pedal-data gap above (ensemble vs swap); (3) checking Transkun's much
heavier dependency footprint (torch/torchaudio/tensorboard/pandas, ~50MB+
wheel) against the project's current dependency weight before adopting
it for real. Logged here and in memory
(`transkun_v2_vs_bytedance_experiment.md`).
