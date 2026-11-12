// Builds the structured "performance diff" from the DTW-aligned onset
// sequences. This is the deterministic-layer output that, in the full
// design, would be sent to the LLM layer for feedback -- here it's just
// surfaced directly for inspection.

const ALIGNMENT_LOW_CONFIDENCE_THRESHOLD = 0.7; // cosine distance (on the DTW alignment feature) above this = likely a bad match, not a real pitch deviation
const INTERVAL_MATCH_TOLERANCE_SEMITONES = 1.5;

/**
 * @param {object} reference  reference JSON (from make_reference.py)
 * @param {object} performance  output of analyzePerformance()
 * @param {object} alignment  output of dtwAlign()
 * @param {object} [options]
 * @param {"fixed"|"adaptive"} [options.tempoMode="fixed"]
 *   "fixed": diff timing against the reference's literal tempo (did you hit
 *   the target tempo). "adaptive": estimate the player's own overall tempo
 *   from the data (median actual/expected IOI ratio), rescale the reference
 *   timeline by it, and diff against that -- reports rhythmic consistency
 *   relative to the player's own pace, not against an absolute target.
 */
function buildDiff(reference, performance, alignment, options = {}) {
  const tempoMode = options.tempoMode || "fixed";
  const { path } = alignment;
  const refOnsets = reference.onsets_sec;
  const perfOnsets = performance.onsets;

  // Collapse the DTW path to one match per reference note (first perf
  // onset mapped to it), since a reference note should correspond to
  // exactly one played attack in a correct performance.
  const matched = [];
  const seenRef = new Set();
  for (const [perfIdx, refIdx] of path) {
    if (seenRef.has(refIdx)) continue;
    seenRef.add(refIdx);
    matched.push({ perfIdx, refIdx });
  }

  // A performance onset reused across multiple reference notes means two
  // expected attacks were not detected as separate onsets (likely slurred/
  // missed detection), not a timing problem on either note individually.
  const perfIdxCounts = {};
  matched.forEach(({ perfIdx }) => { perfIdxCounts[perfIdx] = (perfIdxCounts[perfIdx] || 0) + 1; });
  const mergedNoteCount = matched.filter(({ perfIdx }) => perfIdxCounts[perfIdx] > 1).length;

  // Anchor on the first matched pair to express deviations relative to a
  // shared start, rather than assuming the performance starts at t=0.
  const anchorRefSec = refOnsets[matched[0]?.refIdx ?? 0];
  const anchorPerfSec = perfOnsets[matched[0]?.perfIdx ?? 0] ?? 0;

  let tempoScale = 1;
  if (tempoMode === "adaptive" && matched.length > 1) {
    const ratios = [];
    for (let i = 1; i < matched.length; i++) {
      const expectedIOI = refOnsets[matched[i].refIdx] - refOnsets[matched[i - 1].refIdx];
      const actualIOI = perfOnsets[matched[i].perfIdx] - perfOnsets[matched[i - 1].perfIdx];
      if (expectedIOI > 0) ratios.push(actualIOI / expectedIOI);
    }
    tempoScale = ratios.length ? median(ratios) : 1;
  }

  const expectedOnsetSec = (refIdx) =>
    anchorPerfSec + (refOnsets[refIdx] - anchorRefSec) * tempoScale;

  const perNote = matched.map(({ perfIdx, refIdx }) => {
    const expectedSec = expectedOnsetSec(refIdx);
    const actualSec = perfOnsets[perfIdx];
    // Alignment-quality distance, on whatever feature DTW actually used
    // (pitch-class vector when a melody pitch was detected, HPCP chroma
    // as fallback) -- this is what should drive the confidence flag.
    const alignmentDistance = round(
      cosineDistance(performance.alignmentFeatures[perfIdx], reference.notes[refIdx].chroma)
    );
    // Raw HPCP chroma distance, kept as a secondary diagnostic only.
    const chromaDistance = round(
      cosineDistance(performance.chroma[perfIdx], reference.notes[refIdx].chroma)
    );
    return {
      refIdx,
      perfIdx,
      expectedOnsetSec: round(expectedSec),
      actualOnsetSec: round(actualSec),
      timingDeviationSec: round(actualSec - expectedSec),
      expectedPitchNames: reference.notes[refIdx].pitch_names,
      amplitude: performance.amplitudes[perfIdx],
      alignmentDistance,
      chromaDistance,
      lowConfidenceMatch: alignmentDistance > ALIGNMENT_LOW_CONFIDENCE_THRESHOLD,
      mergedWithAnotherNote: perfIdxCounts[perfIdx] > 1,
    };
  });

  attachIntervalChecks(perNote, reference, performance);

  const deviations = perNote.map((n) => n.timingDeviationSec);
  const tempoStability = computeTempoStability(perNote);
  const lowConfidenceCount = perNote.filter((n) => n.lowConfidenceMatch).length;
  const intervalChecked = perNote.filter((n) => n.intervalFromPrev && n.intervalFromPrev.pitchClassMatch !== null);
  const intervalMatches = intervalChecked.filter((n) => n.intervalFromPrev.pitchClassMatch);

  return {
    tempoMode,
    tempoScale: round(tempoScale),
    estimatedTempoBpm: reference.tempo_bpm ? round(reference.tempo_bpm / tempoScale) : null,
    noteCount: perNote.length,
    referenceNoteCount: reference.notes.length,
    onsetCountRatio: round(performance.onsets.length / reference.notes.length),
    mergedNoteCount,
    meanAbsTimingDeviationSec: round(mean(deviations.map(Math.abs))),
    maxAbsTimingDeviationSec: round(Math.max(0, ...deviations.map(Math.abs))),
    tempoStability,
    lowConfidenceMatchCount: lowConfidenceCount,
    lowConfidenceMatchFraction: round(lowConfidenceCount / perNote.length),
    intervalCheckCoverageFraction: round(intervalChecked.length / Math.max(perNote.length - 1, 1)),
    intervalMatchFraction: intervalChecked.length ? round(intervalMatches.length / intervalChecked.length) : null,
    perNote,
    confidence: {
      timing: "high",
      dynamics: "high",
      melodyPitch: "medium",
      harmonyChords: "not_attempted",
    },
  };
}

/**
 * For each matched note, find the estimated pitch (Hz) at/near its onset
 * from the predominant-pitch contour, then compare the semitone interval
 * to the previous matched note -- both as detected and as expected from the
 * score -- on a mod-12 (pitch-class) basis so octave errors in the pitch
 * tracker don't register as wrong notes.
 */
function attachIntervalChecks(perNote, reference, performance) {
  let prevHz = null, prevMidi = null;
  for (const n of perNote) {
    const hz = performance.onsetPitchHz[n.perfIdx];
    const midi = Math.max(...reference.notes[n.refIdx].midi_pitches); // melody = highest voice, by convention

    if (prevHz && hz) {
      const estimatedSemitones = 12 * Math.log2(hz / prevHz);
      const expectedSemitones = midi - prevMidi;
      const circularDiff = circularSemitoneDiff(expectedSemitones, estimatedSemitones);
      n.intervalFromPrev = {
        expectedSemitones: round(expectedSemitones),
        estimatedSemitones: round(estimatedSemitones),
        pitchClassMatch: Math.abs(circularDiff) <= INTERVAL_MATCH_TOLERANCE_SEMITONES,
      };
    } else {
      n.intervalFromPrev = { expectedSemitones: null, estimatedSemitones: null, pitchClassMatch: null };
    }
    prevHz = hz || prevHz; // carry forward through brief unvoiced gaps
    prevMidi = midi;
  }
}

/** Signed shortest distance between two semitone intervals on a mod-12 circle, in (-6, 6]. */
function circularSemitoneDiff(expected, estimated) {
  let diff = ((expected - estimated) % 12 + 12) % 12; // [0, 12)
  if (diff > 6) diff -= 12; // (-6, 6]
  return diff;
}

function computeTempoStability(perNote) {
  // Ratio of consecutive actual inter-onset intervals to expected ones
  // (expected already reflects tempoMode's scaling). A value consistently
  // near 1.0 = stable tempo relative to that baseline; high variance =
  // rushing/dragging in specific spots.
  const ratios = [];
  for (let i = 1; i < perNote.length; i++) {
    const expectedIOI = perNote[i].expectedOnsetSec - perNote[i - 1].expectedOnsetSec;
    const actualIOI = perNote[i].actualOnsetSec - perNote[i - 1].actualOnsetSec;
    if (expectedIOI > 0) ratios.push(actualIOI / expectedIOI);
  }
  if (ratios.length === 0) return { meanRatio: null, stdDevRatio: null, ratios: [] };
  const m = mean(ratios);
  const variance = mean(ratios.map((r) => (r - m) ** 2));
  return { meanRatio: round(m), stdDevRatio: round(Math.sqrt(variance)), ratios: ratios.map(round) };
}

function mean(arr) {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function median(arr) {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function round(x) {
  return Math.round(x * 1000) / 1000;
}
