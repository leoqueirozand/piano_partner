// Runs Essentia.js over a decoded audio buffer to extract the deterministic,
// non-LLM signals: onset times, per-onset amplitude (dynamics), and a
// predominant-pitch contour. No full polyphonic transcription is attempted.

let essentiaInstance = null;

async function getEssentia() {
  if (essentiaInstance) return essentiaInstance;
  // EssentiaWASM / Essentia are loaded globally via <script> tags in index.html.
  const wasmModule = await EssentiaWASM();
  essentiaInstance = new Essentia(wasmModule);
  return essentiaInstance;
}

// Essentia's OnsetRate algorithm hard-requires 44100Hz input and throws
// (uncatchably, in the WASM build) on a mismatched sample rate. Phones
// commonly record at 48000Hz, so every file must be resampled to 44100Hz
// before any Essentia call, regardless of its original rate.
const ESSENTIA_SAMPLE_RATE = 44100;

async function resampleToEssentiaRate(decodedBuffer) {
  const resampleCtx = new OfflineAudioContext(
    1, // mono output
    Math.ceil(decodedBuffer.duration * ESSENTIA_SAMPLE_RATE),
    ESSENTIA_SAMPLE_RATE
  );
  const source = resampleCtx.createBufferSource();
  source.buffer = decodedBuffer;
  source.connect(resampleCtx.destination);
  source.start();
  const resampled = await resampleCtx.startRendering();
  return resampled.getChannelData(0);
}

async function decodeAudioFile(file) {
  const arrayBuffer = await file.arrayBuffer();
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await audioCtx.decodeAudioData(arrayBuffer);
  const samples = await resampleToEssentiaRate(decoded);
  return { samples, sampleRate: ESSENTIA_SAMPLE_RATE };
}

// Verified against the raw WASM build (Node, decoding a real recording with
// ffmpeg -- bypassing the browser entirely) that SuperFluxExtractor's
// positional args DO match the documented order here -- unlike
// PredominantPitchMelodia, this one isn't alphabetically scrambled.
//
// ratioThreshold=0 is Essentia's OWN documented recommendation "for
// low-energy onsets", but it does NOT hold up on real piano audio: tested
// directly against minueto-estavel.m4a, ratioThreshold=0 produced 0 onsets
// at threshold=0.05 and only 3-8 onsets (far too few for a 17s take) at
// every other threshold value tried. ratioThreshold=8 was the only setting
// that stayed robust (~45-50 onsets, a sane count) across every threshold
// tested on that same file. The default (16) was too strict and also gave 0
// onsets at threshold=0.05.
const SUPERFLUX_RATIO_THRESHOLD = 8;

// `combine` (ms) merges onsets closer together than this into one. Default
// 20ms left a near-duplicate secondary peak ~23-29ms after nearly every real
// attack uncombined (SuperFlux's novelty curve often overshoots right after
// a genuine onset) -- confirmed via a raw onset/amplitude console dump on a
// real recording, where this pattern repeated at almost every detected note.
// Real note-to-note spacing in this piece is 200ms+, so 40ms still can't
// merge genuinely distinct fast notes.
const SUPERFLUX_COMBINE_MS = 40;

/**
 * Detect onset times (seconds) using Essentia's SuperFlux algorithm. Chosen
 * over the simpler OnsetRate because OnsetRate's fixed internal settings
 * were observed to merge fast, closely-spaced attacks (e.g. a short scalar
 * run or ornament) into a single detected onset -- SuperFlux's smaller hop
 * size and maximum-filtered novelty curve resolve these better.
 *
 * Falls back to OnsetRate if SuperFlux ever returns zero onsets (defensive;
 * not expected to trigger with ratioThreshold=8, but cheap insurance against
 * some other pathological input).
 */
function detectOnsets(essentia, samples) {
  if (samples.length === 0) throw new Error("empty audio signal, cannot detect onsets");
  const vectorSignal = essentia.arrayToVector(samples);

  let onsets = [];
  try {
    const result = essentia.SuperFluxExtractor(
      vectorSignal, SUPERFLUX_COMBINE_MS, 2048, 256, SUPERFLUX_RATIO_THRESHOLD, ESSENTIA_SAMPLE_RATE, 0.05
    );
    if (result.onsets.size() > 0) {
      onsets = Array.from(essentia.vectorToArray(result.onsets));
    } else {
      console.warn("SuperFluxExtractor returned zero onsets, falling back to OnsetRate");
    }
    result.onsets.delete();
  } catch (err) {
    console.warn("SuperFluxExtractor failed, falling back to OnsetRate:", err);
  }

  if (onsets.length === 0) {
    const fallback = essentia.OnsetRate(vectorSignal);
    onsets = Array.from(essentia.vectorToArray(fallback.onsets));
    fallback.onsets.delete();
  }

  vectorSignal.delete();
  return onsets;
}

/**
 * Per-onset amplitude: RMS energy of a short window right after each onset.
 * Used as the dynamics signal.
 */
function amplitudeAtOnsets(samples, sampleRate, onsetTimes, windowSec = 0.08) {
  const windowSamples = Math.floor(windowSec * sampleRate);
  return onsetTimes.map((t) => {
    const start = Math.floor(t * sampleRate);
    const end = Math.min(start + windowSamples, samples.length);
    let sumSq = 0;
    for (let i = start; i < end; i++) sumSq += samples[i] * samples[i];
    const n = Math.max(end - start, 1);
    return Math.sqrt(sumSq / n);
  });
}

/**
 * Predominant pitch contour over time via Essentia's PitchMelodia, framed
 * into a time series. Reliable mainly when melody is the loudest/highest
 * voice; treated as medium-confidence, not ground truth, by design.
 */
function predominantPitchContour(essentia, samples, sampleRate) {
  const vectorSignal = essentia.arrayToVector(samples);
  // Positional params after signal are binResolution, filterIterations,
  // frameSize, guessUnvoiced, harmonicWeight, hopSize, ... -- leave all on
  // their documented defaults (sampleRate=44100 is already correct since
  // input is resampled to ESSENTIA_SAMPLE_RATE before this is called).
  const result = essentia.PredominantPitchMelodia(vectorSignal);
  const pitch = Array.from(essentia.vectorToArray(result.pitch));
  const hopSize = 128; // Essentia PitchMelodia default hopSize
  const times = pitch.map((_, i) => (i * hopSize) / sampleRate);
  vectorSignal.delete();
  result.pitch.delete();
  result.pitchConfidence.delete();
  return times.map((t, i) => ({ time: t, hz: pitch[i] }));
}

/**
 * Chroma vector (12-dim pitch-class energy) per onset-aligned window, for
 * DTW alignment against the reference template's chroma vectors.
 */
function chromaAtOnsets(essentia, samples, sampleRate, onsetTimes, windowSec = 0.2) {
  const windowSamples = Math.floor(windowSec * sampleRate);
  return onsetTimes.map((t) => {
    const start = Math.floor(t * sampleRate);
    let end = Math.min(start + windowSamples, samples.length);
    if ((end - start) % 2 !== 0) end -= 1; // Spectrum requires an even-length frame
    const frame = samples.slice(start, end);
    if (frame.length < 2) return new Array(12).fill(0);

    const vectorFrame = essentia.arrayToVector(frame);
    const spectrum = essentia.Spectrum(vectorFrame);
    const peaks = essentia.SpectralPeaks(spectrum.spectrum);
    const hpcp = essentia.HPCP(peaks.frequencies, peaks.magnitudes);
    const chroma = Array.from(essentia.vectorToArray(hpcp.hpcp));

    vectorFrame.delete();
    spectrum.spectrum.delete();
    peaks.frequencies.delete();
    peaks.magnitudes.delete();
    hpcp.hpcp.delete();

    // Essentia's HPCP defaults to 12 or 36 bins; fold down to 12 if needed.
    if (chroma.length === 12) return chroma;
    const folded = new Array(12).fill(0);
    const binsPerSemitone = chroma.length / 12;
    chroma.forEach((v, i) => { folded[Math.floor(i / binsPerSemitone)] += v; });
    return folded;
  });
}

const PITCH_SEARCH_WINDOW_SEC = 0.15; // how far after an onset to look for a voiced pitch-contour sample
const PITCH_CLASS_SIGMA_SEMITONES = 0.8; // tolerance for mistuning/pitch-detection error when building the alignment feature

/**
 * Estimated fundamental (Hz) near each onset, from the predominant-pitch
 * contour, or null where no voiced pitch was found nearby.
 */
function estimatePitchHzAtOnsets(pitchContour, onsetTimes, windowSec = PITCH_SEARCH_WINDOW_SEC) {
  return onsetTimes.map((t) => {
    const voiced = pitchContour.filter((p) => p.time >= t && p.time <= t + windowSec && p.hz > 0);
    if (voiced.length === 0) return null;
    const sorted = voiced.map((p) => p.hz).sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  });
}

/**
 * A 12-dim pitch-class vector peaked (circular Gaussian) around the pitch
 * class implied by a detected fundamental frequency -- same shape as the
 * reference template's one-hot chroma, so it's directly comparable by
 * cosine distance, but built from a sharper, melody-specific signal than
 * general HPCP chroma.
 */
function pitchClassVector(hz, sigmaSemitones = PITCH_CLASS_SIGMA_SEMITONES) {
  const midi = 69 + 12 * Math.log2(hz / 440);
  const pc = ((midi % 12) + 12) % 12;
  const vec = new Array(12).fill(0);
  for (let b = 0; b < 12; b++) {
    const d = Math.min(Math.abs(b - pc), 12 - Math.abs(b - pc));
    vec[b] = Math.exp(-(d * d) / (2 * sigmaSemitones * sigmaSemitones));
  }
  return vec;
}

const ONSET_MIN_AMPLITUDE_RATIO = 0.15; // relative to the take's median onset amplitude; drops spurious low-energy
// onset detections (mechanical clicks, pedal noise, breath) that would otherwise anchor or pollute DTW alignment.

function medianOf(arr) {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Drops onsets whose amplitude is far below the take's typical onset
 * amplitude -- these are almost always non-musical noise, not quiet notes,
 * and left in place they can become the DTW anchor or get silently absorbed
 * into a neighboring note's match, corrupting the whole alignment.
 */
function filterLowAmplitudeOnsets(onsets, amplitudes) {
  if (onsets.length === 0) return { onsets, amplitudes };
  const threshold = medianOf(amplitudes) * ONSET_MIN_AMPLITUDE_RATIO;
  const keptOnsets = [], keptAmplitudes = [];
  onsets.forEach((t, i) => {
    if (amplitudes[i] >= threshold) {
      keptOnsets.push(t);
      keptAmplitudes.push(amplitudes[i]);
    }
  });
  return { onsets: keptOnsets, amplitudes: keptAmplitudes };
}

async function analyzeSamples(samples, sampleRate) {
  const essentia = await getEssentia();

  const rawOnsets = detectOnsets(essentia, samples);
  const rawAmplitudes = amplitudeAtOnsets(samples, sampleRate, rawOnsets);
  const { onsets, amplitudes } = filterLowAmplitudeOnsets(rawOnsets, rawAmplitudes);

  // Temporary diagnostic: dump every detected onset (pre- and post-filter)
  // to the console so false-positive onsets can be pinpointed by timestamp
  // without needing to export and re-analyze the recording offline.
  console.log("[onset debug] all detected onsets (time, amplitude, kept?):");
  rawOnsets.forEach((t, i) => {
    const kept = onsets.includes(t);
    console.log(`  ${i}\t${t.toFixed(3)}s\tamp=${rawAmplitudes[i].toFixed(4)}\tkept=${kept}`);
  });
  const chroma = chromaAtOnsets(essentia, samples, sampleRate, onsets);
  const pitchContour = predominantPitchContour(essentia, samples, sampleRate);
  const onsetPitchHz = estimatePitchHzAtOnsets(pitchContour, onsets);

  // DTW alignment feature: pitch-class vector from the detected melody
  // pitch where available (sharper signal for this near-monophonic
  // excerpt), falling back to HPCP chroma where the pitch tracker found
  // nothing voiced nearby (e.g. a chord or a noisy attack).
  const alignmentFeatures = onsets.map((_, i) =>
    onsetPitchHz[i] ? pitchClassVector(onsetPitchHz[i]) : chroma[i]
  );

  return {
    durationSec: samples.length / sampleRate,
    onsets,
    amplitudes,
    chroma,
    pitchContour,
    onsetPitchHz,
    alignmentFeatures,
  };
}

async function analyzePerformance(file) {
  const { samples, sampleRate } = await decodeAudioFile(file);
  return analyzeSamples(samples, sampleRate);
}
