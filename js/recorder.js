// In-app mic capture that bypasses MediaRecorder entirely, so no lossy codec
// (Opus/AAC) ever sits between the mic and analysis -- unlike a phone's
// voice-memo app (typically AAC), which can smear transients ("pre-echo")
// right around sustained/resonant passages and introduce onsets that were
// never actually played.
//
// ScriptProcessorNode is deprecated in favor of AudioWorklet, but it needs
// no separate module file/registration step, which keeps this prototype
// simple; revisit if a browser actually drops support.

let recorderState = null;

// Phones/interfaces commonly record at 48000Hz or other native rates;
// the backend's transcription model and the rest of the pipeline assume
// one fixed rate, so every recording is resampled to this before it's
// WAV-encoded and submitted.
const RECORDING_SAMPLE_RATE = 44100;

async function resampleRecording(decodedBuffer) {
  const resampleCtx = new OfflineAudioContext(
    1, // mono output
    Math.ceil(decodedBuffer.duration * RECORDING_SAMPLE_RATE),
    RECORDING_SAMPLE_RATE
  );
  const source = resampleCtx.createBufferSource();
  source.buffer = decodedBuffer;
  source.connect(resampleCtx.destination);
  source.start();
  const resampled = await resampleCtx.startRendering();
  return resampled.getChannelData(0);
}

/**
 * Encodes Float32 PCM samples as a standard 16-bit PCM WAV Blob.
 *
 * Dev/testing use only -- the app's recorder is otherwise deliberately
 * volatile (audio never touches disk) to match the project's
 * process-and-discard privacy posture. This exists so we can save real
 * takes once and reuse them across multiple transcription experiments,
 * instead of re-recording every time; it is not meant to ship in the
 * actual recording flow users see.
 */
function encodeWav(samples, sampleRate) {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true); // byte rate
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bytesPerSample * 8, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

/**
 * Triggers a browser download of { samples, sampleRate } as a .wav file.
 * See encodeWav() for why this exists only for dev/testing.
 */
function downloadRecordingAsWav(samples, sampleRate, filename = "recording.wav") {
  const blob = encodeWav(samples, sampleRate);
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function startRecording() {
  // Explicitly off: these three are speech-call features (voice-level
  // normalization, voice/noise separation, call-echo removal) that are on
  // by default and actively mangle piano signal -- AGC re-boosts decaying
  // sustain-pedal tails as if they were quiet speech, and noise
  // suppression mistakes piano overtones/resonance for background noise.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(stream);
  const processor = audioCtx.createScriptProcessor(4096, 1, 1);

  const chunks = [];
  processor.onaudioprocess = (event) => {
    chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  };

  source.connect(processor);
  processor.connect(audioCtx.destination); // required by some browsers to keep the graph running; silent since we never write to output meaningfully

  recorderState = { stream, audioCtx, source, processor, chunks };
}

/**
 * Stops capture and returns { samples, sampleRate } resampled to
 * RECORDING_SAMPLE_RATE (44100), ready to WAV-encode and submit.
 */
async function stopRecording() {
  if (!recorderState) throw new Error("not recording");
  const { stream, audioCtx, source, processor, chunks } = recorderState;
  recorderState = null;

  processor.disconnect();
  source.disconnect();
  stream.getTracks().forEach((t) => t.stop());

  const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  const nativeRate = audioCtx.sampleRate;
  const buffer = audioCtx.createBuffer(1, merged.length, nativeRate);
  buffer.copyToChannel(merged, 0);
  await audioCtx.close();

  const samples = await resampleRecording(buffer);
  return { samples, sampleRate: RECORDING_SAMPLE_RATE };
}
