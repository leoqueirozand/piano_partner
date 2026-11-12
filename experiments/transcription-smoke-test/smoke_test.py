import time
import librosa
from piano_transcription_inference import PianoTranscription, sample_rate
from piano_transcription_inference.utilities import write_events_to_midi

MIN_NOTE_DURATION_SEC = 0.03  # notes shorter than this are almost certainly artifacts, not real piano notes

audio_path = "test-clip.wav"
print(f"Loading audio: {audio_path}")
audio, _ = librosa.load(path=audio_path, sr=sample_rate, mono=True)
print(f"Loaded {len(audio) / sample_rate:.2f}s of audio at {sample_rate}Hz")

print("Loading model (cpu)...")
t0 = time.time()
transcriptor = PianoTranscription(device="cpu", checkpoint_path=None)
print(f"Model loaded in {time.time() - t0:.1f}s")

print("Transcribing...")
t0 = time.time()
result = transcriptor.transcribe(audio, "test-clip-unfiltered.mid")
elapsed = time.time() - t0
print(f"Transcription took {elapsed:.1f}s for {len(audio) / sample_rate:.1f}s of audio")

notes = result["est_note_events"]
pedal_events = result["est_pedal_events"]
print(f"\nTotal note events (unfiltered): {len(notes)}")

kept = [n for n in notes if (n["offset_time"] - n["onset_time"]) >= MIN_NOTE_DURATION_SEC]
dropped = [n for n in notes if (n["offset_time"] - n["onset_time"]) < MIN_NOTE_DURATION_SEC]

print(f"Kept (duration >= {MIN_NOTE_DURATION_SEC}s): {len(kept)}")
print(f"Dropped (duration < {MIN_NOTE_DURATION_SEC}s): {len(dropped)}")

print("\nDropped notes (pitch, duration, velocity):")
for n in sorted(dropped, key=lambda n: n["onset_time"]):
    dur = n["offset_time"] - n["onset_time"]
    print(f"  t={n['onset_time']:.3f}s  midi={n['midi_note']}  dur={dur*1000:.0f}ms  vel={n['velocity']}")

write_events_to_midi(0, kept, pedal_events, "test-clip-filtered.mid")
print("\nWrote test-clip-filtered.mid")
