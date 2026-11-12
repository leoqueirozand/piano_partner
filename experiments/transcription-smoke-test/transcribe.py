import sys
import time
import librosa
from piano_transcription_inference import PianoTranscription, sample_rate
from piano_transcription_inference.utilities import write_events_to_midi

MIN_NOTE_DURATION_SEC = 0.03

# Collapses a sustained note that the model split into two same-pitch onsets
# back-to-back (observed gap ~7ms) into a single note. Real repeated-pitch
# keystrokes in our test clips all had gaps >= 300ms, so 50ms separates the
# two cleanly with margin on both sides.
MAX_MERGE_GAP_SEC = 0.05

# A genuine split-onset artifact looks like one note followed by a short
# trailing fragment (e.g. 133ms + 54ms). Two substantial notes of the same
# pitch landing back-to-back with a tiny gap (observed: 735ms + 302ms, two
# real distinct reference notes, not an artifact) must NOT be merged -- that
# would silently erase real timing information. Require at least one side
# to look like a fragment before merging.
MAX_SUBSTANTIAL_NOTE_DUR_SEC = 0.15

audio_path = sys.argv[1]
out_prefix = sys.argv[2] if len(sys.argv) > 2 else "out"


def merge_adjacent_same_pitch(notes):
    # Compares against the last note of the *same pitch* seen so far, not
    # just the previous note in time overall -- a different pitch sounding
    # in between (e.g. the other hand) must not block a same-pitch merge.
    notes = sorted(notes, key=lambda n: n["onset_time"])
    last_by_pitch = {}
    merged = []
    for n in notes:
        prev = last_by_pitch.get(n["midi_note"])
        prev_dur = (prev["offset_time"] - prev["onset_time"]) if prev is not None else None
        n_dur = n["offset_time"] - n["onset_time"]
        looks_like_fragment = prev is not None and (
            prev_dur < MAX_SUBSTANTIAL_NOTE_DUR_SEC or n_dur < MAX_SUBSTANTIAL_NOTE_DUR_SEC
        )
        if prev is not None and looks_like_fragment and n["onset_time"] - prev["offset_time"] <= MAX_MERGE_GAP_SEC:
            prev["offset_time"] = n["offset_time"]
        else:
            new_note = dict(n)
            merged.append(new_note)
            last_by_pitch[n["midi_note"]] = new_note
    return merged

print(f"Loading audio: {audio_path}")
audio, _ = librosa.load(path=audio_path, sr=sample_rate, mono=True)
print(f"Loaded {len(audio) / sample_rate:.2f}s of audio at {sample_rate}Hz")

transcriptor = PianoTranscription(device="cpu", checkpoint_path=None)

print("Transcribing...")
t0 = time.time()
result = transcriptor.transcribe(audio, f"{out_prefix}-unfiltered.mid")
elapsed = time.time() - t0
print(f"Transcription took {elapsed:.1f}s for {len(audio) / sample_rate:.1f}s of audio")

notes = result["est_note_events"]
pedal_events = result["est_pedal_events"]
print(f"\nTotal note events (unfiltered): {len(notes)}")

kept = [n for n in notes if (n["offset_time"] - n["onset_time"]) >= MIN_NOTE_DURATION_SEC]
dropped = [n for n in notes if (n["offset_time"] - n["onset_time"]) < MIN_NOTE_DURATION_SEC]
print(f"Kept (duration >= {MIN_NOTE_DURATION_SEC}s): {len(kept)}")
print(f"Dropped (duration < {MIN_NOTE_DURATION_SEC}s): {len(dropped)}")

before_merge = len(kept)
kept = merge_adjacent_same_pitch(kept)
if len(kept) != before_merge:
    print(f"Merged same-pitch split onsets (gap <= {MAX_MERGE_GAP_SEC*1000:.0f}ms): {before_merge} -> {len(kept)}")

write_events_to_midi(0, kept, pedal_events, f"{out_prefix}-filtered.mid")
print(f"Wrote {out_prefix}-filtered.mid")

print("\nAll kept notes, sorted by time (onset, midi, duration_ms, velocity):")
for n in sorted(kept, key=lambda n: n["onset_time"]):
    dur = n["offset_time"] - n["onset_time"]
    print(f"  t={n['onset_time']:.3f}s  midi={n['midi_note']}  dur={dur*1000:.0f}ms  vel={n['velocity']}")
