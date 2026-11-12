"""
Prototype: for reference notes that the final (thresholded) note lists call
"missing" or "recovered via Basic Pitch", go back to ByteDance's raw
per-frame probability curves (before any thresholding) at that pitch, in a
window around the expected time, and see what's actually there.

Two curves per (pitch, frame):
  reg_onset_output  -- the regression value thresholded against
                       onset_threshold (0.3) to decide "onset happened here".
                       This is what we already know is near-zero (~0.001-0.007)
                       for the known masked D4 case.
  frame_output      -- a separate "is this key down" signal, thresholded
                       against frame_threshold (0.1), used by the real
                       post-processor to extend/gate notes. Not yet checked
                       for the masked case -- this is the new thing we're
                       testing.
"""
import json
import sys

import librosa
import numpy as np
from piano_transcription_inference import PianoTranscription, sample_rate

AUDIO_PATH = "bach_english_suite_take_1.wav"
REFERENCE_PATH = "bach-3bars-reference.json"
BYTEDANCE_PATH = "bach-clip-bytedance.json"
BASICPITCH_PATH = "bach-clip-basicpitch.json"

FRAMES_PER_SECOND = 100
ONSET_THRESHOLD = 0.3
FRAME_THRESHOLD = 0.1
PROBE_WINDOW_SEC = 0.15  # look this far before/after the expected time

MAX_MERGE_GAP_SEC = 0.05
MAX_SUBSTANTIAL_NOTE_DUR_SEC = 0.15
TIME_WINDOW_SEC = 0.25


def merge_adjacent_same_pitch(notes):
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


def find_candidates(events, expected_time, pitch, unconsumed_only=False):
    out = []
    for n in events:
        if unconsumed_only and n["consumed"]:
            continue
        if n["pitch"] != pitch:
            continue
        dt = n["onset"] - expected_time
        if abs(dt) > TIME_WINDOW_SEC:
            continue
        out.append((abs(dt), n))
    out.sort(key=lambda x: x[0])
    return [n for _, n in out]


print("Loading reference + transcribed note lists...")
with open(REFERENCE_PATH) as f:
    reference = json.load(f)
with open(BYTEDANCE_PATH) as f:
    bytedance = json.load(f)
with open(BASICPITCH_PATH) as f:
    basicpitch = json.load(f)

basicpitch = merge_adjacent_same_pitch(basicpitch)
for n in bytedance:
    n["consumed"] = False
for n in basicpitch:
    n["consumed"] = False

ref_events = []
for n in reference["notes"]:
    for pitch in n["midi_pitches"]:
        ref_events.append({"onset": n["onset_sec"], "duration": n["duration_sec"], "pitch": pitch})
ref_events.sort(key=lambda e: e["onset"])

ref_start, ref_end = ref_events[0]["onset"], max(e["onset"] + e["duration"] for e in ref_events)
perf_start = min(n["onset"] for n in bytedance)
perf_end = max(n["offset"] for n in bytedance)
scale = (perf_end - perf_start) / (ref_end - ref_start)


def to_perf_time(ref_onset):
    return perf_start + (ref_onset - ref_start) * scale


interesting = []  # (ref, expected_t, verdict)
for ref in ref_events:
    expected_t = to_perf_time(ref["onset"])
    pitch = ref["pitch"]
    bd_exact = find_candidates(bytedance, expected_t, pitch, unconsumed_only=True)
    bp_exact = find_candidates(basicpitch, expected_t, pitch, unconsumed_only=True)
    if bd_exact:
        bd_exact[0]["consumed"] = True
        if bp_exact:
            bp_exact[0]["consumed"] = True
        continue  # matched directly, nothing to probe
    if bp_exact:
        bp_exact[0]["consumed"] = True
        interesting.append((ref, expected_t, "recovered via Basic Pitch"))
    else:
        interesting.append((ref, expected_t, "missing"))

print(f"{len(interesting)} reference notes not directly matched by ByteDance -- probing raw output for these.\n")

print("Running ByteDance transcription to get raw (pre-threshold) output arrays...")
audio, _ = librosa.load(path=AUDIO_PATH, sr=sample_rate, mono=True)
transcriptor = PianoTranscription(device="cpu", checkpoint_path=None)
result = transcriptor.transcribe(audio, "/dev/null")
reg_onset = result["output_dict"]["reg_onset_output"]  # (frames, 88)
frame = result["output_dict"]["frame_output"]  # (frames, 88)
print(f"Raw output shape: {reg_onset.shape} ({reg_onset.shape[0] / FRAMES_PER_SECOND:.2f}s)\n")


def probe(pitch_midi, center_t):
    col = pitch_midi - 21  # piano_transcription_inference's begin_note=21 (A0)
    lo_frame = max(0, int((center_t - PROBE_WINDOW_SEC) * FRAMES_PER_SECOND))
    hi_frame = min(reg_onset.shape[0], int((center_t + PROBE_WINDOW_SEC) * FRAMES_PER_SECOND))
    onset_slice = reg_onset[lo_frame:hi_frame, col]
    frame_slice = frame[lo_frame:hi_frame, col]
    onset_max_i = int(np.argmax(onset_slice))
    frame_max_i = int(np.argmax(frame_slice))
    return {
        "onset_max": float(onset_slice[onset_max_i]),
        "onset_max_t": (lo_frame + onset_max_i) / FRAMES_PER_SECOND,
        "frame_max": float(frame_slice[frame_max_i]),
        "frame_max_t": (lo_frame + frame_max_i) / FRAMES_PER_SECOND,
        "frame_mean": float(frame_slice.mean()),
    }


print(f"{'ref_t':>6} {'perf_t':>7} {'pitch':>5} {'verdict':<28} {'onset_max':>9} {'frame_max':>9} {'frame_mean':>10}  note")
for ref, expected_t, verdict in interesting:
    p = probe(ref["pitch"], expected_t)
    onset_hit = p["onset_max"] >= ONSET_THRESHOLD
    frame_hit = p["frame_max"] >= FRAME_THRESHOLD
    note = ""
    if not onset_hit and frame_hit:
        note = "<-- onset suppressed but frame signal present: recoverable via frame-gated search"
    elif not onset_hit and not frame_hit:
        note = "<-- no signal in either curve: genuinely not detected at the model level"
    elif onset_hit:
        note = "<-- onset_max already clears threshold?! (would not have been 'missing' -- check window/pitch)"
    print(f"{ref['onset']:6.3f} {expected_t:7.3f} {ref['pitch']:5d} {verdict:<28} "
          f"{p['onset_max']:9.4f} {p['frame_max']:9.4f} {p['frame_mean']:10.4f}  {note}")
