"""
Slice 2: runs both transcription models against a saved recording and
writes their outputs (MIDI + notes JSON) into the performance's own
folder. Model objects are lazily created once and cached at module level
-- loading the ByteDance checkpoint and the Basic Pitch model each took
several seconds in testing, and this server only ever serves one
performer at a time, so there's no concurrency reason to reload them
per request.

Ported from the experiments/transcription-smoke-test/ scripts
(transcribe.py's merge_adjacent_same_pitch, dump_bytedance_json.py,
dump_basicpitch_json.py) -- same filtering constants, same output shape,
now writing straight into a performance folder instead of loose files.
"""
import json
from pathlib import Path

import librosa

MIN_NOTE_DURATION_SEC = 0.03
MAX_MERGE_GAP_SEC = 0.05
MAX_SUBSTANTIAL_NOTE_DUR_SEC = 0.15

_bytedance_transcriptor = None
_basic_pitch_model_path = None


def merge_adjacent_same_pitch(notes):
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


def run_bytedance(audio_path: Path, out_dir: Path) -> list[dict]:
    global _bytedance_transcriptor
    from piano_transcription_inference import PianoTranscription, sample_rate
    from piano_transcription_inference.utilities import write_events_to_midi

    if _bytedance_transcriptor is None:
        _bytedance_transcriptor = PianoTranscription(device="cpu", checkpoint_path=None)

    audio, _ = librosa.load(path=str(audio_path), sr=sample_rate, mono=True)
    result = _bytedance_transcriptor.transcribe(audio, str(out_dir / "bytedance-unfiltered.mid"))

    notes = result["est_note_events"]
    pedal_events = result["est_pedal_events"]
    kept = [n for n in notes if (n["offset_time"] - n["onset_time"]) >= MIN_NOTE_DURATION_SEC]
    kept = merge_adjacent_same_pitch(kept)
    write_events_to_midi(0, kept, pedal_events, str(out_dir / "bytedance-filtered.mid"))

    out = [
        {"onset": float(n["onset_time"]), "offset": float(n["offset_time"]),
         "pitch": int(n["midi_note"]), "velocity": float(n["velocity"])}
        for n in kept
    ]
    (out_dir / "bytedance.json").write_text(json.dumps(out, indent=2))
    return out


def run_basic_pitch(audio_path: Path, out_dir: Path) -> list[dict]:
    global _basic_pitch_model_path
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    if _basic_pitch_model_path is None:
        _basic_pitch_model_path = ICASSP_2022_MODEL_PATH

    _, midi_data, note_events = predict(str(audio_path), _basic_pitch_model_path)
    midi_data.write(str(out_dir / "basicpitch.mid"))

    out = [
        {"onset": float(start), "offset": float(end), "pitch": int(pitch), "amplitude": float(amp)}
        for start, end, pitch, amp, _ in note_events
    ]
    (out_dir / "basicpitch.json").write_text(json.dumps(out, indent=2))
    return out
