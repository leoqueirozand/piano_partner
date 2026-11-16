"""
Runs the transcription model against a saved recording and writes its
outputs (MIDI + notes JSON + pedal JSON) into the performance's own
folder. The model is lazily created once and cached at module level --
loading a checkpoint takes several seconds, and this server only ever
serves one performer at a time, so there's no concurrency reason to
reload it per request.

Transkun V2 replaced the earlier ByteDance + Basic Pitch pair (see
experiments/transcription-smoke-test/README.md, "Experiment 3") after
it beat both of them by a large, consistent margin on every real-piano
recording tested, including eliminating a reproducible octave-confusion
bug neither older model avoided. Running two independent models was
originally meant to let each catch the other's mistakes, but a direct
cross-check (same README) showed ByteDance's disagreements with Transkun
were false alarms about 3.4x as often as genuine catches -- not worth
the extra model load. If a confidence signal is needed later, Transkun's
public API doesn't expose one (its CRF decoder returns discrete notes,
no per-note probability) and reintroducing a second-model pass would
need a different heuristic than the old "disagreement = dispute" one.
"""
import json
from pathlib import Path

import soundfile as sf

MIN_NOTE_DURATION_SEC = 0.03
MAX_MERGE_GAP_SEC = 0.05
MAX_SUBSTANTIAL_NOTE_DUR_SEC = 0.15

SUSTAIN_PEDAL_CC = 64

_transkun_model = None


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


def get_audio_duration(audio_path: Path) -> float:
    """Header-only read (no decode) -- used by evaluate.py's duration-
    sanity check, which needs the recording's real length to catch a
    model-reported note offset that falls after the audio actually
    ends (see evaluate.py's filter_implausible_notes)."""
    info = sf.info(str(audio_path))
    return info.frames / info.samplerate


def _load_transkun_model():
    """Loads Transkun's bundled default checkpoint once, cached at module
    level -- same reasoning as the old ByteDance/Basic Pitch caching this
    replaces (see module docstring)."""
    global _transkun_model
    if _transkun_model is not None:
        return _transkun_model

    import pkg_resources
    import torch
    import moduleconf
    import transkun

    weight_path = pkg_resources.resource_filename(transkun.__name__, "pretrained/2.0.pt")
    conf_path = pkg_resources.resource_filename(transkun.__name__, "pretrained/2.0.conf")
    conf_manager = moduleconf.parseFromFile(conf_path)
    TransKun = conf_manager["Model"].module.TransKun
    conf = conf_manager["Model"].config

    checkpoint = torch.load(weight_path, map_location="cpu")
    model = TransKun(conf=conf).to("cpu")
    state_key = "best_state_dict" if "best_state_dict" in checkpoint else "state_dict"
    model.load_state_dict(checkpoint[state_key], strict=False)
    model.eval()

    _transkun_model = model
    return model


def run_transkun(audio_path: Path, out_dir: Path) -> list[dict]:
    import torch
    from transkun.transcribe import readAudio
    from transkun.Data import writeMidi

    model = _load_transkun_model()
    torch.set_grad_enabled(False)

    fs, audio = readAudio(str(audio_path))
    if fs != model.fs:
        import soxr
        audio = soxr.resample(audio, fs, model.fs)
    x = torch.from_numpy(audio)

    notes_est = model.transcribe(x, discardSecondHalf=False)
    writeMidi(notes_est).write(str(out_dir / "transkun.mid"))

    # Transkun encodes a sustain-pedal (or other control-change) event as
    # a Note with pitch = -<CC number> instead of a real MIDI pitch (see
    # transkun/Data.py's parseControlChangeSwitch/writeMidi) -- split the
    # two kinds apart rather than filtering by duration/velocity.
    notes = [n for n in notes_est if n.pitch > 0]
    pedal_events = [n for n in notes_est if n.pitch == -SUSTAIN_PEDAL_CC]

    kept = [n for n in notes if (n.end - n.start) >= MIN_NOTE_DURATION_SEC]
    kept = merge_adjacent_same_pitch([
        {"onset_time": n.start, "offset_time": n.end, "midi_note": n.pitch, "velocity": n.velocity}
        for n in kept
    ])

    out = [
        {"onset": float(n["onset_time"]), "offset": float(n["offset_time"]),
         "pitch": int(n["midi_note"]), "velocity": float(n["velocity"])}
        for n in kept
    ]
    (out_dir / "transkun.json").write_text(json.dumps(out, indent=2))

    # Same sustain-pedal shape evaluate.py's noise classification already
    # expects from the old "bytedance-pedal.json" sibling file -- used to
    # tell a quiet decay/resonance bleed-through (landing while the pedal
    # is already down) from a genuinely new, deliberately-played note.
    pedal_out = [
        {"onset": float(p.start), "offset": float(p.end)}
        for p in pedal_events if p.end is not None
    ]
    (out_dir / "transkun-pedal.json").write_text(json.dumps(pedal_out, indent=2))
    return out
