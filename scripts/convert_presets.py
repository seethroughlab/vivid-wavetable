#!/usr/bin/env python3
"""Convert monolithic WavetableSynth factory presets to modular graph JSON files."""

import json
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(ROOT_DIR, "factory_presets", "wavetable_synth.json")
OUTPUT_DIR = os.path.join(ROOT_DIR, "graphs", "presets")

PKG = {"name": "vivid-wavetable"}


def slug(name: str) -> str:
    """Convert preset name like 'Pad/Warm Pad' to 'warm_pad'."""
    # Take the part after the slash (the actual name)
    if "/" in name:
        name = name.split("/", 1)[1]
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def get(params: dict, key: str, default=0.0):
    return params.get(key, default)


def convert_preset(preset: dict) -> dict:
    name = preset["name"]
    p = preset["params"]

    nodes = {}
    connections = []

    # --- Source: Clock + ChordProgression ---
    nodes["clock"] = {
        "type": "Clock",
        "params": {"bpm": 120.0},
    }
    nodes["chords"] = {
        "type": "ChordProgression",
        "params": {
            "key": 1, "mode": 0,
            "degree_0": 0, "degree_1": 2, "degree_2": 4, "degree_3": 5,
            "voicing_0": 1, "voicing_1": 1, "voicing_2": 1, "voicing_3": 1,
        },
    }
    connections.append({"from": "clock/beat_phase", "to": "chords/beat_phase"})

    # --- PolyVoiceAllocator ---
    nodes["voices"] = {
        "type": "PolyVoiceAllocator",
        "pkg": PKG,
        "params": {
            "max_voices": 8,
            "portamento": get(p, "portamento"),
        },
    }
    connections.extend([
        {"from": "chords/notes",      "to": "voices/notes_in"},
        {"from": "chords/velocities", "to": "voices/velocities_in"},
        {"from": "chords/gates",      "to": "voices/gates_in"},
    ])

    # --- WavetableOsc ---
    osc_params = {}
    for key in ["wavetable", "position", "amplitude", "warp_mode", "warp_amount",
                 "unison_voices", "unison_spread", "unison_stereo",
                 "unison_spread_mode", "detune"]:
        if key in p:
            osc_params[key] = p[key]

    nodes["osc"] = {
        "type": "WavetableOsc",
        "pkg": PKG,
        "params": osc_params,
    }
    connections.extend([
        {"from": "voices/frequencies", "to": "osc/frequencies"},
        {"from": "voices/gates",       "to": "osc/gates"},
        {"from": "voices/velocities",  "to": "osc/velocities"},
    ])

    # --- Filter (core) — auto-dups per voice in N-channel chain ---
    filter_params = {}
    for src, dst in [("filter_type", "mode"),
                      ("filter_cutoff", "cutoff"),
                      ("filter_resonance", "resonance"),
                      ("filter_keytrack", "keytrack"),
                      ("filter_drive", "drive")]:
        if src in p:
            filter_params[dst] = p[src]

    nodes["filter"] = {
        "type": "Filter",
        "params": filter_params,
    }
    connections.extend([
        {"from": "osc/output",          "to": "filter/input"},
        {"from": "voices/frequencies",   "to": "filter/frequencies"},
    ])

    # --- SpreadADSR: amp envelope (always present) ---
    nodes["amp_env"] = {
        "type": "SpreadADSR",
        "params": {
            "attack":  get(p, "attack",  0.01),
            "decay":   get(p, "decay",   0.1),
            "sustain": get(p, "sustain", 0.7),
            "release": get(p, "release", 0.3),
        },
    }
    connections.append({"from": "voices/gates", "to": "amp_env/gates"})

    # --- SpreadADSR: filter envelope (if env_amount != 0) ---
    f_env_amt = get(p, "filter_env_amount")
    if f_env_amt != 0:
        nodes["filt_env"] = {
            "type": "SpreadADSR",
            "params": {
                "attack":  get(p, "filter_attack",  0.01),
                "decay":   get(p, "filter_decay",   0.3),
                "sustain": get(p, "filter_sustain",  0.0),
                "release": get(p, "filter_release",  0.3),
            },
        }
        connections.extend([
            {"from": "voices/gates",       "to": "filt_env/gates"},
            {"from": "filt_env/envelopes", "to": "filter/cutoff_mod"},
        ])

    # --- SpreadADSR: position envelope ---
    p_env_amt = get(p, "position_env_amount")
    if p_env_amt != 0:
        nodes["pos_env"] = {
            "type": "SpreadADSR",
            "params": {
                "attack":  get(p, "position_attack",  0.01),
                "decay":   get(p, "position_decay",   0.3),
                "sustain": get(p, "position_sustain",  0.0),
                "release": get(p, "position_release",  0.3),
            },
        }
        connections.extend([
            {"from": "voices/gates",      "to": "pos_env/gates"},
            {"from": "pos_env/envelopes", "to": "osc/position_mod"},
        ])

    # --- SpreadLFOs ---
    lfo_defs = [
        ("pitch_mod",  "pitch_mod_rate",  "pitch_mod_amount",  "pitch_mod_waveform",  "osc/pitch_mod"),
        ("pos_mod",    "wt_pos_mod_rate", "wt_pos_mod_amount", "wt_pos_mod_waveform", "osc/position_mod"),
        ("filter_mod", "filter_mod_rate", "filter_mod_amount", "filter_mod_waveform", "filter/cutoff_mod"),
        ("warp_mod",   "warp_mod_rate",   "warp_mod_amount",   "warp_mod_waveform",   "osc/warp_mod"),
        ("pan_mod",    "pan_mod_rate",    "pan_mod_amount",    "pan_mod_waveform",    "mixer/pan_mod"),
    ]
    for node_id, rate_key, amt_key, wf_key, target_port in lfo_defs:
        amt = get(p, amt_key)
        if amt > 0:
            nodes[node_id] = {
                "type": "SpreadLFO",
                "params": {
                    "frequency": get(p, rate_key),
                    "amplitude": amt,
                    "waveform":  int(get(p, wf_key)),
                },
            }
            connections.extend([
                {"from": "voices/gates",          "to": f"{node_id}/gates"},
                {"from": f"{node_id}/values",     "to": target_port},
            ])

    # --- SubOsc (routed through its own VoiceMixer, bypassing filter) ---
    has_sub = get(p, "sub_level") > 0
    if has_sub:
        nodes["sub"] = {
            "type": "SubOsc",
            "pkg": PKG,
            "params": {
                "level":    p["sub_level"],
                "octave":   int(get(p, "sub_octave")),
                "waveform": int(get(p, "sub_waveform")),
            },
        }
        nodes["sub_mixer"] = {
            "type": "VoiceMixer",
            "pkg": PKG,
            "params": {"stereo_spread": 0.0},
        }
        connections.extend([
            {"from": "voices/frequencies", "to": "sub/frequencies"},
            {"from": "voices/gates",       "to": "sub/gates"},
            {"from": "sub/output",         "to": "sub_mixer/input"},
            {"from": "amp_env/envelopes",  "to": "sub_mixer/amp_env"},
        ])

    # --- Noise (core) — auto-dups per voice, bypasses filter ---
    has_noise = get(p, "noise_level") > 0
    if has_noise:
        # Core Noise: mode 0=White, 1=Pink, 2=Brown, 3=Blue, 4=Violet
        nodes["noise"] = {
            "type": "Noise",
            "params": {
                "amplitude": p["noise_level"],
                "color":     int(get(p, "noise_type")),
            },
        }
        nodes["noise_gain"] = {
            "type": "Gain",
            "params": {"gain": 1.0},
        }
        connections.extend([
            {"from": "noise/output",       "to": "noise_gain/input"},
        ])

    # --- VoiceMixer ---
    mixer_params = {
        "stereo_spread": get(p, "stereo_spread", 0.5),
    }
    if "vel_to_volume" in p:
        mixer_params["vel_to_volume"] = p["vel_to_volume"]

    nodes["mixer"] = {
        "type": "VoiceMixer",
        "pkg": PKG,
        "params": mixer_params,
    }
    # Main audio path: filter → mixer
    connections.extend([
        {"from": "filter/output",      "to": "mixer/input"},
        {"from": "amp_env/envelopes",  "to": "mixer/amp_env"},
        {"from": "voices/velocities",  "to": "mixer/velocities"},
    ])

    # --- Output chain ---
    # If sub/noise present, sum their stereo outputs with main via Mixer
    if has_sub or has_noise:
        nodes["sum"] = {
            "type": "Mixer",
            "params": {"gain_1": 1.0, "gain_2": 1.0, "gain_3": 1.0},
        }
        connections.append({"from": "mixer/output", "to": "sum/input_1"})
        if has_sub:
            connections.append({"from": "sub_mixer/output", "to": "sum/input_2"})
        if has_noise:
            input_n = "input_3" if has_sub else "input_2"
            connections.append({"from": "noise_gain/output", "to": f"sum/{input_n}"})
        sum_out = "sum/output"
    else:
        sum_out = "mixer/output"

    nodes["reverb"] = {
        "type": "Reverb",
        "params": {"room_size": 0.5, "mix": 0.2},
    }
    nodes["out"] = {"type": "audio_out"}
    connections.extend([
        {"from": sum_out,         "to": "reverb/input"},
        {"from": "reverb/output", "to": "out/input"},
    ])

    return {"nodes": nodes, "connections": connections}


def main():
    with open(INPUT_PATH) as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for preset in data["presets"]:
        graph = convert_preset(preset)
        filename = slug(preset["name"]) + ".json"
        out_path = os.path.join(OUTPUT_DIR, filename)
        with open(out_path, "w") as f:
            json.dump(graph, f, indent=2)
        print(f"  {preset['name']:30s} → {filename}")

    print(f"\nConverted {len(data['presets'])} presets to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
