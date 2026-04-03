#!/usr/bin/env python3
"""Generate rich multi-oscillator graph presets for the modular wavetable system."""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "graphs", "presets")
PKG = {"name": "vivid-wavetable"}


def source_nodes():
    """Standard note source: Clock → ChordProgression → PolyVoiceAllocator."""
    nodes = {
        "clock": {"type": "Clock", "params": {"bpm": 120.0}},
        "chords": {
            "type": "ChordProgression",
            "params": {
                "key": 1, "mode": 0,
                "degree_0": 0, "degree_1": 2, "degree_2": 4, "degree_3": 5,
                "voicing_0": 1, "voicing_1": 1, "voicing_2": 1, "voicing_3": 1,
            },
        },
        "voices": {
            "type": "PolyVoiceAllocator", "pkg": PKG,
            "params": {"max_voices": 8, "portamento": 0.0},
        },
    }
    conns = [
        ("clock/beat_phase", "chords/beat_phase"),
        ("chords/notes", "voices/notes_in"),
        ("chords/velocities", "voices/velocities_in"),
        ("chords/gates", "voices/gates_in"),
    ]
    return nodes, conns


def voice_wires(target, freq=True, gates=True, vel=True):
    """Wire PolyVoiceAllocator spreads to a target node."""
    c = []
    if freq:  c.append(("voices/frequencies", f"{target}/frequencies"))
    if gates: c.append(("voices/gates", f"{target}/gates"))
    if vel:   c.append(("voices/velocities", f"{target}/velocities"))
    return c


def amp_env(attack=0.01, decay=0.3, sustain=0.7, release=0.5):
    return {
        "type": "SpreadADSR",
        "params": {"attack": attack, "decay": decay, "sustain": sustain, "release": release},
    }


def filt_env(attack=0.01, decay=0.5, sustain=0.2, release=0.4):
    return {
        "type": "SpreadADSR",
        "params": {"attack": attack, "decay": decay, "sustain": sustain, "release": release},
    }


def conn(from_port, to_port):
    return (from_port, to_port)


def build_graph(nodes, connections):
    return {
        "nodes": nodes,
        "connections": [{"from": f, "to": t} for f, t in connections],
    }


# =============================================================================
# Preset definitions
# =============================================================================

def warm_dual_pad():
    n, c = source_nodes()
    n["amp_env"] = amp_env(0.8, 0.5, 0.7, 1.5)
    n["filt_env"] = filt_env(1.0, 0.8, 0.3, 1.5)
    c.append(("voices/gates", "amp_env/gates"))
    c.append(("voices/gates", "filt_env/gates"))

    n["osc_a"] = {"type": "WavetableOsc", "pkg": PKG,
                   "params": {"wavetable": 3, "position": 0.3, "amplitude": 0.25,
                              "unison_voices": 4, "unison_spread": 30.0, "unison_stereo": 0.8}}
    n["osc_b"] = {"type": "AnalogOsc", "pkg": PKG,
                   "params": {"waveform": 1, "amplitude": 0.2, "detune": 8.0}}
    c += voice_wires("osc_a") + voice_wires("osc_b")

    n["filt_a"] = {"type": "Filter",
                    "params": {"mode": 1, "cutoff": 3000.0, "resonance": 0.15}}
    n["filt_b"] = {"type": "Filter",
                    "params": {"mode": 0, "cutoff": 5000.0, "resonance": 0.1}}
    c += [("osc_a/output", "filt_a/input"), ("osc_b/output", "filt_b/input"),
          ("voices/frequencies", "filt_a/frequencies"),
          ("voices/frequencies", "filt_b/frequencies"),
          ("filt_env/envelopes", "filt_a/cutoff_mod"), ("filt_env/envelopes", "filt_b/cutoff_mod")]

    # Per-voice chorus (auto-dups to N instances, each voice gets own modulation)
    n["chorus_a"] = {"type": "Chorus", "params": {"rate": 0.3, "depth": 0.4, "voices": 3, "mix": 0.3}}
    n["chorus_b"] = {"type": "Chorus", "params": {"rate": 0.35, "depth": 0.35, "voices": 3, "mix": 0.3}}
    c += [("filt_a/output", "chorus_a/input"), ("filt_b/output", "chorus_b/input")]

    n["mix_a"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.7}}
    n["mix_b"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.7}}
    c += [("chorus_a/output", "mix_a/input"), ("chorus_b/output", "mix_b/input"),
          ("amp_env/envelopes", "mix_a/amp_env"), ("amp_env/envelopes", "mix_b/amp_env"),
          ("voices/velocities", "mix_a/velocities"), ("voices/velocities", "mix_b/velocities")]

    n["sum"] = {"type": "Mixer", "params": {"gain_1": 1.0, "gain_2": 1.0}}
    n["reverb"] = {"type": "Reverb", "params": {"room_size": 0.75, "mix": 0.35}}
    n["out"] = {"type": "audio_out"}
    c += [("mix_a/output", "sum/input_1"), ("mix_b/output", "sum/input_2"),
          ("sum/output", "reverb/input"), ("reverb/output", "out/input")]
    return build_graph(n, c)


def supersaw_stack():
    n, c = source_nodes()
    n["amp_env"] = amp_env(0.01, 0.1, 0.8, 0.3)
    c.append(("voices/gates", "amp_env/gates"))

    n["osc_a"] = {"type": "WavetableOsc", "pkg": PKG,
                   "params": {"wavetable": 1, "position": 0.4, "amplitude": 0.25,
                              "unison_voices": 8, "unison_spread": 40.0, "unison_stereo": 1.0,
                              "unison_spread_mode": 1, "detune": 15.0}}
    n["osc_b"] = {"type": "AnalogOsc", "pkg": PKG,
                   "params": {"waveform": 1, "amplitude": 0.2, "detune": 20.0}}
    c += voice_wires("osc_a") + voice_wires("osc_b")

    n["filter"] = {"type": "Filter",
                    "params": {"mode": 1, "cutoff": 6000.0, "resonance": 0.2}}
    # Both oscs into same filter - use Mixer to combine first (stereo post-mix)
    n["mix_a"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.9}}
    n["mix_b"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.5}}
    c += [("osc_a/output", "filter/input"), ("voices/frequencies", "filter/frequencies"),
          ("filter/output", "mix_a/input"), ("amp_env/envelopes", "mix_a/amp_env"),
          ("osc_b/output", "mix_b/input"), ("amp_env/envelopes", "mix_b/amp_env")]

    n["sum"] = {"type": "Mixer", "params": {"gain_1": 1.0, "gain_2": 0.7}}
    n["chorus"] = {"type": "Chorus", "params": {"rate": 0.5, "depth": 0.3, "mix": 0.25}}
    n["reverb"] = {"type": "Reverb", "params": {"room_size": 0.5, "mix": 0.2}}
    n["out"] = {"type": "audio_out"}
    c += [("mix_a/output", "sum/input_1"), ("mix_b/output", "sum/input_2"),
          ("sum/output", "chorus/input"), ("chorus/output", "reverb/input"),
          ("reverb/output", "out/input")]
    return build_graph(n, c)


def digital_strings():
    n, c = source_nodes()
    n["amp_env"] = amp_env(0.5, 0.5, 0.6, 1.0)
    c.append(("voices/gates", "amp_env/gates"))

    n["osc_a"] = {"type": "WavetableOsc", "pkg": PKG,
                   "params": {"wavetable": 7, "position": 0.4, "amplitude": 0.2,
                              "unison_voices": 4, "unison_spread": 25.0}}
    n["osc_b"] = {"type": "WavetableOsc", "pkg": PKG,
                   "params": {"wavetable": 4, "position": 0.6, "amplitude": 0.15,
                              "unison_voices": 2, "unison_spread": 15.0}}
    c += voice_wires("osc_a") + voice_wires("osc_b")

    n["filter"] = {"type": "Filter",
                    "params": {"mode": 0, "cutoff": 4000.0, "resonance": 0.1}}
    n["mixer"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.8}}
    n["mix_b"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.6}}
    c += [("osc_a/output", "filter/input"), ("voices/frequencies", "filter/frequencies"),
          ("filter/output", "mixer/input"), ("amp_env/envelopes", "mixer/amp_env"),
          ("osc_b/output", "mix_b/input"), ("amp_env/envelopes", "mix_b/amp_env")]

    n["sum"] = {"type": "Mixer", "params": {"gain_1": 1.0, "gain_2": 0.8}}
    n["phaser"] = {"type": "Phaser", "params": {"rate": 0.2, "depth": 0.5, "feedback": 0.3, "mix": 0.3}}
    n["reverb"] = {"type": "Reverb", "params": {"room_size": 0.8, "mix": 0.4}}
    n["out"] = {"type": "audio_out"}
    c += [("mixer/output", "sum/input_1"), ("mix_b/output", "sum/input_2"),
          ("sum/output", "phaser/input"), ("phaser/output", "reverb/input"),
          ("reverb/output", "out/input")]
    return build_graph(n, c)


def hybrid_bass():
    n, c = source_nodes()
    n["amp_env"] = amp_env(0.005, 0.2, 0.9, 0.15)
    n["filt_env"] = filt_env(0.005, 0.3, 0.3, 0.15)
    c += [("voices/gates", "amp_env/gates"), ("voices/gates", "filt_env/gates")]

    n["osc"] = {"type": "AnalogOsc", "pkg": PKG,
                 "params": {"waveform": 2, "amplitude": 0.35}}
    n["sub"] = {"type": "SubOsc", "pkg": PKG,
                 "params": {"level": 0.6, "octave": 0, "waveform": 3}}
    c += voice_wires("osc") + [("voices/frequencies", "sub/frequencies"),
                                 ("voices/gates", "sub/gates")]

    n["filter"] = {"type": "Filter",
                    "params": {"mode": 6, "cutoff": 800.0,
                               "resonance": 0.35, "drive": 0.2}}
    c += [("osc/output", "filter/input"), ("voices/frequencies", "filter/frequencies"),
          ("filt_env/envelopes", "filter/cutoff_mod")]

    # Per-voice distortion (auto-dups, each voice gets its own saturation)
    n["dist"] = {"type": "Distortion", "params": {"drive": 2.0, "tone": 0.4, "level": 0.8, "mix": 0.5}}
    c.append(("filter/output", "dist/input"))

    n["mixer"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.0}}
    n["sub_mix"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.0}}
    c += [("dist/output", "mixer/input"), ("amp_env/envelopes", "mixer/amp_env"),
          ("sub/output", "sub_mix/input"), ("amp_env/envelopes", "sub_mix/amp_env")]

    n["sum"] = {"type": "Mixer", "params": {"gain_1": 1.0, "gain_2": 1.0}}
    n["comp"] = {"type": "Compressor", "params": {"threshold": -15.0, "ratio": 4.0, "attack": 5.0, "release": 50.0}}
    n["out"] = {"type": "audio_out"}
    c += [("mixer/output", "sum/input_1"), ("sub_mix/output", "sum/input_2"),
          ("sum/output", "comp/input"), ("comp/output", "out/input")]
    return build_graph(n, c)


def fm_piano():
    n, c = source_nodes()
    n["amp_env"] = amp_env(0.005, 0.8, 0.3, 0.6)
    n["filt_env"] = filt_env(0.005, 0.6, 0.1, 0.4)
    c += [("voices/gates", "amp_env/gates"), ("voices/gates", "filt_env/gates")]

    n["modulator"] = {"type": "AnalogOsc", "pkg": PKG,
                       "params": {"waveform": 0, "amplitude": 0.5}}
    n["carrier"] = {"type": "WavetableOsc", "pkg": PKG,
                     "params": {"wavetable": 0, "position": 0.0, "amplitude": 0.3,
                                "interaction_mode": 1, "interaction_depth": 0.4,
                                "interaction_input_gain": 1.0, "interaction_tracking": 1.0}}
    c += voice_wires("modulator") + voice_wires("carrier")
    c.append(("modulator/output", "carrier/mod_input"))

    n["filter"] = {"type": "Filter",
                    "params": {"mode": 1, "cutoff": 4000.0,
                               "resonance": 0.1, "keytrack": 0.5}}
    c += [("carrier/output", "filter/input"), ("voices/frequencies", "filter/frequencies"),
          ("filt_env/envelopes", "filter/cutoff_mod")]

    n["mixer"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.3, "vel_to_volume": 1.0}}
    c += [("filter/output", "mixer/input"), ("amp_env/envelopes", "mixer/amp_env"),
          ("voices/velocities", "mixer/velocities")]

    n["reverb"] = {"type": "Reverb", "params": {"room_size": 0.6, "mix": 0.25}}
    n["out"] = {"type": "audio_out"}
    c += [("mixer/output", "reverb/input"), ("reverb/output", "out/input")]
    return build_graph(n, c)


def fm_metallic_mod():
    n, c = source_nodes()
    n["amp_env"] = amp_env(0.001, 0.4, 0.0, 0.3)
    c.append(("voices/gates", "amp_env/gates"))

    n["modulator"] = {"type": "AnalogOsc", "pkg": PKG,
                       "params": {"waveform": 1, "amplitude": 0.6}}
    n["carrier"] = {"type": "WavetableOsc", "pkg": PKG,
                     "params": {"wavetable": 8, "position": 0.3, "amplitude": 0.3,
                                "interaction_mode": 1, "interaction_depth": 0.7,
                                "interaction_input_gain": 1.0, "interaction_tracking": 1.0}}
    c += voice_wires("modulator") + voice_wires("carrier")
    c.append(("modulator/output", "carrier/mod_input"))

    n["filter"] = {"type": "Filter",
                    "params": {"mode": 5, "cutoff": 3000.0,
                               "resonance": 0.5, "keytrack": 1.0}}
    c += [("carrier/output", "filter/input"), ("voices/frequencies", "filter/frequencies"),
          ]

    n["mixer"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.5}}
    c += [("filter/output", "mixer/input"), ("amp_env/envelopes", "mixer/amp_env")]

    n["delay"] = {"type": "Delay", "params": {"time": 300.0, "feedback": 0.35, "mix": 0.3}}
    n["reverb"] = {"type": "Reverb", "params": {"room_size": 0.6, "mix": 0.3}}
    n["out"] = {"type": "audio_out"}
    c += [("mixer/output", "delay/input"), ("delay/output", "reverb/input"),
          ("reverb/output", "out/input")]
    return build_graph(n, c)


def fm_bell():
    n, c = source_nodes()
    n["amp_env"] = amp_env(0.001, 1.5, 0.0, 1.0)
    c.append(("voices/gates", "amp_env/gates"))

    n["modulator"] = {"type": "AnalogOsc", "pkg": PKG,
                       "params": {"waveform": 0, "amplitude": 0.5}}
    n["carrier"] = {"type": "WavetableOsc", "pkg": PKG,
                     "params": {"wavetable": 2, "position": 0.2, "amplitude": 0.25,
                                "interaction_mode": 1, "interaction_depth": 0.5,
                                "interaction_input_gain": 1.0, "interaction_tracking": 1.0}}
    c += voice_wires("modulator") + voice_wires("carrier")
    c.append(("modulator/output", "carrier/mod_input"))

    n["filter"] = {"type": "Filter",
                    "params": {"mode": 0, "cutoff": 8000.0, "resonance": 0.05}}
    c += [("carrier/output", "filter/input"), ("voices/frequencies", "filter/frequencies"),
          ]

    n["mixer"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.4}}
    c += [("filter/output", "mixer/input"), ("amp_env/envelopes", "mixer/amp_env")]

    n["ppd"] = {"type": "PingPongDelay", "params": {"time": 375.0, "feedback": 0.4, "spread": 1.0, "mix": 0.35}}
    n["reverb"] = {"type": "Reverb", "params": {"room_size": 0.8, "mix": 0.4}}
    n["out"] = {"type": "audio_out"}
    c += [("mixer/output", "ppd/input"), ("ppd/output", "reverb/input"),
          ("reverb/output", "out/input")]
    return build_graph(n, c)


def fm_bass():
    n, c = source_nodes()
    n["amp_env"] = amp_env(0.005, 0.2, 0.8, 0.15)
    n["filt_env"] = filt_env(0.005, 0.3, 0.2, 0.15)
    c += [("voices/gates", "amp_env/gates"), ("voices/gates", "filt_env/gates")]

    n["modulator"] = {"type": "AnalogOsc", "pkg": PKG,
                       "params": {"waveform": 2, "amplitude": 0.4}}
    n["carrier"] = {"type": "AnalogOsc", "pkg": PKG,
                     "params": {"waveform": 1, "amplitude": 0.35,
                                "interaction_mode": 1, "interaction_depth": 0.3,
                                "interaction_input_gain": 1.0, "interaction_tracking": 1.0}}
    c += voice_wires("modulator") + voice_wires("carrier")
    c.append(("modulator/output", "carrier/mod_input"))

    n["filter"] = {"type": "Filter",
                    "params": {"mode": 6, "cutoff": 1000.0,
                               "resonance": 0.4, "drive": 0.3}}
    c += [("carrier/output", "filter/input"), ("voices/frequencies", "filter/frequencies"),
          ("filt_env/envelopes", "filter/cutoff_mod")]

    # Per-voice distortion
    n["dist"] = {"type": "Distortion", "params": {"drive": 2.5, "tone": 0.3, "level": 0.8, "mix": 0.6}}
    c.append(("filter/output", "dist/input"))

    n["mixer"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.0}}
    c += [("dist/output", "mixer/input"), ("amp_env/envelopes", "mixer/amp_env")]

    n["out"] = {"type": "audio_out"}
    c.append(("mixer/output", "out/input"))
    return build_graph(n, c)


def rm_clang():
    n, c = source_nodes()
    n["amp_env"] = amp_env(0.001, 0.6, 0.0, 0.4)
    c.append(("voices/gates", "amp_env/gates"))

    n["modulator"] = {"type": "AnalogOsc", "pkg": PKG,
                       "params": {"waveform": 1, "amplitude": 0.5}}
    n["carrier"] = {"type": "WavetableOsc", "pkg": PKG,
                     "params": {"wavetable": 8, "position": 0.5, "amplitude": 0.3,
                                "interaction_mode": 3, "interaction_depth": 1.0,
                                "interaction_input_gain": 1.0, "interaction_tracking": 1.0}}
    c += voice_wires("modulator") + voice_wires("carrier")
    c.append(("modulator/output", "carrier/mod_input"))

    n["filter"] = {"type": "Filter",
                    "params": {"mode": 3, "cutoff": 2000.0, "resonance": 0.4}}
    c += [("carrier/output", "filter/input"), ("voices/frequencies", "filter/frequencies"),
          ]

    n["mixer"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.6}}
    c += [("filter/output", "mixer/input"), ("amp_env/envelopes", "mixer/amp_env")]

    n["reverb"] = {"type": "Reverb", "params": {"room_size": 0.7, "mix": 0.4}}
    n["out"] = {"type": "audio_out"}
    c += [("mixer/output", "reverb/input"), ("reverb/output", "out/input")]
    return build_graph(n, c)


def am_tremolo_pad():
    n, c = source_nodes()
    n["amp_env"] = amp_env(0.8, 0.5, 0.7, 1.5)
    c.append(("voices/gates", "amp_env/gates"))

    n["modulator"] = {"type": "AnalogOsc", "pkg": PKG,
                       "params": {"waveform": 0, "amplitude": 0.3}}
    n["carrier"] = {"type": "WavetableOsc", "pkg": PKG,
                     "params": {"wavetable": 3, "position": 0.4, "amplitude": 0.25,
                                "unison_voices": 4, "unison_spread": 25.0,
                                "interaction_mode": 4, "interaction_depth": 0.6,
                                "interaction_input_gain": 1.0, "interaction_tracking": 1.0}}
    c += voice_wires("modulator") + voice_wires("carrier")
    c.append(("modulator/output", "carrier/mod_input"))

    n["filter"] = {"type": "Filter",
                    "params": {"mode": 1, "cutoff": 3000.0, "resonance": 0.15}}
    c += [("carrier/output", "filter/input"), ("voices/frequencies", "filter/frequencies"),
          ]

    # Per-voice chorus
    n["chorus"] = {"type": "Chorus", "params": {"rate": 0.4, "depth": 0.3, "mix": 0.25}}
    c.append(("filter/output", "chorus/input"))

    n["mixer"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.8}}
    c += [("chorus/output", "mixer/input"), ("amp_env/envelopes", "mixer/amp_env")]

    n["reverb"] = {"type": "Reverb", "params": {"room_size": 0.8, "mix": 0.4}}
    n["out"] = {"type": "audio_out"}
    c += [("mixer/output", "reverb/input"), ("reverb/output", "out/input")]
    return build_graph(n, c)


def bitcrush_lead():
    n, c = source_nodes()
    n["amp_env"] = amp_env(0.01, 0.1, 0.8, 0.25)
    n["filt_env"] = filt_env(0.01, 0.3, 0.3, 0.25)
    c += [("voices/gates", "amp_env/gates"), ("voices/gates", "filt_env/gates")]

    n["osc"] = {"type": "AnalogOsc", "pkg": PKG,
                 "params": {"waveform": 1, "amplitude": 0.3, "detune": 12.0}}
    c += voice_wires("osc")

    n["filter"] = {"type": "Filter",
                    "params": {"mode": 1, "cutoff": 4000.0, "resonance": 0.2}}
    c += [("osc/output", "filter/input"), ("voices/frequencies", "filter/frequencies"),
          ("filt_env/envelopes", "filter/cutoff_mod")]

    # Per-voice bitcrush
    n["crush"] = {"type": "Bitcrush", "params": {"bits": 6, "rate": 12000.0, "mix": 0.7}}
    c.append(("filter/output", "crush/input"))

    n["mixer"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.5}}
    c += [("crush/output", "mixer/input"), ("amp_env/envelopes", "mixer/amp_env")]

    n["delay"] = {"type": "Delay", "params": {"time": 200.0, "feedback": 0.3, "mix": 0.25}}
    n["reverb"] = {"type": "Reverb", "params": {"room_size": 0.4, "mix": 0.2}}
    n["out"] = {"type": "audio_out"}
    c += [("mixer/output", "delay/input"), ("delay/output", "reverb/input"),
          ("reverb/output", "out/input")]
    return build_graph(n, c)


def phaser_pad():
    n, c = source_nodes()
    n["amp_env"] = amp_env(1.0, 0.6, 0.7, 2.0)
    c.append(("voices/gates", "amp_env/gates"))

    n["osc"] = {"type": "WavetableOsc", "pkg": PKG,
                 "params": {"wavetable": 6, "position": 0.5, "amplitude": 0.25,
                            "unison_voices": 6, "unison_spread": 30.0, "unison_stereo": 0.9}}
    c += voice_wires("osc")

    n["filter"] = {"type": "Filter",
                    "params": {"mode": 0, "cutoff": 5000.0, "resonance": 0.1}}
    c += [("osc/output", "filter/input"), ("voices/frequencies", "filter/frequencies"),
          ]

    n["mixer"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.8}}
    c += [("filter/output", "mixer/input"), ("amp_env/envelopes", "mixer/amp_env")]

    n["phaser"] = {"type": "Phaser", "params": {"rate": 0.15, "depth": 0.8, "stages": 4, "feedback": 0.4, "mix": 0.5}}
    n["reverb"] = {"type": "Reverb", "params": {"room_size": 0.85, "mix": 0.4}}
    n["out"] = {"type": "audio_out"}
    c += [("mixer/output", "phaser/input"), ("phaser/output", "reverb/input"),
          ("reverb/output", "out/input")]
    return build_graph(n, c)


def flanger_keys():
    n, c = source_nodes()
    n["amp_env"] = amp_env(0.01, 0.3, 0.6, 0.4)
    c.append(("voices/gates", "amp_env/gates"))

    n["osc"] = {"type": "WavetableOsc", "pkg": PKG,
                 "params": {"wavetable": 0, "position": 0.2, "amplitude": 0.3}}
    c += voice_wires("osc")

    n["filter"] = {"type": "Filter",
                    "params": {"mode": 1, "cutoff": 5000.0, "resonance": 0.15}}
    c += [("osc/output", "filter/input"), ("voices/frequencies", "filter/frequencies"),
          ]

    n["mixer"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.4}}
    c += [("filter/output", "mixer/input"), ("amp_env/envelopes", "mixer/amp_env")]

    n["flanger"] = {"type": "Flanger", "params": {"rate": 0.3, "depth": 0.6, "feedback": 0.5, "mix": 0.4}}
    n["reverb"] = {"type": "Reverb", "params": {"room_size": 0.5, "mix": 0.25}}
    n["out"] = {"type": "audio_out"}
    c += [("mixer/output", "flanger/input"), ("flanger/output", "reverb/input"),
          ("reverb/output", "out/input")]
    return build_graph(n, c)


def compressed_pluck():
    n, c = source_nodes()
    n["amp_env"] = amp_env(0.001, 0.3, 0.0, 0.2)
    n["filt_env"] = filt_env(0.001, 0.2, 0.0, 0.15)
    c += [("voices/gates", "amp_env/gates"), ("voices/gates", "filt_env/gates")]

    n["osc"] = {"type": "WavetableOsc", "pkg": PKG,
                 "params": {"wavetable": 2, "position": 0.5, "amplitude": 0.3}}
    c += voice_wires("osc")

    n["filter"] = {"type": "Filter",
                    "params": {"mode": 1, "cutoff": 8000.0,
                               "resonance": 0.15, "keytrack": 0.5}}
    c += [("osc/output", "filter/input"), ("voices/frequencies", "filter/frequencies"),
          ("filt_env/envelopes", "filter/cutoff_mod")]

    n["mixer"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.3, "vel_to_volume": 1.0}}
    c += [("filter/output", "mixer/input"), ("amp_env/envelopes", "mixer/amp_env"),
          ("voices/velocities", "mixer/velocities")]

    n["comp"] = {"type": "Compressor", "params": {"threshold": -18.0, "ratio": 6.0, "attack": 2.0, "release": 80.0}}
    n["ppd"] = {"type": "PingPongDelay", "params": {"time": 250.0, "feedback": 0.35, "spread": 1.0, "mix": 0.3}}
    n["reverb"] = {"type": "Reverb", "params": {"room_size": 0.5, "mix": 0.2}}
    n["out"] = {"type": "audio_out"}
    c += [("mixer/output", "comp/input"), ("comp/output", "ppd/input"),
          ("ppd/output", "reverb/input"), ("reverb/output", "out/input")]
    return build_graph(n, c)


def dual_filter_split():
    n, c = source_nodes()
    n["amp_env"] = amp_env(0.3, 0.5, 0.7, 1.0)
    n["filt_env"] = filt_env(0.3, 0.8, 0.3, 1.0)
    c += [("voices/gates", "amp_env/gates"), ("voices/gates", "filt_env/gates")]

    n["osc"] = {"type": "WavetableOsc", "pkg": PKG,
                 "params": {"wavetable": 1, "position": 0.5, "amplitude": 0.25,
                            "unison_voices": 8, "unison_spread": 35.0, "unison_stereo": 0.9,
                            "unison_spread_mode": 1}}
    c += voice_wires("osc")

    n["filt_a"] = {"type": "Filter",
                    "params": {"mode": 6, "cutoff": 2000.0, "resonance": 0.3}}
    n["filt_b"] = {"type": "Filter",
                    "params": {"mode": 7, "cutoff": 3000.0, "resonance": 0.4}}
    c += [("osc/output", "filt_a/input"), ("osc/output", "filt_b/input"),
          ("voices/frequencies", "filt_a/frequencies"),
          ("voices/frequencies", "filt_b/frequencies"),
          ("filt_env/envelopes", "filt_a/cutoff_mod"), ("filt_env/envelopes", "filt_b/cutoff_mod")]

    n["mix_a"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.7}}
    n["mix_b"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.7}}
    c += [("filt_a/output", "mix_a/input"), ("filt_b/output", "mix_b/input"),
          ("amp_env/envelopes", "mix_a/amp_env"), ("amp_env/envelopes", "mix_b/amp_env")]

    n["sum"] = {"type": "Mixer", "params": {"gain_1": 0.7, "gain_2": 0.5}}
    n["reverb"] = {"type": "Reverb", "params": {"room_size": 0.7, "mix": 0.35}}
    n["out"] = {"type": "audio_out"}
    c += [("mix_a/output", "sum/input_1"), ("mix_b/output", "sum/input_2"),
          ("sum/output", "reverb/input"), ("reverb/output", "out/input")]
    return build_graph(n, c)


def noise_osc_layer():
    n, c = source_nodes()
    n["amp_env"] = amp_env(0.5, 0.6, 0.5, 1.5)
    n["filt_env"] = filt_env(0.5, 1.0, 0.2, 1.0)
    c += [("voices/gates", "amp_env/gates"), ("voices/gates", "filt_env/gates")]

    n["osc"] = {"type": "WavetableOsc", "pkg": PKG,
                 "params": {"wavetable": 4, "position": 0.6, "amplitude": 0.2,
                            "unison_voices": 3, "unison_spread": 20.0}}
    n["noise"] = {"type": "Noise",
                   "params": {"amplitude": 0.2, "color": 1}}
    c += voice_wires("osc") + [("voices/gates", "noise/gates")]

    n["filter"] = {"type": "Filter",
                    "params": {"mode": 1, "cutoff": 3000.0, "resonance": 0.15}}
    c += [("osc/output", "filter/input"), ("voices/frequencies", "filter/frequencies"),
          ("filt_env/envelopes", "filter/cutoff_mod")]

    # Per-voice chorus on the filtered osc
    n["chorus"] = {"type": "Chorus", "params": {"rate": 0.3, "depth": 0.4, "mix": 0.3}}
    c.append(("filter/output", "chorus/input"))

    n["mixer"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.7}}
    n["noise_mix"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.9}}
    c += [("chorus/output", "mixer/input"), ("amp_env/envelopes", "mixer/amp_env"),
          ("noise/output", "noise_mix/input"), ("amp_env/envelopes", "noise_mix/amp_env")]

    n["sum"] = {"type": "Mixer", "params": {"gain_1": 1.0, "gain_2": 0.6}}
    n["reverb"] = {"type": "Reverb", "params": {"room_size": 0.7, "mix": 0.35}}
    n["out"] = {"type": "audio_out"}
    c += [("mixer/output", "sum/input_1"), ("noise_mix/output", "sum/input_2"),
          ("sum/output", "reverb/input"), ("reverb/output", "out/input")]
    return build_graph(n, c)


def sub_lead():
    n, c = source_nodes()
    n["amp_env"] = amp_env(0.01, 0.15, 0.75, 0.3)
    n["filt_env"] = filt_env(0.01, 0.4, 0.2, 0.3)
    c += [("voices/gates", "amp_env/gates"), ("voices/gates", "filt_env/gates")]

    n["osc"] = {"type": "AnalogOsc", "pkg": PKG,
                 "params": {"waveform": 1, "amplitude": 0.3, "detune": 10.0}}
    n["sub"] = {"type": "SubOsc", "pkg": PKG,
                 "params": {"level": 0.5, "octave": 1, "waveform": 0}}
    c += voice_wires("osc") + [("voices/frequencies", "sub/frequencies"),
                                 ("voices/gates", "sub/gates")]

    n["filter"] = {"type": "Filter",
                    "params": {"mode": 1, "cutoff": 3000.0,
                               "resonance": 0.2, "drive": 0.15}}
    c += [("osc/output", "filter/input"), ("voices/frequencies", "filter/frequencies"),
          ("filt_env/envelopes", "filter/cutoff_mod")]

    # Per-voice distortion
    n["dist"] = {"type": "Distortion", "params": {"drive": 1.5, "tone": 0.5, "level": 0.9, "mix": 0.3}}
    c.append(("filter/output", "dist/input"))

    n["mixer"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.3}}
    n["sub_mix"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.0}}
    c += [("dist/output", "mixer/input"), ("amp_env/envelopes", "mixer/amp_env"),
          ("sub/output", "sub_mix/input"), ("amp_env/envelopes", "sub_mix/amp_env")]

    n["sum"] = {"type": "Mixer", "params": {"gain_1": 1.0, "gain_2": 1.0}}
    n["reverb"] = {"type": "Reverb", "params": {"room_size": 0.4, "mix": 0.15}}
    n["out"] = {"type": "audio_out"}
    c += [("mixer/output", "sum/input_1"), ("sub_mix/output", "sum/input_2"),
          ("sum/output", "reverb/input"), ("reverb/output", "out/input")]
    return build_graph(n, c)


def acid_squelch():
    n, c = source_nodes()
    n["amp_env"] = amp_env(0.005, 0.15, 0.7, 0.1)
    n["filt_env"] = filt_env(0.005, 0.2, 0.0, 0.1)
    c += [("voices/gates", "amp_env/gates"), ("voices/gates", "filt_env/gates")]

    n["osc"] = {"type": "AnalogOsc", "pkg": PKG,
                 "params": {"waveform": 1, "amplitude": 0.35}}
    c += voice_wires("osc")

    n["filter"] = {"type": "Filter",
                    "params": {"mode": 6, "cutoff": 400.0,
                               "resonance": 0.7, "drive": 0.4}}
    c += [("osc/output", "filter/input"), ("voices/frequencies", "filter/frequencies"),
          ("filt_env/envelopes", "filter/cutoff_mod")]

    # Per-voice distortion
    n["dist"] = {"type": "Distortion", "params": {"drive": 3.0, "tone": 0.6, "level": 0.7, "mix": 0.5}}
    c.append(("filter/output", "dist/input"))

    n["mixer"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.0}}
    c += [("dist/output", "mixer/input"), ("amp_env/envelopes", "mixer/amp_env")]

    n["delay"] = {"type": "Delay", "params": {"time": 150.0, "feedback": 0.4, "mix": 0.25}}
    n["out"] = {"type": "audio_out"}
    c += [("mixer/output", "delay/input"), ("delay/output", "out/input")]
    return build_graph(n, c)


def glass_arp():
    n, c = source_nodes()
    n["amp_env"] = amp_env(0.001, 0.5, 0.0, 0.3)
    c.append(("voices/gates", "amp_env/gates"))

    n["modulator"] = {"type": "AnalogOsc", "pkg": PKG,
                       "params": {"waveform": 0, "amplitude": 0.4}}
    n["carrier"] = {"type": "WavetableOsc", "pkg": PKG,
                     "params": {"wavetable": 2, "position": 0.4, "amplitude": 0.25,
                                "interaction_mode": 1, "interaction_depth": 0.3,
                                "interaction_input_gain": 1.0, "interaction_tracking": 1.0}}
    c += voice_wires("modulator") + voice_wires("carrier")
    c.append(("modulator/output", "carrier/mod_input"))

    n["filter"] = {"type": "Filter",
                    "params": {"mode": 0, "cutoff": 6000.0,
                               "resonance": 0.1, "keytrack": 0.5}}
    c += [("carrier/output", "filter/input"), ("voices/frequencies", "filter/frequencies"),
          ]

    n["mixer"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.5}}
    c += [("filter/output", "mixer/input"), ("amp_env/envelopes", "mixer/amp_env")]

    n["ppd"] = {"type": "PingPongDelay", "params": {"time": 200.0, "feedback": 0.45, "spread": 1.0, "mix": 0.4}}
    n["reverb"] = {"type": "Reverb", "params": {"room_size": 0.6, "mix": 0.3}}
    n["out"] = {"type": "audio_out"}
    c += [("mixer/output", "ppd/input"), ("ppd/output", "reverb/input"),
          ("reverb/output", "out/input")]
    return build_graph(n, c)


def drone_wash():
    n, c = source_nodes()
    n["amp_env"] = amp_env(2.0, 1.0, 0.7, 3.0)
    c.append(("voices/gates", "amp_env/gates"))

    n["osc_a"] = {"type": "WavetableOsc", "pkg": PKG,
                   "params": {"wavetable": 4, "position": 0.5, "amplitude": 0.2,
                              "unison_voices": 8, "unison_spread": 50.0, "unison_stereo": 1.0,
                              "unison_spread_mode": 2}}
    n["osc_b"] = {"type": "WavetableOsc", "pkg": PKG,
                   "params": {"wavetable": 3, "position": 0.6, "amplitude": 0.15,
                              "unison_voices": 4, "unison_spread": 25.0}}
    n["noise"] = {"type": "Noise",
                   "params": {"amplitude": 0.15, "color": 1}}
    c += voice_wires("osc_a") + voice_wires("osc_b") + [("voices/gates", "noise/gates")]

    n["filter"] = {"type": "Filter",
                    "params": {"mode": 1, "cutoff": 2500.0, "resonance": 0.15}}
    c += [("osc_a/output", "filter/input"), ("voices/frequencies", "filter/frequencies"),
          ]

    n["mix_a"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.9}}
    n["mix_b"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 0.7}}
    n["noise_mix"] = {"type": "VoiceMixer", "pkg": PKG, "params": {"stereo_spread": 1.0}}
    c += [("filter/output", "mix_a/input"), ("amp_env/envelopes", "mix_a/amp_env"),
          ("osc_b/output", "mix_b/input"), ("amp_env/envelopes", "mix_b/amp_env"),
          ("noise/output", "noise_mix/input"), ("amp_env/envelopes", "noise_mix/amp_env")]

    n["sum"] = {"type": "Mixer", "params": {"gain_1": 1.0, "gain_2": 0.7, "gain_3": 0.4}}
    n["phaser"] = {"type": "Phaser", "params": {"rate": 0.08, "depth": 0.6, "stages": 4, "feedback": 0.3, "mix": 0.4}}
    n["reverb"] = {"type": "Reverb", "params": {"room_size": 0.9, "damping": 0.3, "mix": 0.5}}
    n["out"] = {"type": "audio_out"}
    c += [("mix_a/output", "sum/input_1"), ("mix_b/output", "sum/input_2"),
          ("noise_mix/output", "sum/input_3"),
          ("sum/output", "phaser/input"), ("phaser/output", "reverb/input"),
          ("reverb/output", "out/input")]
    return build_graph(n, c)


# =============================================================================

# This script remains as a lightweight seed-graph generator for the retained
# library rather than a way to recreate every historical expansion experiment.
PRESETS = {
    "warm_dual_pad": warm_dual_pad,
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, func in PRESETS.items():
        graph = func()
        path = os.path.join(OUTPUT_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(graph, f, indent=2)
        node_count = len(graph["nodes"])
        conn_count = len(graph["connections"])
        print(f"  {name:25s} ({node_count} nodes, {conn_count} connections)")

    print(f"\nGenerated {len(PRESETS)} retained seed presets to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
