#!/usr/bin/env python3
"""Generate factory wavetable WAV files with original harmonic content.

Each wavetable is a mono 48kHz 16-bit WAV with frames of 2048 samples.
The frame axis morphs the harmonic spectrum so scanning position produces
timbral movement.

Usage:
    python3 scripts/generate_factory_wavetables.py
"""

import struct
import math
import os
import wave

SAMPLES_PER_FRAME = 2048
SAMPLE_RATE = 48000


def write_wavetable_wav(path: str, frames: list[list[float]]):
    """Write a list of single-cycle frames as a mono 16-bit WAV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        buf = bytearray()
        for frame in frames:
            for s in frame:
                s = max(-1.0, min(1.0, s))
                buf += struct.pack("<h", int(s * 32767))
        wf.writeframes(buf)
    print(f"  {path} ({len(frames)} frames, {len(frames) * SAMPLES_PER_FRAME} samples)")


def make_frame(harmonics: list[tuple[int, float, float]], n=SAMPLES_PER_FRAME) -> list[float]:
    """Synthesize one frame from a list of (harmonic_number, amplitude, phase_offset)."""
    frame = [0.0] * n
    for h, amp, phase in harmonics:
        for i in range(n):
            t = i / n
            frame[i] += amp * math.sin(2 * math.pi * h * t + phase)
    # Normalize peak to 0.95
    peak = max(abs(s) for s in frame) or 1.0
    scale = 0.95 / peak
    return [s * scale for s in frame]


def lerp_harmonics(a: list[tuple[int, float, float]],
                   b: list[tuple[int, float, float]],
                   t: float) -> list[tuple[int, float, float]]:
    """Crossfade between two harmonic specs. Assumes same harmonic numbers."""
    result = []
    a_dict = {h: (amp, ph) for h, amp, ph in a}
    b_dict = {h: (amp, ph) for h, amp, ph in b}
    all_h = sorted(set(k for k in a_dict) | set(k for k in b_dict))
    for h in all_h:
        a_amp, a_ph = a_dict.get(h, (0.0, 0.0))
        b_amp, b_ph = b_dict.get(h, (0.0, 0.0))
        result.append((h, a_amp + (b_amp - a_amp) * t, a_ph + (b_ph - a_ph) * t))
    return result


def morph_frames(start_harmonics, end_harmonics, num_frames=48):
    """Generate frames that morph between two harmonic specs."""
    frames = []
    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        harmonics = lerp_harmonics(start_harmonics, end_harmonics, t)
        frames.append(make_frame(harmonics))
    return frames


# ---------------------------------------------------------------------------
# Factory wavetable definitions
# ---------------------------------------------------------------------------

def gen_warm_keys_core():
    """Warm keys: fundamental-heavy with gentle odd harmonics, morphing to
    richer even+odd blend. Designed for hybrid keys body."""
    start = [(1, 1.0, 0), (2, 0.08, 0), (3, 0.35, 0), (5, 0.12, 0), (7, 0.04, 0)]
    end = [(1, 0.8, 0), (2, 0.4, 0), (3, 0.3, 0), (4, 0.2, 0), (5, 0.15, 0),
           (6, 0.1, 0), (7, 0.08, 0), (8, 0.05, 0)]
    return morph_frames(start, end, 48)


def gen_analog_soft():
    """Soft analog: rounded saw-like spectrum that thins to near-sine.
    Useful for gentle pad layers and soft lead tones."""
    start = [(1, 1.0, 0), (2, 0.5, 0), (3, 0.33, 0), (4, 0.25, 0),
             (5, 0.2, 0), (6, 0.16, 0), (7, 0.14, 0), (8, 0.12, 0)]
    end = [(1, 1.0, 0), (2, 0.05, 0), (3, 0.02, 0)]
    return morph_frames(start, end, 48)


def gen_rooted_bass_edge():
    """Bass edge: heavy fundamental + sub harmonics morphing to aggressive
    odd-harmonic edge. For bass instruments that need bite on the scan axis."""
    start = [(1, 1.0, 0), (2, 0.6, 0), (3, 0.15, 0)]
    end = [(1, 0.7, 0), (2, 0.2, 0), (3, 0.55, 0), (5, 0.4, 0),
           (7, 0.3, 0), (9, 0.2, 0), (11, 0.12, 0), (13, 0.06, 0)]
    return morph_frames(start, end, 48)


def gen_bright_pluck_edge():
    """Bright pluck: thin, harmonically rich attack character that morphs
    to a brighter, more metallic edge. For pluck and keys transients."""
    start = [(1, 0.6, 0), (2, 0.5, 0), (3, 0.45, 0), (4, 0.4, 0),
             (5, 0.35, 0), (6, 0.3, 0), (7, 0.25, 0), (8, 0.2, 0),
             (9, 0.15, 0), (10, 0.1, 0)]
    end = [(1, 0.3, 0), (3, 0.5, 0), (5, 0.6, 0), (7, 0.5, 0),
           (9, 0.4, 0), (11, 0.3, 0), (13, 0.2, 0), (15, 0.15, 0)]
    return morph_frames(start, end, 48)


def gen_vocal_pad_sweep():
    """Vocal pad: formant-like resonance peaks that shift across the morph
    axis, creating vowel-like movement. For vocal pads and choir textures."""
    num_frames = 64
    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        # Shift formant peaks through the morph
        f1_center = 3 + t * 2        # formant 1 sweeps harmonics 3-5
        f2_center = 7 + t * 4        # formant 2 sweeps harmonics 7-11
        harmonics = []
        for h in range(1, 20):
            amp = 0.0
            # Fundamental
            if h == 1:
                amp = 0.8
            # Formant 1
            dist1 = abs(h - f1_center)
            amp += 0.5 * math.exp(-dist1 * dist1 * 0.5)
            # Formant 2
            dist2 = abs(h - f2_center)
            amp += 0.3 * math.exp(-dist2 * dist2 * 0.3)
            if amp > 0.01:
                harmonics.append((h, amp, 0.0))
        frames.append(make_frame(harmonics))
    return frames


def gen_glass_motion():
    """Glass motion: metallic, bell-like partials with slight inharmonicity.
    Morphs from pure bell to dense metallic shimmer. For glass keys and leads."""
    num_frames = 48
    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        harmonics = []
        # Bell-like partials with slight inharmonic stretch
        for k in range(1, 14):
            # Inharmonic ratio: slightly stretched
            ratio = k * (1.0 + 0.003 * k * k)
            h = round(ratio)
            if h < 1:
                h = 1
            amp = (0.8 / k) * (1.0 - 0.3 * t) + t * 0.15 * math.sin(k * 1.7)
            amp = max(0.0, amp)
            phase = t * k * 0.4
            harmonics.append((h, amp, phase))
        # Deduplicate harmonics by number (keep highest amp)
        h_dict: dict[int, tuple[float, float]] = {}
        for h, amp, ph in harmonics:
            if h not in h_dict or amp > h_dict[h][0]:
                h_dict[h] = (amp, ph)
        frames.append(make_frame([(h, a, p) for h, (a, p) in sorted(h_dict.items())]))
    return frames


def gen_metallic_hollow():
    """Metallic hollow: suppressed even harmonics creating a hollow, metallic
    organ-like tone. Morphs from hollow to full. For metallic pad and drone layers."""
    start = [(1, 1.0, 0), (3, 0.6, 0), (5, 0.35, 0), (7, 0.2, 0),
             (9, 0.12, 0), (11, 0.06, 0)]
    end = [(1, 0.8, 0), (2, 0.5, 0), (3, 0.45, 0), (4, 0.35, 0),
           (5, 0.3, 0), (6, 0.25, 0), (7, 0.2, 0), (8, 0.15, 0),
           (9, 0.1, 0), (10, 0.08, 0)]
    return morph_frames(start, end, 48)


def gen_harmonic_rich():
    """Harmonic rich: dense, evenly weighted harmonic stack that thins
    progressively. For rich pad foundations and layered textures."""
    start = [(h, 0.7 / math.sqrt(h), 0) for h in range(1, 16)]
    end = [(1, 1.0, 0), (2, 0.3, 0), (3, 0.5, 0), (5, 0.15, 0), (8, 0.08, 0)]
    return morph_frames(start, end, 48)


def gen_texture_tide():
    """Texture tide: complex evolving texture with phase-shifted partials
    creating interference patterns. For ambient beds and motion textures."""
    num_frames = 48
    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        harmonics = []
        for h in range(1, 18):
            amp = 0.5 / (h ** 0.6) * (0.6 + 0.4 * math.sin(t * math.pi * 2 + h * 0.7))
            phase = t * h * 0.8 + math.sin(h * 0.3) * 1.5
            harmonics.append((h, max(0.0, amp), phase))
        frames.append(make_frame(harmonics))
    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FACTORY_SET = [
    ("assets/wavetables/warm-keys-core.wav",      gen_warm_keys_core),
    ("assets/wavetables/analog-soft.wav",          gen_analog_soft),
    ("assets/wavetables/rooted-bass-edge.wav",     gen_rooted_bass_edge),
    ("assets/wavetables/bright-pluck-edge.wav",    gen_bright_pluck_edge),
    ("assets/wavetables/vocal-pad-sweep.wav",      gen_vocal_pad_sweep),
    ("assets/wavetables/glass-motion.wav",         gen_glass_motion),
    ("assets/wavetables/metallic-hollow.wav",      gen_metallic_hollow),
    ("assets/wavetables/harmonic-rich.wav",        gen_harmonic_rich),
    ("assets/wavetables/texture-tide.wav",         gen_texture_tide),
]


def main():
    print(f"Generating {len(FACTORY_SET)} factory wavetables...")
    for path, gen_fn in FACTORY_SET:
        frames = gen_fn()
        write_wavetable_wav(path, frames)
    print(f"Done. {len(FACTORY_SET)} factory wavetables generated.")


if __name__ == "__main__":
    main()
