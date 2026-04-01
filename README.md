# vivid-wavetable

`vivid-wavetable` is a modular wavetable synthesis package for Vivid, providing polyphonic oscillators, voice allocation, and mixing as separate composable operators.

## Preview

![vivid-wavetable preview](docs/images/preview.png)

## Operators

- **PolyVoiceAllocator** — converts MIDI and control inputs into polyphonic lane arrays (frequencies, gates, velocities, lane_ids)
- **WavetableOsc** — polyphonic wavetable oscillator with per-voice channel output, warp modes, unison, and audio-rate modulation
- **AnalogOsc** — polyphonic virtual analog oscillator with PolyBLEP anti-aliasing (sine, saw, square, triangle, pulse)
- **SubOsc** — polyphonic sub oscillator (sine, triangle, saw, square, noise)
- **VoiceMixer** — sums N-channel per-voice audio to stereo with panning, velocity, and envelope control

## Contents

- `src/` — operator source files
- `factory_presets/` — per-operator factory presets
- `graphs/core/wavetable_modular_demo.json` — core smoke graph (modular chain)
- `graphs/extended/` — extended demo graphs (require `vivid-sequencers`)
- `graphs/presets/` — preset demo graphs
- `tests/` — package tests
- `archive/` — legacy WavetableSynth monolith (frozen, not built)

## Local development

From vivid-core:

```bash
./build/vivid link ../vivid-wavetable
./build/vivid rebuild vivid-wavetable
```

## CI smoke coverage

The package CI workflow:

1. Clones and builds vivid-core (`test_demo_graphs` + core operators).
2. Builds package operators and package tests.
3. Runs package tests.
4. Runs graph smoke tests against `graphs/core/` using the modular demo graph.
5. Optionally runs `graphs/extended/` when `VIVID_RUN_EXTENDED_GRAPHS=1` is set as a repo variable.

## License

MIT (see `LICENSE`).
