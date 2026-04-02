# vivid-wavetable

`vivid-wavetable` is an audio-first modular wavetable synthesis package for Vivid. Active graphs are written against the fixed-cadence core and use `_au` core control operators explicitly where clocking, envelopes, and modulation are needed.

## Preview

![vivid-wavetable preview](docs/images/preview.png)

## Operators

- **PolyVoiceAllocator** — converts MIDI and control inputs into polyphonic lane arrays (frequencies, gates, velocities, lane_ids)
- **WavetableOsc** — polyphonic wavetable oscillator with family/member source selection, phase/drift motion controls, warp modes, unison, and audio-rate modulation
- **AnalogOsc** — polyphonic virtual analog oscillator with PolyBLEP anti-aliasing (sine, saw, square, triangle, pulse)
- **SubOsc** — polyphonic sub oscillator (sine, triangle, saw, square, noise)
- **NoiseLayer** — polyphonic per-note noise/air source for breath, attack detail, and texture layers
- **VoiceMixer** — sums N-channel per-voice audio to stereo with panning, velocity, and envelope control

## Contents

- `src/` — operator source files
- `factory_presets/` — per-operator factory presets
- `graphs/core/wavetable_modular_demo.json` — core smoke graph (modular chain)
- `graphs/presets/` — preset demo graphs
- `tests/` — package tests
- `archive/` — legacy WavetableSynth monolith (frozen, not built)

## Local development

From vivid-core:

```bash
./build/vivid link ../vivid-wavetable
./build/vivid rebuild vivid-wavetable
```

## Wavetable families

`WavetableOsc` now organizes built-in tables as `family + member` instead of one flat coarse selector.

- Families: `AnalogWarm`, `BrightDigital`, `VocalFormant`, `Metallic`, `HarmonicSpectral`, `TextureMotion`
- Shared members: `Core`, `Soft`, `Rich`, `Hollow`, `Sweep`, `Glass`, `Edge`, `Air`

The shared member labels are intentionally approximate tonal roles so presets can move between families without changing how the control surface reads.

## CI smoke coverage

The package CI workflow:

1. Clones and builds vivid-core (`test_demo_graphs` + core operators).
2. Builds package operators and all package tests, including `test_audio_correctness`.
3. Runs package `ctest` against the active modular surface.
4. Runs graph smoke tests against `graphs/core/` and `graphs/presets/` after copying the package dylibs into the vivid-core build.
5. Leaves `archive/` out of active smoke coverage.

## License

MIT (see `LICENSE`).
