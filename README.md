# vivid-wavetable

`vivid-wavetable` is an audio-first modular wavetable synthesis package for Vivid.
The maintained package surface now follows the native-note model:

- note sources emit `notes_out`
- synths consume `notes_in`
- `NoteBreakout` exposes shared per-voice control lanes when a graph needs envelopes, keytracking, or layered coordination

## Preview

![vivid-wavetable preview](docs/images/preview.png)

## Operators

- **WavetableLayer** — production polyphonic wavetable renderer with internal unison and stereo summing
- **WavetableOsc** — advanced legacy wavetable oscillator with per-voice audio output and interaction features not present in `WavetableLayer`
- **AnalogOsc** — polyphonic virtual analog oscillator
- **SubOsc** — polyphonic sub oscillator
- **NoiseLayer** — polyphonic per-note noise and air source
- **VoiceDrive** — lane-preserving soft drive for per-voice body and velocity-sensitive harmonic density
- **VoiceMixer** — sums per-voice audio to stereo for auxiliary layers such as `AnalogOsc`, `SubOsc`, and `NoiseLayer`

## Contents

- `src/` — operator source files
- `modules/` — instrument-facing subgraph modules
- `assets/wavetables/` — factory wavetable wav files
- `factory_presets/` — per-operator factory presets
- `graphs/core/` — core smoke graphs and integration fixtures
- `graphs/presets/` — curated showcase graphs
- `tests/` — package tests
- `archive/` — frozen historical content

## Local development

From vivid-core:

```bash
./build/vivid link ../vivid-wavetable
./build/vivid rebuild vivid-wavetable
```

## Build a synth

If you are new to the package, start with the maintained native-note path.

For listening-oriented validation, use
[`docs/wavetable-operator-validation-guide.md`](docs/wavetable-operator-validation-guide.md).

### Step 1: Add a note source and a synth

Create:

- `ClockAu` as `clock`
- `ChordProgressionAu` as `chords`
- `WavetableLayer` as `wt`
- `audio_out` as `out`

Connect:

```text
clock/beat_phase -> chords/beat_phase
chords/notes_out -> wt/notes_in
wt/output -> out/input
```

Recommended starting params:

- `clock/bpm = 96`
- `wt/amplitude = 0.25`
- `wt/wavetable_family = AnalogWarm`
- `wt/wavetable_member = Core`
- `wt/position = 0.35`
- `wt/unison_voices = 2`
- `wt/unison_spread = 12`

What this does musically:

- `ChordProgressionAu` creates the note stream
- `WavetableLayer` allocates voices internally and renders stereo directly
- this is the simplest playable production path in the package

### Step 2: Add shared per-note articulation

Create:

- `NoteBreakout` as `voice_breakout`
- `EnvelopeAu` as `amp_env`

Connect:

```text
chords/notes_out -> voice_breakout/notes_in
voice_breakout/voice_gates -> amp_env/gate
voice_breakout/voice_ids -> amp_env/lane_ids
amp_env/value -> wt/voice_gain_audio
```

Recommended starting params:

- `amp_env/attack = 0.01`
- `amp_env/decay = 0.25`
- `amp_env/sustain = 0.70`
- `amp_env/release = 0.40`

What this does musically:

- `NoteBreakout` keeps one shared per-voice control view of the note stream
- `EnvelopeAu` shapes each note independently
- `WavetableLayer` still handles the actual audio rendering and stereo sum

### Step 3: Add a filter layer with keytracking

Create:

- `Filter` as `filter`
- `EnvelopeAu` as `filt_env`

Rewire:

```text
wt/output -> filter/input
filter/output -> out/input
voice_breakout/voice_freqs -> filter/frequencies
voice_breakout/voice_gates -> filt_env/gate
voice_breakout/voice_ids -> filt_env/lane_ids
filt_env/value -> filter/cutoff_mod
```

Recommended starting params:

- `filter/mode = LowPass`
- `filter/cutoff = 2200`
- `filter/resonance = 0.18`
- `filt_env/attack = 0.02`
- `filt_env/decay = 0.50`
- `filt_env/sustain = 0.20`
- `filt_env/release = 0.35`

What this does musically:

- `voice_freqs` gives the filter a per-note keytracking reference
- the filter envelope stays aligned with the same note identities as the amp envelope

### Step 4: Add auxiliary layers

For layered instruments, send the same note stream to additional synths and keep one shared `NoteBreakout` for control lanes.

Typical pattern:

```text
chords/notes_out -> analog/notes_in
chords/notes_out -> sub/notes_in
analog/voices_out -> mix_analog/input
sub/voices_out -> mix_sub/input
amp_env/value -> mix_analog/amp_env_audio
amp_env/value -> mix_sub/amp_env_audio
voice_breakout/voice_velocities -> mix_analog/velocities
voice_breakout/voice_velocities -> mix_sub/velocities
```

Use `VoiceMixer` only for auxiliary per-voice audio reduction. `WavetableLayer` itself does not need a `VoiceMixer`.

### Step 5: Use the advanced legacy path only when needed

If you specifically need oscillator interaction modes or feedback-style warp behavior that `WavetableLayer` excludes, use:

- `WavetableOsc + VoiceMixer`

That is an advanced legacy surface, not the default production path.

## Good next references

- `graphs/core/wavetable_layer_filter_integration.json` — minimal `notes_in + NoteBreakout` articulation path
- `graphs/core/wavetable_modular_demo.json` — two synths coordinated by one shared `NoteBreakout`
- `modules/layer_pad.vivid-module.json` — canonical single-layer instrument module
- `modules/hybrid_keys.vivid-module.json` — layered wavetable + analog instrument with shared control lanes

For maintained docs and concrete references:

- [`docs/wavetable-operator-validation-guide.md`](docs/wavetable-operator-validation-guide.md)
- [`docs/showcase-library.md`](docs/showcase-library.md)

## Module Instruments

- `LayerPad` — recommended single-layer production voice built on `WavetableLayer`
- `DualWavetablePad` — dual-layer module with shared `NoteBreakout` control
- `HybridKeys` — `WavetableLayer` plus analog support layer with shared articulation
- `SubAirPad` — `WavetableLayer` plus sub and air layers
- `GlassInteractionKeys` — advanced legacy interaction-focused module

New production content should use `WavetableLayer`, `NoteBreakout`, and the maintained Layer-based modules. Use raw `WavetableOsc + VoiceMixer` only when you need excluded legacy interaction features.
