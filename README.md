# vivid-wavetable

`vivid-wavetable` is an audio-first modular wavetable synthesis package for Vivid. Active graphs are written against the fixed-cadence core and use `_au` core control operators explicitly where clocking, envelopes, and modulation are needed.

## Preview

![vivid-wavetable preview](docs/images/preview.png)

## Operators

- **PolyVoiceAllocator** — converts MIDI and control inputs into polyphonic lane arrays (frequencies, gates, velocities, lane_ids)
- **WavetableOsc** — polyphonic wavetable oscillator with family/member source selection, phase/drift motion controls, warp modes, unison, and conditioned oscillator interaction
- **AnalogOsc** — polyphonic virtual analog oscillator with PolyBLEP anti-aliasing (sine, saw, square, triangle, pulse) and conditioned oscillator interaction
- **SubOsc** — polyphonic sub oscillator (sine, triangle, saw, square, noise)
- **NoiseLayer** — polyphonic per-note noise/air source for breath, attack detail, and texture layers
- **VoiceDrive** — lane-preserving soft drive for per-voice body, glue, and velocity-sensitive harmonic density
- **VoiceMixer** — sums N-channel per-voice audio to stereo with panning, velocity, envelope control, and optional output glue

## Contents

- `src/` — operator source files
- `modules/` — instrument-facing subgraph modules
- `assets/wavetables/` — factory wavetable wav files for the instrument library
- `factory_presets/` — per-operator factory presets
- `graphs/core/` — core smoke graphs (modular chain + asset smoke)
- `graphs/presets/` — curated showcase library and instrument graphs
- `tests/` — package tests
- `archive/` — legacy WavetableSynth monolith (frozen, not built)

## Local development

From vivid-core:

```bash
./build/vivid link ../vivid-wavetable
./build/vivid rebuild vivid-wavetable
```

## Build a synth

If you are new to the package, start here.

This is the beginner path for building a usable polysynth graph with the current lane architecture. It is meant to teach graph construction and musical intent, not debugging. If you want to verify that an operator is behaving correctly or isolate a bug by listening to tiny proof graphs, use the validation guide instead:

- [`docs/wavetable-operator-validation-guide.md`](docs/wavetable-operator-validation-guide.md)

### Step 1: Add a self-playing note source

Create these nodes:

- `ClockAu` as `clock`
- `ChordProgressionAu` as `chords`
- `PolyVoiceAllocator` as `voices`

Connect:

```text
clock/beat_phase -> chords/beat_phase
chords/notes -> voices/notes_in
chords/velocities -> voices/velocities_in
chords/gates -> voices/gates_in
```

Recommended starting params:

- `clock/bpm = 96`
- `voices/max_voices = 6`

What this does musically:

- `ClockAu` keeps time
- `ChordProgressionAu` produces note events
- `PolyVoiceAllocator` turns those notes into one lane per note

### Step 2: Add the main oscillator voice

Create:

- `WavetableOsc` as `osc`

Connect:

```text
voices/frequencies -> osc/frequencies
voices/gates -> osc/gates
voices/velocities -> osc/velocities
voices/lane_ids -> osc/lane_ids
```

Recommended starting params:

- `osc/amplitude = 0.25`
- `osc/wavetable_family = AnalogWarm`
- `osc/wavetable_member = Core`
- `osc/position = 0.35`
- `osc/unison_voices = 2`
- `osc/unison_spread = 12`

What this does musically:

- this is the raw tone source
- before reduction and envelopes, it will sound rougher than a finished preset, and that is okay

### Step 3: Add a per-note amp envelope and final reduction

Create:

- `EnvelopeAu` as `amp_env`
- `VoiceMixer` as `mixer`
- `audio_out` as `out`

Connect:

```text
voices/gates -> amp_env/gate
osc/output -> mixer/input
amp_env/value -> mixer/amp_env_audio
voices/velocities -> mixer/velocities
mixer/output -> out/input
```

Recommended starting params:

- `amp_env/attack = 0.01`
- `amp_env/decay = 0.25`
- `amp_env/sustain = 0.70`
- `amp_env/release = 0.40`
- `mixer/stereo_spread = 0.70`

What this does musically:

- `EnvelopeAu` shapes each note independently
- `VoiceMixer` is where the separate note lanes become one stereo output
- this is the first point where the patch should sound like a real playable synth instead of a raw lane test

### Step 4: Add a musical filter layer

Create:

- `Filter` as `filter`
- `EnvelopeAu` as `filt_env`

Rewire:

```text
osc/output -> filter/input
filter/output -> mixer/input
voices/frequencies -> filter/frequencies
voices/gates -> filt_env/gate
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

- adds note-shaped brightness and movement
- makes the synth feel played rather than statically bright

> **Advanced tone shaping:** For selected finished voices, the package uses `DualFilter` as its advanced tone-shaping option. `DualFilter` provides two independent filter stages with configurable routing (serial, parallel, or crossover split) for richer body/edge separation. See the `DualWavetablePad`, `HybridKeys`, and `SubAirPad` module internals for examples.

### Step 5: Add optional character layers

Once the basic synth is working, layer in one extra character block at a time:

- `SubOsc` for low support
- `NoiseLayer` for air, breath, and transient detail
- `VoiceDrive` for body and per-voice saturation before reduction

Typical connections:

```text
voices/frequencies,gates,velocities,lane_ids -> SubOsc/...
voices/frequencies,gates,velocities,lane_ids -> NoiseLayer/...
osc/output -> VoiceDrive/input
voices/velocities -> VoiceDrive/velocities
VoiceDrive/output -> mixer/input
```

Good first-use settings:

- `SubOsc/level = 0.20`
- `NoiseLayer/level = 0.06`
- `NoiseLayer/tone = 0.68`
- `VoiceDrive/drive = 0.18`
- `VoiceDrive/tone = 0.52`

### Step 6: Explore wavetable motion and interaction

After the basic graph feels clear, try the two most important extension paths:

- wavetable motion:
  - `LfoAu/value -> osc/position_mod_audio`
  - or `EnvelopeAu/value -> osc/position_mod_audio`
- oscillator interaction:
  - feed another oscillator into `osc/mod_input`
  - start with `interaction_mode = PM`
  - then raise `interaction_depth` slowly

Recommended interaction starting point:

- `interaction_mode = PM`
- `interaction_depth = 0.18`
- `interaction_input_gain = 1.0`
- `interaction_tracking = 1.0`

### Good next reference graphs

After building the basic synth once by hand, these retained graphs are good next examples:

- `graphs/presets/single_osc_motion_reference.json` for clear wavetable motion
- `graphs/presets/airy_keys.json` for Pass 3 layering
- `graphs/presets/fm_glass_keys.json` for Pass 4 interaction

The old synth-building doc is now just a pointer:

- [`docs/synth-building-tutorial.md`](docs/synth-building-tutorial.md)

## Module Instruments

Pass 1 of the April 4 instrument-adoption plan adds the package's first instrument-facing module wrappers on top of the existing operator stack.

- `HybridKeys` — the clearest finished-voice hybrid keys instrument in the package
- `GlassInteractionKeys` — a compact wrapper around the playable glass/interaction voice
- `DualWavetablePad` — the new canonical dual-wavetable pad architecture
- `SubAirPad` — a canonical wavetable + sub + air pad voice

These modules are additive. The package still exposes the underlying operators and the retained plain-graph reference patches, but the new module surface gives Vivid a much smaller instrument entrypoint for common voice architectures.

## Instrument Library

The package ships a browseable instrument library alongside its self-playing examples. Instrument graphs use `MidiInput` and carry `content_kind: instrument` metadata for host browsing.

**Keys**
- **Glass Interaction Keys** (`glass_interaction_instrument.json`) — hero: interactive glass keys with aftertouch-to-interaction mapping
- **Hybrid Keys** (`hybrid_keys_instrument.json`) — reference: dual-layer wavetable + analog keys

**Pads**
- **Dual Wavetable Pad** (`dual_wavetable_pad_instrument.json`) — hero: layered dual-wavetable pad with shared motion
- **Sub Air Pad** (`sub_air_pad_instrument.json`) — reference: wavetable + sub + air pad

**Bass**
- **Rooted Sub Bass** (`rooted_sub_bass_instrument.json`) — hero: grounded sub-layered bass

**Texture**
- **Motion Texture** (`motion_texture_instrument.json`) — utility: LFO-driven motion texture bed

## Factory Wavetable Assets

The package ships 6 factory wavetable files under `assets/wavetables/`, declared in the package manifest. All four modules expose `wavetable_source` and file params so instruments can switch between the builtin bank and custom wav files.

- **Package factory assets** — read-only, shipped with the package, safe to reference in committed graphs
- **User-imported workspace assets** — imported into the local workspace library via `import_asset`, consumed through the same `wavetable_source=Custom` + `wav_file` workflow, but not committed to the repo

Package CI automatically validates the manifest-declared factory asset set, factory wavetable loading, and package-relative asset-backed smoke graphs. The workspace-import flow is supported by the same file-backed module surface, but it is currently documented as a manual or opt-in integration check rather than something package tests prove on their own.

See [`docs/wavetable-asset-workflow.md`](docs/wavetable-asset-workflow.md) for details.

## Performance Surface

Pass 3 adds a package-wide performance vocabulary to the module instruments. Each module tags selected exposed params with `performance_page` and `performance_role` metadata so hosts can present a coherent live-control surface.

Five canonical roles:

- `motion` — wavetable travel or movement depth (headline: DualWavetablePad)
- `brightness` — top-end openness / main tone opening (all four modules)
- `air` — upper noise, shimmer, or breath support (headline: SubAirPad)
- `body` — low-mid weight, drive, or glue (headline: HybridKeys)
- `interaction` — carrier/modulator complexity amount (headline: GlassInteractionKeys)

Not every module exposes every role. But when a role appears, it means the same musical thing across the package.

Those names are the package's user-facing control vocabulary. The stable exposed module params remain compatibility-oriented names such as `filter_cutoff`, `drive`, `air_level`, `filter_tone`, `motion_amount`, and `interaction_depth`.

Two performance pages:

- **Performance** — the live macro layer (motion, air, interaction, body, brightness)
- **Timbre** — the tone/brightness detail layer

Modules also declare named `mod_sources` and `mod_destinations` for module-local modulation. The expressive demo (`graphs/presets/expressive_glass_keys.json`) shows scalar `aftertouch` and `expression` from `MidiInput` driving interaction and brightness on the GlassInteractionKeys module.

## Curated showcase library

The package now ships a deliberately curated preset library instead of carrying every experiment from the expansion passes forward.

- Showcase overview: [`docs/showcase-library.md`](docs/showcase-library.md)
- Retained motion reference: `graphs/presets/single_osc_motion_reference.json`
- Clear Pass 3 reference: `graphs/presets/airy_keys.json`
- Clear Pass 4 reference: `graphs/presets/fm_glass_keys.json`

The retained library is organized around eight listening families:

- pads and beds
- keys and brass
- plucks and bells
- leads
- basses
- textures and drones
- arp and sequence patches
- cinematic hybrids

## Wavetable families

`WavetableOsc` now organizes built-in tables as `family + member` instead of one flat coarse selector.

- Families: `AnalogWarm`, `BrightDigital`, `VocalFormant`, `Metallic`, `HarmonicSpectral`, `TextureMotion`
- Shared members: `Core`, `Soft`, `Rich`, `Hollow`, `Sweep`, `Glass`, `Edge`, `Air`

The shared member labels are intentionally approximate tonal roles so presets can move between families without changing how the control surface reads.

## Pass 3 layering blocks

Pass 3's character-layering surface is built around three lightweight roles:

- `NoiseLayer` for per-note air, breath, and transient detail
- `VoiceDrive` for body and per-voice harmonic glue before reduction
- `VoiceMixer.glue` for subtle post-sum cohesion on dense layered sounds

## Pass 4 interaction surface

Pass 4 redesigns oscillator-to-oscillator interaction around one shared carrier-side model in `WavetableOsc` and `AnalogOsc`:

- `interaction_mode` = `Off`, `FM`, `PM`, `RM`, `AM`
- `interaction_depth` for the musical amount
- `interaction_input_gain` for how hard the incoming modulator drives the carrier
- `interaction_tracking` for how much the interaction follows carrier pitch

`PM` is the preferred starting point for stable glass and metallic keys/leads. Use `FM` when you want stronger growl or more obviously pitch-coupled interaction. `RM` and `AM` are now depth-aware and intended to be dialed, not used as all-or-nothing tricks.

## CI smoke coverage

The package CI workflow:

1. Clones and builds vivid-core (`test_demo_graphs` + core operators).
2. Builds package operators and all package tests, including `test_audio_correctness`.
3. Runs package `ctest` against the active modular surface.
4. Runs graph smoke tests against `graphs/core/` plus focused hero/reference batches from `graphs/presets/` after copying the package dylibs into the vivid-core build.
5. Leaves `archive/` out of active smoke coverage.

## License

MIT (see `LICENSE`).
