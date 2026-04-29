# Wavetable Operator Validation Guide

This guide is for **human listening validation** of the maintained `vivid-wavetable` package surface.

The production model is now:

```text
note stream -> synth
```

When a patch needs shared per-note control state, add:

```text
note stream -> NoteBreakout
```

Use `NoteBreakout` for `voice_gates`, `voice_ids`, `voice_freqs`, and other per-voice control lanes. The legacy `VoiceAllocator` graph operator was removed in Phase 3 of the midi-native-protocol migration.

## Production Validation Surface

Validate these as the current maintained path:

- `WavetableLayer` (built-in per-voice ADSR in the **Envelope** group: `attack`, `decay`, `sustain`, `release`; expression in the **Expression** group: `pressure_to_amp`, `timbre_to_position`)
- `NoteBreakout`
- `Filter` / `DualFilter` keytracking from `NoteBreakout/voice_freqs`
- External `Envelope` for **filter** modulation and for auxiliary per-voice layers — no longer for layer voice gain (PR3 retired `voice_gain_audio`; the layer's internal ADSR is the sole per-voice gain envelope)
- Layer-based modules such as `LayerPad`, `DualWavetablePad`, `HybridKeys`, and `SubAirPad`

Validate these as advanced legacy or auxiliary surfaces:

- `WavetableOsc`
- `VoiceMixer` when reducing auxiliary per-voice layers
- interaction-heavy modules that specifically require `WavetableOsc`

## Core Mental Model

- note sources emit `notes_out`
- synths consume `notes_in`
- `WavetableLayer` allocates and sums voices internally, and runs its own per-voice ADSR
- `NoteBreakout` exists when multiple downstream operators need the same per-voice control view (filter envelopes, aux layer gain, etc.)

Think of it like this:

- before the synth, the graph carries a native note stream
- `NoteBreakout` turns that stream into shared control lanes for envelopes, filters, and velocity-aware reducers
- after `WavetableLayer`, the signal is already stereo audio with amplitude articulation already applied

## Stage 1: Silent synth sanity

Build:

- `WavetableLayer` named `wt`
- `audio_out` named `out`

Connect:

```text
wt/output -> out/input
```

What you should hear:

- silence

Why this is correct:

- `WavetableLayer` should not self-start without a note stream

## Stage 2: Basic native-note path

Add:

- `Clock` named `clock`
- `ChordProgression` named `chords`

Connect:

```text
clock/beat_phase -> chords/beat_phase
chords/notes_out -> wt/notes_in
```

Recommended params:

- `clock/bpm = 96`
- `wt/amplitude = 0.05` (per-voice; see Tone & headroom checks for why this is much lower than a mono lead would use)
- `chords/gate_length = 0.97` (sustain across chord boundaries; see Tone & headroom checks)

What you should hear:

- a playable stereo wavetable voice
- note changes that follow the chord source

If it sounds wrong:

- silence usually means `notes_out -> notes_in` is missing
- stuck notes usually mean the note source is not emitting the stream you expect

## Stage 3: Per-note amplitude articulation (internal ADSR)

`WavetableLayer` runs its own per-voice ADSR — there is no `voice_gain_audio` port to wire an external envelope into. Articulation is set directly on the layer.

Set on `wt`:

- `wt/attack = 0.01` (seconds)
- `wt/decay = 0.20`
- `wt/sustain = 0.70`
- `wt/release = 0.40`

What you should hear:

- notes articulate cleanly instead of staying constantly open
- release tails ride out per-note (the layer tracks voice identity internally)

If it sounds wrong:

- flat sustain usually means `sustain` is at 1.0 — drop it
- chopped attacks usually mean `attack` is too short for the patch (try 0.01–0.05 s)
- no release tail usually means `release` is below the gate gap

Optional MPE expression (Stage 1.5):

- `wt/pressure_to_amp = 0.5` scales voice gain by `1 + depth × slot.pressure` (default 0.5; 0 disables, see `src/wavetable_layer.cpp:532–536`)
- `wt/timbre_to_position` offsets wavetable position from MPE Y/timbre (`src/wavetable_layer.cpp:506`)

These have no effect for non-MPE note sources but cost nothing to leave at defaults.

## Stage 4: Add filter articulation and keytracking

This is where `NoteBreakout` enters — the filter envelope and the filter's pitch tracking both need per-voice lanes.

Add:

- `NoteBreakout` named `voice_breakout`
- `Filter` or `DualFilter`
- `Envelope` named `filt_env`

Connect:

```text
chords/notes_out -> voice_breakout/notes_in
wt/output -> filter/input
voice_breakout/voice_freqs -> filter/frequencies
voice_breakout/voice_gates -> filt_env/gate
voice_breakout/voice_ids -> filt_env/lane_ids
filt_env/value -> filter/cutoff_mod
filter/output -> out/input
```

What you should hear:

- filter motion that follows each note rather than acting like one global sweep
- keytracking that keeps higher notes brighter when enabled

If it sounds wrong:

- static tone usually means `filt_env/value -> filter/cutoff_mod` is missing
- incorrect keyboard tracking usually means `voice_freqs -> filter/frequencies` is missing

## Stage 5: Validate layered instruments

For layered instruments, keep one shared note stream and one shared `NoteBreakout`. Each layer's amp envelope is its own (the layer's internal ADSR for `WavetableLayer`; explicit `Envelope` + `VoiceMixer/amp_env_audio` for auxiliary layers like `AnalogOsc` / `SubOsc` reduced through `VoiceMixer`).

Typical pattern:

```text
chords/notes_out -> wt/notes_in
chords/notes_out -> analog/notes_in
chords/notes_out -> sub/notes_in
chords/notes_out -> voice_breakout/notes_in
analog/voices_out -> mix_analog/input
sub/voices_out -> mix_sub/input
amp_env/value -> mix_analog/amp_env_audio
amp_env/value -> mix_sub/amp_env_audio
voice_breakout/voice_velocities -> mix_analog/velocities
voice_breakout/voice_velocities -> mix_sub/velocities
```

What you should hear:

- the same note stream driving all layers coherently
- `VoiceMixer` only reducing auxiliary per-voice layers, not replacing `WavetableLayer`'s stereo output path

If it sounds wrong:

- layers drifting apart usually means the synths are not sharing the same note stream family
- dead per-note modulation on auxiliary layers usually means their `amp_env_audio` or `velocities` hookups are missing

## Bottom-up isolation: when nothing sounds right

If **multiple presets sound wrong in similar ways** (dull, harsh, thin, distorted, clicking, lopsided), the cause is almost always *below* the preset level — wavetable lookup, internal ADSR, voice summing, expression routing, or the filter — not a parameter you can fix per-preset. Tuning a preset on top of a broken DSP path just hides where the rot is. This section is a stepwise reduction: build the simplest possible signal and add one layer of complexity at a time until the symptom appears. The rung where it appears is the rung that owns the bug.

Run each step in a fresh, hand-built graph (don't open a preset). At each step, **compare against the "good" description before proceeding** — moving on past a wrong-sounding step folds noise into the next.

### Step 1 — Audio path sanity (no wavetable)

Build: `Oscillator` (the audio operator, *not* `WavetableLayer`) → `audio_out`. Set `frequency = 440`, `waveform = sine`, `amplitude = 0.1`.

- **Good**: clean 440 Hz sine, no clicks, no aliasing, no DC.
- **Bad**: aliased / crackly / silent / DC-offset.
- **Implicates**: not a wavetable problem. Runtime, sample rate, output path, or master gain. Stop here and check `mcp__vivid__runtime_status` and the audio device config.

### Step 2 — Bare `WavetableLayer`, single voice, no expression, no filter

Build: `Clock` → `ChordProgression` (single degree, voicing=1) → `WavetableLayer/notes_in` → `audio_out`.

`WavetableLayer` params:
- `amplitude = 0.1`, `unison_voices = 1`
- `wavetable_family = 0`, `wavetable_member = 0` (basic sine), `position = 0`
- `attack = 0.01`, `decay = 0.1`, `sustain = 1.0`, `release = 0.1`
- `pressure_to_amp = 0`, `timbre_to_position = 0` (force expression off)
- `phase_random = 0`, `drift_amount = 0`

- **Good**: clean fundamental on each note, no harmonics beyond what the wavetable contains, no clicks at the 256-sample declick boundary.
- **Bad symptoms and what they implicate**:
  - Aliased / fizzy harmonic content → wavetable lookup or interpolation (`src/wavetable_layer_renderer*.cpp`)
  - Sounds detuned or warbling at `unison_voices=1` → unison spread or detune leaking when it shouldn't
  - Click on each note-on past 5 ms → declick ramp not being applied (`kDeClickSamples` at `src/wavetable_layer_renderer.h:85`)
  - Silent → `notes_out → notes_in` not propagating, or per-voice gain stuck at 0

### Step 3 — Internal ADSR

Same graph as Step 2, but set `sustain = 0.5`, `release = 0.5`, `gate_length` on chords = 0.5 (short gate, audible release tail).

- **Good**: each note attacks, decays to half, sustains, releases over ~0.5 s with no clicks at any envelope-stage boundary.
- **Bad**: stuck-on (no envelope shape) → ADSR not running. Click at note-off → release stage not blending with declick.

### Step 4 — Polyphony summing

Same as Step 3 but use a 4-voice chord (`degree_0..3`, all `voicing_N=1`).

- **Good**: 4 distinct partials sum cleanly. Loudness ≈ 4× single-voice (no per-poly normalization — the guide's headroom section explains why).
- **Bad symptoms**:
  - Some voices missing or stuck → voice allocator (`VoiceTable<N>` per `src/wavetable_layer.cpp`).
  - Loudness scales weirdly (e.g. quieter than 1 voice) → summing or normalization regression. Compare against `4 × amplitude × (1/√unison)` from `src/wavetable_layer.cpp:247`.
  - Voices "fight" (intermittent dropouts) → voice stealing.

### Step 5 — Expression defaults

Critical step: the Phase 4/5 expression params are recent (`pressure_to_amp`, `timbre_to_position`). Their defaults must be **inert for non-MPE note sources**.

Take the Step 4 graph and toggle `pressure_to_amp` between **0** and the default (**0.5**) without changing anything else. ChordProgression emits no MPE pressure, so `slot.pressure = 0` and the math `gain *= (1 + 0.5 × 0) = 1.0` should produce *bit-identical* output.

- **Good**: switching `pressure_to_amp` between 0 and 0.5 produces no audible change.
- **Bad**: any audible change → either `slot.pressure` is not initializing to 0 for non-MPE notes (initialization bug in the note-stream → slot path) or the math at `src/wavetable_layer.cpp:532–536` is off. This is a **prime suspect** if every preset sounds louder/quieter than it should — the default boost would apply to every voice silently.

Repeat with `timbre_to_position`: should also be inert at default for non-MPE input. Reference: `src/wavetable_layer.cpp:506`.

### Step 6 — Filter neutrality

Take the Step 4 graph (no expression). Add `Filter` between `wt/output` and `out/input`. Set `cutoff = 20000`, `resonance = 0`, `mode = 0` (LP).

- **Good**: a 20 kHz LP with Q=0 should sound *identical* to no filter. Bypass the filter via `mcp__vivid__set_node_bypassed` and A/B — they should match.
- **Bad**: filter darkens the signal at 20 kHz / Q=0 → filter coefficients are wrong, or the filter is processing in a state that doesn't account for the requested cutoff.

### Step 7 — Audio-rate modulation

Add an `Lfo` at audio rate (rate ~0.5 Hz) and connect its output to `wt/pitch_mod_audio`. Depth should produce ~5 cents of pitch wobble.

- **Good**: pitch wobble at 0.5 Hz audible.
- **Bad**: silent / no wobble → audio-rate mod port regression. There is a known prior incident around audio-rate mod ports (port indexing, `_mod_audio` vs `_mod` confusion, VoiceMixer `amp_env_audio` zero-filled buffers). If this step fails, the symptom is consistent with a regression of one of those fixes; inspect `graph_compiler.cpp` port indexing and `WavetableLayer::collect_ports` for the audio mod port declarations.

### Step 8 — Preset comparison against a known-good

Pick a preset that's known to have sounded right historically (the prior audio-rate mod debug from late March 2026 cited `dream_keys.json` at RMS 0.061, peak 0.288). Load it via `mcp__vivid__load_graph`, `wait_for_audio_settle`, `sample_node_outputs` on `audio_out`.

- **If the historical reference also sounds wrong now**: regression at HEAD. `git bisect` between the last known-good commit and HEAD using "preset loads and produces clean audio at expected RMS" as the test.
- **If the historical reference sounds right**: issue is in newer presets, or in the recent harshness/headroom tuning pass — start from the offending preset and reduce by toggling its newest-changed nodes off one at a time.

### Reporting a finding

When a step fails, capture:
1. The step number and exact graph (`mcp__vivid__inspect_graph` with `detail=full`).
2. `mcp__vivid__sample_node_outputs` on the relevant node (raw audio float values are more diagnostic than RMS).
3. Current `git rev-parse HEAD` and any uncommitted state (`git status`, `git diff`).
4. The "good" expectation from this guide alongside the observed behavior.

This packet is enough to either bisect or to point a fix at a specific operator without re-running the whole isolation chain.

## Tone & headroom checks

These are the listening targets the recent preset pass converged on. A patch that lands outside these ranges is usually too bright, too loud, or clicking on chord changes.

**Voice summing has no auto-normalization across polyphony.** A per-voice `amplitude` tuned at 1 voice clips at 8. Targets:

- pad / keys patches at 6–8-voice polyphony: `amplitude` ≈ 0.02–0.05 per voice
- drive / sub layers feeding `VoiceMixer`: ≈ 0.005–0.05
- monophonic leads: 0.10–0.20 is fine

**Filter cutoff is the harshness knob.** Targets after the recent tuning pass:

- pads / keys: cutoff 1.4–2.4 kHz
- leads / plucks: cutoff 3–5 kHz
- resonance below ≈ 0.25 unless the patch is explicitly resonant — Q above that brings the high-mid sparkle back

**Retrigger clicks live at chord boundaries.** Two settings keep them away:

- on the note source, `gate_length ≈ 0.95–0.98` so voices sustain across the chord boundary instead of releasing+re-attacking
- the layer applies a 256-sample declick ramp on note-on (`src/wavetable_layer_renderer.h:85`); this masks the transient but does not remove the cause — long gates do

If a chord-driven preset clicks: lengthen the gate first, then the layer's `attack`, before reaching for declick or filter changes.

## Recommended validation references

Listen against the tuned module presets — they reflect both the Phase 3–5 wiring and the harshness/headroom pass:

- `graphs/presets/dual_wavetable_pad_module_demo.json` against `modules/dual_wavetable_pad.vivid-module.json`
- `graphs/presets/hybrid_keys_module_demo.json` against `modules/hybrid_keys.vivid-module.json`
- `graphs/presets/sub_air_pad_module_demo.json` against `modules/sub_air_pad.vivid-module.json`
- `graphs/presets/glass_interaction_keys_module_demo.json` against `modules/glass_interaction_keys.vivid-module.json`

For raw `WavetableLayer` validation (no module shell):

- `graphs/core/wavetable_layer_filter_integration.json` — single layer + external mono `Filter` post stereo reduction
- `graphs/core/wavetable_modular_demo.json` — two layers with shared `filt_env` driving per-filter cutoff
- `graphs/core/wavetable_layer_stress.json` — four parallel layers, 4-voice unison each (Phase 5 stress)

These fixtures use raw layer wiring at pre-tuning amplitudes — they exercise the layer's voicing and routing surface, not the harshness/headroom targets that apply to presets.
