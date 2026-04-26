# Wavetable Operator Validation Guide

This guide is for **human listening validation** of the current `vivid-wavetable` package after the `WavetableLayer` cutover.

The active production path is now:

```text
VoiceAllocator -> WavetableLayer -> voice_gain_audio -> stereo output
```

`WavetableOsc + VoiceMixer` remains available for advanced legacy behavior that `WavetableLayer` intentionally excludes, especially FM/PM/RM/AM oscillator interaction and feedback-style warp. `VoiceMixer` also remains the normal reduction stage for auxiliary per-voice sources such as `SubOsc`, `AnalogOsc`, and `NoiseLayer` when those layers are mixed beside a `WavetableLayer` body.

Automated regression checks already cover package manifests, module surfaces, graph load/smoke behavior, renderer correctness, and scalar/optimized backend equivalence. This document is for the listening checks that still need human ears.

## Production Validation Surface

Validate these as the active production path:

- `VoiceAllocator`
- `WavetableLayer`
- `EnvelopeAu` driving `WavetableLayer/voice_gain_audio`
- `LayerPad` and Layer-based module instruments
- optional character layers such as `SubOsc`, `AnalogOsc`, and `NoiseLayer` when reduced separately before final stereo mixing

Validate these as non-primary surfaces:

- `WavetableOsc`
- `VoiceMixer` when it is reducing a legacy wavetable voice or an auxiliary per-voice layer
- `GlassInteractionKeys` and other interaction patches using `mod_input`, FM/PM/RM/AM, or feedback-style warp

## What Lanes Mean

In `vivid-wavetable`, one note becomes one lane.

- `VoiceAllocator` turns notes into `frequencies`, `gates`, `velocities`, and `lane_ids`.
- `WavetableLayer` consumes those lanes internally, renders all unison voices, and outputs one stereo signal.
- `EnvelopeAu/value -> WavetableLayer/voice_gain_audio` is the normal per-note amplitude path.
- Extra per-note sources such as `SubOsc`, `AnalogOsc`, and `NoiseLayer` may still need their own reduction before being mixed with the `WavetableLayer` stereo bus.

Think of it like this:

- before `WavetableLayer`, note lanes describe playable voices
- inside `WavetableLayer`, voice rendering and unison summing happen in one production operator
- after `WavetableLayer`, the signal is normal stereo audio

## Stage 1: Silent Layer Sanity

Build this graph:

- `WavetableLayer` named `osc`
- `audio_out` named `out`

Connect:

```text
osc/output -> out/input
```

Set these params on `osc`:

- `amplitude = 0.25`
- `wavetable_source = Builtin`
- `wavetable_family = AnalogWarm`
- `wavetable_member = Core`
- `position = 0.35`

What you should hear:

- silence

Why this is correct:

- `WavetableLayer` should not self-start without note lanes
- it needs lane inputs such as frequency, gate, and lane id before it should make sound

If it sounds wrong:

- if you hear a pitched tone or hiss, the production layer is not respecting note-lane input

## Stage 2: Add VoiceAllocator

Add:

- `ClockAu` named `clock`
- `ChordProgressionAu` named `chords`
- `VoiceAllocator` named `voices`

Connect:

```text
clock/beat_phase -> chords/beat_phase
chords/notes -> voices/notes_in
chords/velocities -> voices/velocities_in
chords/gates -> voices/gates_in
voices/frequencies -> osc/frequencies
voices/gates -> osc/gates
voices/velocities -> osc/velocities
voices/lane_ids -> osc/lane_ids
```

Recommended params:

- `clock/bpm = 96`
- `voices/max_voices = 6`
- `osc/amplitude = 0.20`

What you should hear:

- a raw sustained polyphonic wavetable tone
- note changes should follow the chord source

If it sounds wrong:

- silence usually means one of the four voice-lane connections is missing
- stuck or smeared notes usually means `lane_ids` is missing or unstable

## Stage 3: Add Per-Note Amplitude

Add:

- `EnvelopeAu` named `amp_env`

Connect:

```text
voices/gates -> amp_env/gate
voices/lane_ids -> amp_env/lane_ids
amp_env/value -> osc/voice_gain_audio
```

Recommended params:

- `amp_env/attack = 0.01`
- `amp_env/decay = 0.25`
- `amp_env/sustain = 0.70`
- `amp_env/release = 0.40`

What you should hear:

- notes articulate cleanly instead of staying constantly open
- release tails should remain stable during chord changes

If it sounds wrong:

- if notes do not release, confirm `amp_env/value` reaches `osc/voice_gain_audio`
- if voices click or swap envelopes, confirm both `voices/lane_ids -> osc/lane_ids` and `voices/lane_ids -> amp_env/lane_ids`

## Stage 4: Validate Wavetable Motion

Use the same graph and add one modulation source at a time.

Position motion:

```text
LfoAu/value -> osc/position_mod_audio
```

Warp motion:

```text
LfoAu/value -> osc/warp_mod_audio
```

Recommended checks:

- changing `wavetable_family` and `wavetable_member` should clearly change source color
- changing `position` should change timbre, not note pitch
- `warp_mode` and `warp_amount` should change edge or motion while staying bounded
- enabling drift should thicken motion without runaway level changes

If it sounds wrong:

- if modulation changes pitch instead of tone, check that it is patched to `position_mod_audio` or `warp_mod_audio`, not `pitch_mod_audio`
- if a warp mode is expected to be FM feedback, that is a legacy `WavetableOsc` check, not a `WavetableLayer` check

## Stage 5: Validate Character Layers

For extra sources, keep `WavetableLayer` as the main wavetable body and reduce extra per-note sources separately before final stereo mixing.

Typical pattern:

```text
voices/... -> WavetableLayer/...
amp_env/value -> WavetableLayer/voice_gain_audio
voices/... -> SubOsc/...
SubOsc/output -> VoiceMixer/input
amp_env/value -> VoiceMixer/amp_env_audio
WavetableLayer/output -> Mixer/input_1
VoiceMixer/output -> Mixer/input_2
Mixer/output -> audio_out/input
```

What you should hear:

- the WavetableLayer body remains the main stereo source
- sub, analog, or noise layers support the body without forcing the WavetableLayer body back through a per-voice mixer path

If it sounds wrong:

- if the patch collapses or lanes feel confused, check that `WavetableLayer/output` is not routed into `VoiceMixer/input`
- if an auxiliary per-note source is silent, confirm it still has its own `VoiceMixer` and envelope input

## Stage 6: Reference Instrument Checks

Load and listen to the active production references:

- `graphs/presets/layer_pad_instrument.json`
- `graphs/core/wavetable_layer_pad_demo.json`
- migrated non-interaction instruments such as `bright_pluck_instrument.json`, `rooted_sub_bass_instrument.json`, `metallic_hollow_lead_instrument.json`, and `motion_texture_instrument.json`

Expected result:

- all should produce non-silent stereo output
- `LayerPad` should remain the clearest recommended production pad path
- migrated instruments may differ slightly from the old `WavetableOsc + VoiceMixer` versions, but should preserve their role, rough loudness, and musical identity

## Advanced Legacy Checks

Use this section only when validating features that are intentionally outside `WavetableLayer` v1.

Retained advanced legacy content includes:

- `modules/glass_interaction_keys.vivid-module.json`
- `graphs/presets/glass_interaction_instrument.json`
- `graphs/presets/controlled_metallic_lead.json`
- `graphs/presets/fm_glass_keys.json`
- `graphs/presets/growl_crossmod_bass.json`
- `graphs/presets/hybrid_motion_arp.json`
- `graphs/presets/spectral_interaction_texture.json`
- `graphs/presets/orbit_drone.json`

Validate these only for the advanced interaction or feedback behavior they preserve:

- `WavetableOsc/mod_input` drives the intended carrier
- FM/PM/RM/AM depth changes are audible and controllable
- feedback-style warp remains bounded and musically useful
- these patches are labeled legacy/advanced and are not presented as the recommended production path

## Final Checklist

- `WavetableLayer` stays silent without note lanes.
- `VoiceAllocator` lanes make `WavetableLayer` produce a playable stereo voice.
- `EnvelopeAu/value -> voice_gain_audio` gives per-note articulation and release.
- Position and warp modulation change timbre rather than pitch.
- Extra per-note sources use their own reduction before final stereo mixing.
- Active production docs and examples lead with `WavetableLayer` / `LayerPad`.
- `WavetableOsc + VoiceMixer` appears as the advanced legacy wavetable path for excluded interaction/feedback behavior, while `VoiceMixer` remains valid for auxiliary per-voice layers.
