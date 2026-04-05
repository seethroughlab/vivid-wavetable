# Showcase Library

`vivid-wavetable` now uses a curated preset library instead of carrying every expansion experiment forward.

Pass 1 of the April 4 roadmap also introduces the package's first module-backed instruments. Those modules are additive instrument-facing wrappers; the curated preset family map below still refers to the retained plain-graph showcase library.

The goal of this set is simple:

- every retained graph has a clear role
- every major engine improvement from Passes 1-4 is audible in at least one hero patch
- every important feature also has a simpler reference patch nearby

## Family Map

### Pads and Beds

- Hero: `moving_vocal_pad.json`
- Reference: `warm_dual_pad.json`
- Also retained: `ambient_air_bed.json`, `cinematic_haze_bed.json`, `rooted_sub_pad.json`, `supersaw_fabric_pad.json`

### Keys and Brass

- Hero: `fm_glass_keys.json`
- Reference: `airy_keys.json`
- Also retained: `aurora_hybrid_keys.json`, `dream_keys.json`, `halo_hybrid_brass.json`, `low_brass_pad.json`

### Plucks and Bells

- Hero: `noisy_pluck.json`
- Reference: `mirror_bells.json`
- Also retained: `animated_neon_pluck.json`, `metallic_pluck.json`, `resin_mallet.json`

### Leads

- Hero: `controlled_metallic_lead.json`
- Reference: `analog_stack_lead.json`
- Also retained: `supersaw_lead.json`, `sync_sweep_lead.json`

### Basses

- Hero: `growl_crossmod_bass.json`
- Reference: `rooted_sub_bass.json`
- Also retained: `driven_hybrid_bass.json`, `sub_bass.json`

### Textures and Drones

- Hero: `spectral_interaction_texture.json`
- Reference: `single_osc_motion_reference.json`
- Also retained: `orbit_drone.json`, `texture_tide_bed.json`

### Arp and Sequence

- Hero: `hybrid_motion_arp.json`
- Reference: `crystal_pattern_arp.json`

### Cinematic Hybrids

- Hero: `cinematic_haze_bed.json`
- Reference: `ambient_air_bed.json`
- Also retained: `formant_choir.json`

## DualFilter Recipe Anchors

The April 4 Pass 2 adoption establishes `DualFilter` as the package's advanced tone-shaping primitive. These graphs serve as the canonical filter recipe anchors:

- **Warm body:** `warm_dual_pad.json`, `dual_wavetable_pad_module_demo.json`
- **Bright keys / brass contour:** `airy_keys.json`, `halo_hybrid_brass.json`, `hybrid_keys_module_demo.json`
- **Bass bite:** `driven_hybrid_bass.json`
- **Vocal / formant-adjacent:** `moving_vocal_pad.json`, `formant_choir.json`

## Pass 3 Performance Anchors

The April 4 Pass 3 adoption establishes the package performance-surface convention. These module demos are the canonical performance-surface anchors:

- **Motion headline:** `dual_wavetable_pad_module_demo.json`
- **Brightness + body headline:** `hybrid_keys_module_demo.json`
- **Air headline:** `sub_air_pad_module_demo.json`
- **Interaction headline:** `glass_interaction_keys_module_demo.json`
- **Expressive play:** `expressive_glass_keys.json` (scalar pressure/slide control)

Those role labels are the live-control vocabulary. The shipped module param names stay on their stable compatibility surface, including `filter_cutoff`, `drive`, `air_level`, `filter_tone`, `motion_amount`, and `interaction_depth`.

## Feature Coverage

### Pass 1 — Baseline, Leveling, Unison

- Hero: `supersaw_fabric_pad.json`
- Reference: `warm_dual_pad.json`

### Pass 2 — Wavetable Families and Motion / DualFilter Adoption

- Hero: `moving_vocal_pad.json`
- Reference: `single_osc_motion_reference.json`
- DualFilter anchors: `warm_dual_pad.json`, `airy_keys.json`, `halo_hybrid_brass.json`, `driven_hybrid_bass.json`, `formant_choir.json`

### Pass 3 — NoiseLayer, VoiceDrive, VoiceMixer Glue, Improved SubOsc

- Hero: `driven_hybrid_bass.json`
- Reference: `airy_keys.json`

### Pass 4 — Interaction Redesign

- Hero: `spectral_interaction_texture.json`
- Reference: `fm_glass_keys.json`
- Additional keepers: `controlled_metallic_lead.json`, `growl_crossmod_bass.json`, `hybrid_motion_arp.json`

## Retained Preset Set

The final retained preset library is:

- `airy_keys.json`
- `ambient_air_bed.json`
- `analog_stack_lead.json`
- `animated_neon_pluck.json`
- `aurora_hybrid_keys.json`
- `cinematic_haze_bed.json`
- `controlled_metallic_lead.json`
- `crystal_pattern_arp.json`
- `dream_keys.json`
- `driven_hybrid_bass.json`
- `fm_glass_keys.json`
- `formant_choir.json`
- `growl_crossmod_bass.json`
- `halo_hybrid_brass.json`
- `hybrid_motion_arp.json`
- `low_brass_pad.json`
- `metallic_pluck.json`
- `mirror_bells.json`
- `moving_vocal_pad.json`
- `noisy_pluck.json`
- `orbit_drone.json`
- `resin_mallet.json`
- `rooted_sub_bass.json`
- `rooted_sub_pad.json`
- `single_osc_motion_reference.json`
- `spectral_interaction_texture.json`
- `sub_bass.json`
- `supersaw_fabric_pad.json`
- `supersaw_lead.json`
- `sync_sweep_lead.json`
- `texture_tide_bed.json`
- `warm_dual_pad.json`

Everything else in the old expansion library was retired because it was redundant, weaker than a nearby patch, or no longer necessary after the engine work in Passes 1-4 settled.

## Instrument Library

Pass 4 adds a browseable instrument library separate from the self-playing showcase examples above. Instrument graphs use `MidiInput`, carry `content_kind: instrument` metadata, and are intended as the "play this instrument" layer.

### Keys

- Hero: `glass_interaction_instrument.json` — interactive glass keys with pressure mapping
- Reference: `hybrid_keys_instrument.json` — dual-layer wavetable + analog keys

### Pads

- Hero: `dual_wavetable_pad_instrument.json` — layered dual-wavetable pad with shared motion
- Reference: `sub_air_pad_instrument.json` — wavetable + sub + air pad

### Bass

- Hero: `rooted_sub_bass_instrument.json` — grounded sub-layered bass

### Pluck

- Hero: `bright_pluck_instrument.json` — crisp bell-adjacent pluck with a short spatial tail

### Lead

- Hero: `metallic_hollow_lead_instrument.json` — focused metallic lead with controlled edge

### Texture

- Utility: `motion_texture_instrument.json` — LFO-driven motion texture bed
