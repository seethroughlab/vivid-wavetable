# vivid-wavetable: Feature Gap Analysis vs. Commercial Synths

## Context

With role bindings shipped (Phases 1–6), vivid-wavetable can now bind external Envelope and LFO nodes into per-voice modulation roles. This opens a design space that didn't exist before: improvements to bindable operators benefit every host automatically, and new bindable operator types are immediately usable.

This document analyzes what's now easier to add, what's still missing compared to Serum / Pigments / Ableton Wavetable, and where the highest-leverage work is.

---

## Current State

**vivid-wavetable has:** 1 wavetable oscillator (9 built-in tables, 32-64 frames each), sub oscillator, noise oscillator, 8 warp modes, 8 filter types, 3 internal ADSR envelopes (amp/filter/position), unison (up to 16 voices, 3 spread modes), portamento, velocity sensitivity, 9 factory presets.

**5 role binding slots:** `amp_env`, `filt_env`, `pos_env` (per-voice, Envelope/LFO), `pitch_mod`, `wt_pos_mod` (per-voice, LFO/Envelope).

**Only 2 bindable operator types exist:** Envelope (linear ADSR) and LFO (sine/saw/square/tri, free or phase-synced).

---

## What Role Bindings Make Easier

### Tier 1: Enhance existing bindable operators (zero changes to WavetableSynth)

These are high-leverage because every host operator benefits automatically:

**1. MSEG (Multi-Segment Envelope Generator)**
- New bindable operator. Arbitrary breakpoint envelopes with per-segment curve types (linear, exp, log, S-curve). Loopable regions. This is Serum's most distinctive modulation feature.
- Binds to any existing envelope role (`amp_env`, `filt_env`, `pos_env`) with no WavetableSynth changes.
- Outputs `value` just like Envelope does.

**2. Enhanced LFO**
- Add waveform types: sample & hold, smooth random, noise, custom drawn.
- Add rate modes: Hz, beat-synced (1/4, 1/8, 1/16, dotted, triplet).
- Add unipolar/bipolar switch, phase offset, fade-in time.
- Existing LFO operator can be extended; all current bindings benefit.

**3. Envelope curve shapes**
- Add curve type per stage (exponential, logarithmic, S-curve) to the existing Envelope operator.
- Linear-only envelopes are the biggest audible gap vs. any commercial synth.

### Tier 2: New bindable operator types (minor WavetableSynth changes to add new roles)

**4. Step Sequencer modulator**
- New bindable control operator. 1-32 steps, per-step value, glide, gate length.
- Bind to `pitch_mod` for arpeggiated sequences, or `wt_pos_mod` for rhythmic wavetable sweeps.
- WavetableSynth already has `pitch_mod` and `wt_pos_mod` roles that accept LFO — adding StepSequencer to `allowed_operator_types` is a one-line change.

**5. Random / S&H modulator**
- New bindable operator. Timed random values with smoothing/slew. Useful for evolving pads.
- Binds to any modulation role.

**6. Macro operator**
- A trivial bindable operator: one knob (0-1), one output. But binding it to `wt_pos_mod` or `pitch_mod` gives the user a dedicated performance control that's visible on the canvas.
- This is how Serum's macro knobs work conceptually — a named control routed to destinations.

### Tier 3: New roles on WavetableSynth (requires WavetableSynth code changes)

**7. `filter_mod` role**
- Currently filter cutoff modulation comes only from the internal filter envelope or external `filter_env` spread port. Adding a dedicated `filter_mod` role (like `pitch_mod`) would let users bind an LFO to wobble the filter independently of the filter envelope.
- Small code change: read bound value, add to cutoff.

**8. `warp_mod` role**
- Modulate `warp_amount` from a bound operator. No current modulation path for this parameter.
- Enables evolving timbral effects (e.g., LFO sweeping sync amount).

**9. `pan_mod` role**
- Per-voice stereo position modulation. Useful for wide, moving pads.

---

## What Role Bindings Don't Help With

These are features that require changes to WavetableSynth itself or entirely new operators:

### Oscillator Architecture

**10. Second oscillator / oscillator layering**
- Serum has 2 full oscillators + sub + noise. Pigments has up to 4 engines.
- vivid-wavetable has 1 oscillator + sub + noise.
- Options: (a) add a second oscillator internally, (b) create an OscillatorLayer operator that hosts two WavetableSynths, (c) rely on the graph — users can already place two WavetableSynth nodes and mix them through a Mixer.
- The graph approach (c) already works today and is arguably more flexible.

**11. Custom wavetable import**
- Serum's ecosystem runs on user-created wavetables (.wav files, 256 or 2048 samples/frame).
- Would require: file parameter for .wav loading, runtime FFT mipmap generation, frame count detection.
- This is internal to WavetableSynth, no role binding involvement.

**12. Wavetable editor / spectral drawing**
- Serum allows drawing harmonics per frame. This is a major UI feature, not an operator concern.
- Long-term goal, but very high effort.

### Effects

**13. Built-in effects chain**
- Serum/Pigments include reverb, delay, chorus, phaser, distortion, EQ, compressor.
- vivid-wavetable has no post-processing effects.
- Options: (a) add effects internally (large scope), (b) create separate effect operators and wire them in the graph (already possible with any audio operator), (c) create a "channel strip" composite operator.
- The graph approach works today — this is a vivid strength. But having a curated effects chain inside the synth is a UX convenience.

### Modulation Matrix

**14. General-purpose modulation matrix**
- Serum/Pigments let you drag any source to any destination with an amount slider.
- Role bindings are named, fixed slots — not a free-form matrix.
- A full mod matrix would be a significant architecture addition (either internal to WavetableSynth, or a new vivid-wide feature).
- However: role bindings + the graph's wire system together cover most of the same ground, just with different UX.

---

## Priority Ranking (impact vs. effort)

| # | Feature | Effort | Impact | Where |
|---|---------|--------|--------|-------|
| 3 | Envelope curve shapes | Small | High | Envelope operator |
| 2 | Enhanced LFO (shapes, sync) | Medium | High | LFO operator |
| 1 | MSEG operator | Medium-Large | Very High | New operator |
| 11 | Wavetable import (.wav) | Medium | Very High | WavetableSynth |
| 7 | `filter_mod` role | Small | Medium | WavetableSynth |
| 8 | `warp_mod` role | Small | Medium | WavetableSynth |
| 4 | Step sequencer modulator | Medium | Medium | New operator |
| 6 | Macro operator | Small | Medium | New operator |
| 5 | Random/S&H modulator | Small | Medium | New operator |
| 9 | `pan_mod` role | Small | Low | WavetableSynth |
| 13 | Effects (graph-based) | Already works | — | — |
| 10 | Second oscillator | Large | Medium | WavetableSynth |
| 14 | Mod matrix | Very Large | High | Architecture |
| 12 | Wavetable editor | Very Large | High | UI |

---

## Recommended Sequence

**Phase A — Modulation quality (biggest audible improvement, leverages role bindings)**
1. Envelope curve shapes (exp/log/S-curve per stage)
2. Enhanced LFO (more shapes, beat sync, fade-in)
3. `filter_mod` + `warp_mod` roles on WavetableSynth

**Phase B — New modulation sources**
4. MSEG operator (new bindable type)
5. Random/S&H operator
6. Macro operator

**Phase C — Sound source expansion**
7. Wavetable import (.wav)
8. Step sequencer modulator

**Phase D — Polish**
9. More factory presets exploiting the new modulators
10. `pan_mod` role
