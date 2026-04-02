## Goal

Broaden the source material and make a single `WavetableOsc` feel more alive before heavy FX or complicated graph layering.

This pass should meaningfully raise the ceiling of the core wavetable source. The result should be that a simple graph with one `WavetableOsc`, sensible envelopes, and light FX can already produce more interesting sustained tones than it can today.

## Engine Changes

- Expand the built-in wavetable library substantially beyond the current small bank.
- Organize the new wavetable content into tonal families with clear listening intent:
  - analog warm,
  - bright digital,
  - vocal/formant,
  - metallic,
  - harmonic/spectral,
  - texture/noise-adjacent.
- Keep the wavetable-bank implementation internal to the existing bank module.
  - Do not create a new oscillator type for the expanded content.
  - Preserve the current package/operator identity.
- Improve `WavetableOsc` motion controls so sustained sounds feel more alive:
  - add note-start phase behavior,
  - add random phase per note,
  - add slow drift or slop per voice,
  - add smoothing for wavetable position motion,
  - add smoothing for warp modulation where needed,
  - improve stereo phase/spread behavior so width sounds lush instead of comb-filtered.
- Keep these additions coherent as part of `WavetableOsc`.
  - Do not fork the operator into a separate “advanced” version.
  - Prefer a small number of well-named, high-value controls over a large number of niche parameters.
- Make note-start behavior safe.
  - New phase randomization or start-phase options must not introduce obvious clicks at note onset.

## Graph / Demo Work

- Add 4-6 new graphs specifically designed to prove the new wavetable content and motion depth.
- The listening set must include:
  - moving vocal pad,
  - supersaw-like pad,
  - evolving texture bed,
  - animated pluck,
  - bright modern lead.
- The graphs should isolate the benefit of the new source engine.
  - Use light FX where needed, but do not hide the oscillator behind oversized reverb or chorus.
  - Make the motion audible from the raw or near-raw oscillator path.
- Reuse the graph conventions established in Pass 1 so the library still feels consistent.
- Where useful, include A/B-style graphs or simplified references that show one oscillator carrying more of the sound on its own.

## Tests and Listening

- Add tests for wavetable-bank behavior:
  - generated tables initialize correctly,
  - family expansion does not break sampling behavior,
  - wavetable import/build paths remain stable.
- Add tests for the new motion parameters:
  - phase-related options stay finite,
  - drift/slop does not produce unstable output,
  - smoothing behaves continuously under movement.
- Run the standard package validation flow after the engine work lands.
- Audition the new graphs with two listening modes:
  - sustained held notes to judge living motion and stereo width,
  - modulation sweeps to judge smoothness and click resistance.

## Acceptance Criteria

- The expanded wavetable families are audibly distinct and useful as separate tonal starting points.
- A single `WavetableOsc` can now produce richer sustained sounds before chorus/reverb-heavy graph dressing.
- New motion controls add audible depth without making the operator fragile or confusing.
- No phase/randomization option introduces obvious clicks on note start.
- The package still passes build, test, and smoke coverage.

## Explicit Non-Goals

- No separate `AdvancedWavetableOsc` or parallel oscillator fork.
- No full modulation matrix.
- No new voice-character operators yet.
- No graph work whose main purpose is to mask weak source material with oversized FX.
