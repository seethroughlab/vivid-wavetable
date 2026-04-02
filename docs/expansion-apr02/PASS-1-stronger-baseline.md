## Goal

Raise the quality floor of the current engine without changing the package shape, and produce the first clearly better listening set quickly.

This pass is intentionally conservative. The work should make the existing `WavetableOsc`, `AnalogOsc`, `SubOsc`, and `VoiceMixer` sound smoother and more mixable before we introduce any new operator types or broaden the wavetable library.

## Engine Changes

- Add smoothing on modulation paths in `WavetableOsc` where abrupt changes currently sound steppy, brittle, or zippery.
- Keep the smoothing targeted and lightweight.
  - Prioritize wavetable position, warp amount, and any per-voice modulation path that is currently most likely to click or spit under slow sweeps.
  - Do not add a large set of user-facing smoothing controls in this pass.
- Audit the default gain staging across `WavetableOsc`, `AnalogOsc`, `SubOsc`, and `VoiceMixer`.
  - Retune defaults so common layered graphs land closer to a usable level without immediate clipping or harshness.
  - Preserve backward-compatible parameter ranges and operator identities.
- Review unison behavior in `WavetableOsc`.
  - Retune defaults for `unison_voices`, `unison_spread`, `unison_stereo`, and spread mode behavior so wider settings sound smoother and less metallic.
  - Do not redesign unison architecture yet; this pass is about better defaults and safer behavior.
- Keep the code changes parameter-light.
  - Favor internal improvements and better defaults over new public knobs.
  - If a new parameter feels necessary, it must solve a concrete audible issue that cannot be handled internally.

## Graph / Demo Work

- Add 6-8 new listening graphs under `graphs/presets/` or `graphs/extended/`.
- Do not expand `graphs/core/` beyond smoke-safe content.
- The new graph set must include:
  - 2 wavetable + analog hybrids,
  - 2 wavetable + sub foundations,
  - 1 animated pad,
  - 1 arp or sequence graph that uses `vivid-sequencers`,
  - 1 cinematic texture graph.
- Use these graphs to establish the baseline patch language for later passes:
  - layered oscillators instead of single-oscillator recipes,
  - deliberate envelope motion,
  - controlled chorus/delay/reverb use,
  - stable gain staging that does not rely on last-minute output attenuation.
- Name the graphs by sound family rather than by implementation detail so later showcase curation is easier.
- Keep every graph easy to audition.
  - Prefer self-playing or sequenced demos when that improves listening.
  - If a graph is interactive-first, place it outside smoke coverage and document the intended interaction.

## Tests and Listening

- Run the package validation flow unchanged:
  - configure + build package operators,
  - run package tests,
  - run the `vivid` link/rebuild/uninstall cycle,
  - run `test_demo_graphs` against `graphs/core/`.
- Add at least one focused regression test for one of:
  - modulation smoothing behaving continuously under parameter sweeps,
  - safer default gain behavior in a dense layered patch path.
- Audition the new graphs specifically for:
  - harshness in upper harmonics,
  - clicks on note start or modulation transitions,
  - brittle stereo spread at medium/high unison settings,
  - level jumps when layering oscillators.
- Treat listening as a gate, not a nice-to-have.
  - If a graph is technically correct but not musically compelling, it is not complete.

## Acceptance Criteria

- At least 3 of the new graphs are clear keeper sounds worth carrying forward into the final showcase library.
- The package still passes the existing build, test, and smoke workflow.
- The engine changes do not add new instability or confusing operator surface area.
- The new graphs sound clearly richer than the current one-oscillator-heavy baseline.
- Unison-heavy sounds are smoother and easier to use than before this pass.

## Explicit Non-Goals

- No new operator types.
- No major wavetable-bank expansion.
- No modulation-matrix work.
- No attempt to solve every richness gap in one pass.
- No promotion of expressive listening graphs into `graphs/core/` unless they are also smoke-safe.
