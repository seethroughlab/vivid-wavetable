## Goal

Make oscillator-to-oscillator interaction a reliable source of complexity and character instead of a fragile trick that only works in narrow settings.

This pass should make cross-modulation patches feel intentional and playable. The result should be new sound families that are hard to reach through static layering alone.

## Engine Changes

- Improve FM/RM/AM usability and level staging in `WavetableOsc` and `AnalogOsc`.
  - Calibrate modulation depth behavior so the useful range is easy to reach.
  - Avoid a design where most of the range is either too subtle or immediately unusable.
- Review cross-mod routing behavior in polyphonic graphs.
  - Ensure the current lane and channel handling remains stable when oscillators modulate one another.
  - Reduce the chance of runaway level behavior or brittle patch setup.
- Add 1-2 new musically useful interaction or warp modes only if they clearly unlock new patch families.
  - The standard for inclusion is audible usefulness, not novelty.
  - New modes should be understandable enough to use in real graphs.
- Keep the graph model as the modulation-routing surface.
  - Improve the operator behavior, not the overall package architecture.
  - Do not introduce a modulation-matrix system in this pass.

## Graph / Demo Work

- Add 4-5 graphs built around interaction rather than static stacking.
- The graph set must include:
  - FM glass keys,
  - controlled metallic lead,
  - motion-rich hybrid arp,
  - growl bass,
  - spectral texture patch.
- Each graph should prove that interaction-based richness is now more usable.
  - The patch should not require extreme gain trimming or obscure routing to stay musical.
  - The modulation depth should sit in a controllable range.
- Keep the graphs readable enough that they can serve as reference patches for later sound design.

## Tests and Listening

- Add regression tests for finite and stable output when cross-mod is enabled.
- Add coverage for the improved modulation-depth handling where practical.
- Run the standard package validation flow after engine and graph changes land.
- Audition the graphs across low, medium, and high interaction depth settings to ensure:
  - low depth is audible,
  - medium depth is musically useful,
  - high depth remains controlled enough to be intentionally usable.
- Specifically listen for:
  - runaway level jumps,
  - brittle alias-like harshness that is unrelated to the intended patch,
  - unstable polyphonic behavior when multiple voices interact.

## Acceptance Criteria

- Cross-mod patches are easier to dial in than before this pass.
- At least 3 new interaction-based graphs open clearly new sonic territory for the package.
- New interaction or warp modes, if added, each have a clear patch-family reason to exist.
- No new polyphonic instability or runaway level behavior is introduced.
- Build, test, and smoke validation continue to pass.

## Explicit Non-Goals

- No full modulation matrix.
- No replacement of the graph-routing model.
- No addition of exotic interaction modes that sound novel but are not musically useful.
- No overfitting the engine to a single demo patch.
