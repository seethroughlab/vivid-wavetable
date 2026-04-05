# Pass 3 — Module-Local Modulation, Performance Pages, and Expressive Play

## Summary
This pass should make the Pass 1 module surface feel playable and instrument-like instead of only compact. The goal is to adopt the shipped Vivid performance-page metadata, module-local modulation surface, and expressive `MidiInput` lanes in a way that gives `vivid-wavetable` one small, coherent live-control vocabulary.

Pass 3 should standardize five package performance roles:

- `motion`
- `brightness`
- `air`
- `body`
- `interaction`

Not every module needs every role. But when one of these roles appears, it should mean the same musical thing across the package.

This pass is not a full modulation matrix expansion. It is a package adoption pass that uses the current core surfaces to make the existing module instruments easier to play, easier to browse, and easier to perform.

## Key Changes
### 1. Establish one package performance-surface convention
Use the current exposed-param metadata instead of inventing a second control model.

- Tag selected exposed params with `performance_page`, `performance_order`, and `performance_role`.
- Use two pages consistently:
  - `Performance` for the live macro layer
  - `Timbre` for the main tone/brightness layer
- Use `performance_role = "macro"` for the package's named live controls.
- Use `performance_role = "expression"` only for the primary exposed timbre control when that makes the surface read more clearly.
- Keep the current compact module surfaces, but converge the live-facing param names on the canonical role names where the current names are too implementation-specific.

Canonical meanings for this pass:

- `motion` = wavetable travel or movement depth
- `brightness` = top-end openness / main tone opening
- `air` = upper noise, shimmer, or breath support
- `body` = low-mid weight, drive, or glue
- `interaction` = carrier/modulator complexity amount

### 2. Retrofit the four existing modules around those roles
Apply the convention to the current Pass 1 module set instead of adding new module types.

- `DualWavetablePad`
  - make this the clearest `motion` instrument
  - expose `motion` as the main live macro
  - expose `brightness` as the main timbre control
  - keep deeper layer-selection controls available but off the primary performance page
- `HybridKeys`
  - make this the clearest `brightness + body` instrument
  - expose `brightness` as the main live tone control
  - expose `body` as the main weight/glue control
  - keep `analog_blend` as a secondary voice-shaping control, not a headline performance macro
- `SubAirPad`
  - make this the clearest `air` instrument
  - expose `air` as the main live macro
  - expose `brightness` as the main timbre control
  - keep the sub support readable and subordinate to the main body/air story
- `GlassInteractionKeys`
  - make this the clearest `interaction` instrument
  - expose `interaction` as the main live macro
  - expose `brightness` and `body` as the supporting performance controls
  - keep the patch's PM/glass identity stable while making the live control surface more obvious

The result should be that the package has four existing instrument wrappers, each with a different headline role, but all speaking the same small performance language.

### 3. Adopt module-local modulation properly
Use the current core modulation surface the way it was intended: named module `mod_sources` and `mod_destinations` in the module definitions, with authored `mod_assignments` on selected module instances in graphs.

Module definition changes:

- Add compact `mod_destinations` to the modules using the same role vocabulary as the performance surface:
  - `motion`
  - `brightness`
  - `air`
  - `body`
  - `interaction`
  - only declare the destinations that make musical sense for that module
- Add only a few `mod_sources` per module:
  - internal movement sources where the module already has an obvious internal motion block
  - port-based expressive sources only where they are musically justified
- Do not expose raw internal node params such as filter helpers, intermediate envelope amounts, or oscillator implementation details as public modulation destinations

Adoption targets:

- `DualWavetablePad`
  - declare the internal motion LFO as a named modulation source
  - use instance-local `mod_assignments` so one motion source can animate both wavetable travel and a small amount of tone movement without exposing extra internals
- `GlassInteractionKeys`
  - declare expressive port-based modulation sources for the first real expressive-play path
  - use instance-local `mod_assignments` so expressive input can move `interaction` and `brightness` without reopening the module internals
- `HybridKeys` and `SubAirPad`
  - add only the minimum named destinations needed to align with the package macro vocabulary in this pass
  - do not force a larger modulation design onto them if the performance-page surface already covers the musical need

### 4. Add one focused expressive-play path
Adopt the new expressive `MidiInput` lane outputs through one clear package example instead of trying to retrofit everything at once.

- Add one new interactive example graph in `graphs/presets/` centered on `GlassInteractionKeys`
- Use `MidiInput`'s shipped expressive lane names directly:
  - `pressures`
  - `slides`
  - optionally `expressions` if the patch benefits from it
- Keep the expressive module contract aligned with core naming and ranges; do not invent package-specific expressive port names
- Use expressive input to drive musically obvious destinations:
  - pressure increases `interaction`
  - slide or expression opens `brightness`
- Keep the example obviously performance-oriented and not dependent on hidden graph complexity

This graph should be an interactive example, not a headless smoke fixture.

### 5. Refresh the module demos and package docs around the new surface
Update the existing module demo graphs so the first-load experience matches the Pass 3 control story.

Graph updates:

- retune `dual_wavetable_pad_module_demo.json` so `motion` is immediately obvious
- retune `hybrid_keys_module_demo.json` so `brightness` and `body` are the useful first controls
- retune `sub_air_pad_module_demo.json` so `air` is audible but still integrated
- retune `glass_interaction_keys_module_demo.json` so `interaction` reads as a performance control rather than a hidden implementation detail
- add one new `MidiInput`-driven expressive demo in `graphs/presets/`

Doc updates:

- `README.md`
  - add a short `Performance Surface` section explaining the package role vocabulary
  - document that not every module exposes every role, but the roles keep shared meanings
  - mention the new expressive demo as an interactive example
- `docs/showcase-library.md`
  - annotate which module demos are the Pass 3 performance anchors
- `docs/expansion-april04/PASS-3.md`
  - capture this pass breakdown in the same style as `PASS-1.md` and `PASS-2.md`

## Public Interfaces / Types
Public-facing changes in this pass should be:

- selected module exposed params gain `performance_page`, `performance_order`, and `performance_role` metadata
- selected module exposed params converge on the package's canonical live-control names:
  - `motion`
  - `brightness`
  - `air`
  - `body`
  - `interaction`
- selected modules gain compact `mod_sources` and `mod_destinations` definitions
- selected module demo graph instances start using authored `mod_assignments`
- one expressive module path begins consuming the shipped expressive `MidiInput` lane outputs using the exact core port names

Interfaces that should stay stable:

- the four existing module types remain the same
- the current note / velocity / gate instrument contract remains intact
- no new package operators are introduced
- no package-specific performance-state model is introduced
- no custom expressive transport is introduced outside the current `MidiInput` contract

## Test Plan
Baseline validation remains required:

- configure and build package operators successfully
- run package tests successfully
- run `vivid` link / rebuild / uninstall successfully
- run `test_demo_graphs` against `graphs/core/` successfully

Pass-specific validation:

- module files load with valid `performance_page` / `performance_role` metadata and valid `mod_sources` / `mod_destinations`
- updated module demo graphs compile and load cleanly after the surface rename/tagging work
- module-instance `mod_assignments` in the selected demo graphs serialize and reload cleanly
- the live-control vocabulary remains compact and readable in the module inspector surface
- the expressive demo graph stays in `graphs/presets/` and is not promoted into `graphs/core/`

Listening acceptance:

- `DualWavetablePad` has an obvious and useful `motion` control
- `HybridKeys` has clearly different `brightness` and `body` moves
- `SubAirPad` has an `air` control that feels integrated rather than detached hiss
- `GlassInteractionKeys` has an `interaction` control with a readable sweet spot
- the expressive demo's pressure/slide behavior is musically useful and stable, not merely technically wired

Operational rule for validation:

- treat the `MidiInput` expressive demo as an interactive example, not a headless smoke fixture
- keep headless smoke focused on smoke-safe `graphs/core/` plus any non-interactive module demos that still compile cleanly without live MIDI input

## Assumptions and Defaults
- Pass 3 builds on the existing Pass 1 module set already present in the package
- the package should use the current core performance metadata as the UI surface, not invent a second macro system
- module-local modulation means module `mod_sources` / `mod_destinations` plus instance-level `mod_assignments`, not reopened module internals
- only one focused expressive-play path is required in this pass; broad expressive retrofits are deferred
- the package keeps the current curated scope and avoids turning performance adoption into a preset-sprawl pass
- MIDI-driven expressive demos remain interactive examples under `graphs/presets/`, not `graphs/core/` smoke coverage
