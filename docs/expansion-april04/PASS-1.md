# Pass 1 — Module-First Instrument Architectures

> Historical note: this pass document records the April 4 adoption plan. The current package now includes the later WavetableLayer / LayerPad production path; use the README and active guides for current user-facing behavior.

## Summary
This pass is the first real package adoption of Vivid's subgraph-module system. The goal is to make `vivid-wavetable` feel less like a set of operators plus glue and more like a small family of coherent instruments.

This pass will add four package-owned modules:

- `HybridKeys`
- `DualWavetablePad`
- `SubAirPad`
- `GlassInteractionKeys`

The existing operators remain the internal sound engine. Modules are an instrument-facing wrapper over the current graphs and operators.

## Key Changes
### 1. Add the first package module surface
- Create a new top-level `modules/` directory.
- Add four `.vivid-module.json` files.
- Add a `modules` array to `vivid-package.json`.
- Keep modules as an additional package surface beside operators and graphs.

Module targets:

- `modules/hybrid_keys.vivid-module.json`
  - module type: `HybridKeys`
  - source voice reference: `aurora_hybrid_keys.json`
  - role: easiest finished-voice keys instrument in the package

- `modules/glass_interaction_keys.vivid-module.json`
  - module type: `GlassInteractionKeys`
  - source voice reference: `fm_glass_keys.json`
  - role: coherent instrument-facing wrapper for the interaction/glass architecture

- `modules/dual_wavetable_pad.vivid-module.json`
  - module type: `DualWavetablePad`
  - new module from scratch
  - tonal references: `moving_vocal_pad.json` and `supersaw_fabric_pad.json`
  - role: canonical dual-wavetable pad architecture missing from the retained library

- `modules/sub_air_pad.vivid-module.json`
  - module type: `SubAirPad`
  - new module assembled from package patterns
  - tonal references: `rooted_sub_pad.json`, `ambient_air_bed.json`, and/or `airy_keys.json`
  - role: canonical wavetable + sub + air architecture

### 2. Standardize the v1 module boundary
Use these defaults for all four modules:

- external inputs:
  - `notes`
  - `velocities`
  - `gates`
- external output:
  - `output`
- internal graph should contain:
  - `VoiceAllocator`
  - package voice/layer operators
  - per-note envelopes
  - `VoiceMixer`
  - filter/drive/noise/sub blocks as needed
- internal graph should not contain:
  - `ClockAu`
  - `ChordProgressionAu`
  - `NotePatternAu`
  - large time-based FX unless essential to core identity

The modules represent the **instrument voice**, not the self-playing demo patch.

### 3. Exposed control-surface conventions
Each module should expose a compact grouped surface, not every internal param.

Use these standard groups:

- `Osc A`
- `Osc B` or `Layer`
- `Tone`
- `Motion`
- `Character`
- `Output`

Intended exposed surface per module:

- `HybridKeys`
  - wavetable family/member/position
  - analog blend
  - filter cutoff/resonance
  - envelope character
  - light drive
  - output level

- `GlassInteractionKeys`
  - wavetable family/member/position
  - modulator amount
  - `interaction_depth`
  - brightness/filter controls
  - drive/body
  - output level

- `DualWavetablePad`
  - both wavetable families/members/positions
  - blend
  - motion amount
  - filter tone
  - width/unison
  - output level

- `SubAirPad`
  - wavetable family/member/position
  - sub level
  - air/noise level
  - filter tone
  - body/glue
  - output level

Pass 1 defaults:

- use grouped exposed params
- do not use `performance_page` or `performance_role` yet
- do not use module-local `mod_assignments` yet
- do not expose note-lane plumbing

### 4. Add companion module demo graphs
Add these package graphs:

- `graphs/presets/hybrid_keys_module_demo.json`
- `graphs/presets/glass_interaction_keys_module_demo.json`
- `graphs/presets/dual_wavetable_pad_module_demo.json`
- `graphs/presets/sub_air_pad_module_demo.json`

Each demo graph should:

- use `ClockAu` + `ChordProgressionAu` or `NotePatternAu` outside the module
- connect into the module `notes/velocities/gates`
- route module `output` to `audio_out`
- keep external FX minimal
- exist to prove the module is easier to use than the raw source graph

The original plain-graph reference patches stay in place in this pass.

### 5. Package docs and discoverability
Update:

- `README.md`
  - add `modules/` to package layout
  - add a short `Module Instruments` section listing the four module types
- `docs/showcase-library.md`
  - add a note that Pass 1 introduces the first instrument-facing wrappers
- `tests/test_package_manifest.cpp`
  - extend assertions for the new `modules` field and all four module paths

## Public Interfaces / Types
Public additions in this pass:

- manifest-level `modules` array in `vivid-package.json`
- new package content type: `.vivid-module.json`
- new module types:
  - `HybridKeys`
  - `GlassInteractionKeys`
  - `DualWavetablePad`
  - `SubAirPad`

No operator API changes.
No graph-schema changes.
No lane-architecture changes.

## Test Plan
- package configure/build stays green
- package tests stay green
- `vivid` link/rebuild/uninstall stays green
- `test_demo_graphs` on `graphs/core/` stays green
- after link/rebuild, verify the four module types appear in the Vivid type catalog/query surface
- each module demo graph loads and compiles without missing-module placeholders
- include at least 2-3 module demo graphs in focused headless smoke

Listening acceptance:

- `HybridKeys` reads as a usable keys instrument
- `DualWavetablePad` sounds like a coherent dual-wavetable pad
- `SubAirPad` feels like one voice with low support and air
- `GlassInteractionKeys` preserves the current PM/interaction identity while becoming easier to operate

## Assumptions and Defaults
- implement the full four-pattern set in Pass 1
- two modules are direct adoptions, two are new canonical instruments
- modules wrap the instrument voice only
- Pass 1 defers `performance_page` and `mod_assignments` adoption to Pass 3
- existing retained showcase graphs remain in place beside the new module content
