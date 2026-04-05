# Pass 4 — Asset-Driven Wavetable Workflow and Instrument Library

## Summary
This pass should make `vivid-wavetable` behave like a curated instrument package instead of only a collection of self-playing demo graphs. The core work is to adopt the shipped Vivid asset-library and instrument-browser metadata surfaces, move the package's custom wavetable story onto those surfaces, and publish a small set of instrument-ready graphs that are easier to browse and play.

Pass 4 should center on three outcomes:

- package-owned factory wavetable assets live in a real package asset root instead of being implied or ad hoc
- the package proves that factory package assets and imported workspace assets can use the same `WavetableOsc` custom-file path workflow
- the final retained library is split more clearly into:
  - instrument-ready graph entries
  - self-playing examples and module demos

This pass should not try to invent a new preset format, a wavetable editor, or a package-specific browser. It should adopt the current Vivid contracts as they already exist.

## Key Changes
### 1. Establish the package wavetable asset surface
Add a real factory wavetable asset root and declare it explicitly in the package manifest.

- add `assets/wavetables/`
- add an `assets` block to `vivid-package.json`:
  - `"assets": { "wavetables": ["assets/wavetables"] }`
- keep the asset kind limited to `wavetables`; do not add any broader asset taxonomy in this pass
- ship a compact factory asset set aimed at the final instrument roles, not a large sample dump

Use this factory set as the default target:

- `assets/wavetables/warm-keys-core.wav`
- `assets/wavetables/vocal-pad-sweep.wav`
- `assets/wavetables/glass-motion.wav`
- `assets/wavetables/rooted-bass-edge.wav`
- `assets/wavetables/texture-tide.wav`
- `assets/wavetables/bright-pluck-edge.wav`

These files should be the package's factory wavetable library for Pass 4. If an asset is not used by at least one retained instrument or reference graph, it should not ship in this pass.

### 2. Expose asset-backed wavetable selection on the module surface
Adopt the existing `WavetableOsc` custom-file workflow on the package modules where it materially helps instrument reuse.

Expose these new module params:

- `HybridKeys`
  - `wavetable_source`
  - `wavetable_file`
- `GlassInteractionKeys`
  - `wavetable_source`
  - `wavetable_file`
- `SubAirPad`
  - `wavetable_source`
  - `wavetable_file`
- `DualWavetablePad`
  - `osc_a_source`
  - `osc_a_file`
  - `osc_b_source`
  - `osc_b_file`

Binding rules:

- source params bind to the relevant internal `WavetableOsc/wavetable_source`
- file params bind to the relevant internal `WavetableOsc/wav_file`
- builtin family/member params stay in place and remain active when source is `Builtin`
- custom-file params are additive; do not remove the builtin wavetable path

Module defaults should remain builtin-first for safety and readability. Asset-backed behavior should be exercised by instrument graphs and curated examples, not by making every module default to a file-backed wavetable.

### 3. Create a real instrument-ready graph library
Keep the self-playing demo graphs, but stop treating them as the only browseable surface. Add a small set of graph entries whose purpose is "play this instrument," not "watch this demo run."

Add these instrument-ready graphs under `graphs/presets/`:

- `hybrid_keys_instrument.json`
  - built on `HybridKeys`
  - category: `Keys`
  - family: `Hybrid`
  - role: `reference`
  - playability: `midi`
- `glass_interaction_instrument.json`
  - built on `GlassInteractionKeys`
  - category: `Keys`
  - family: `Interaction`
  - role: `hero`
  - playability: `midi`
- `dual_wavetable_pad_instrument.json`
  - built on `DualWavetablePad`
  - category: `Pads`
  - family: `DualWavetable`
  - role: `hero`
  - playability: `midi`
- `sub_air_pad_instrument.json`
  - built on `SubAirPad`
  - category: `Pads`
  - family: `SubAir`
  - role: `reference`
  - playability: `midi`
- `rooted_sub_bass_instrument.json`
  - built from the retained sub-support bass voice pattern
  - category: `Bass`
  - family: `SubSupport`
  - role: `hero`
  - playability: `midi`
- `motion_texture_instrument.json`
  - built from the retained motion-heavy wavetable voice pattern
  - category: `Texture`
  - family: `Motion`
  - role: `utility`
  - playability: `midi`

Authoring rules for these graphs:

- set `meta.content_kind` to `instrument`
- fill `meta.category`, `meta.family`, `meta.role`, and `meta.playability`
- add `preview_controls` for 3-4 meaningful macro params only
- use `MidiInput` as the primary note-entry surface
- keep external FX light and readable
- prefer one instrument node named `instrument` when the graph is module-backed
- do not reuse the self-playing module demo graphs as the instrument library surface

Preview-control defaults:

- keys instruments: brightness, body, output
- pad instruments: motion, tone, blend/body, output
- bass instruments: bite/body, sub support, output
- interaction instrument: interaction depth, brightness, body, output
- motion instrument: motion amount, tone, output

### 4. Keep examples and instruments clearly separated
Do not mark every retained preset as an instrument. The browser distinction only helps if the package is selective.

Use these curation rules:

- module demo graphs remain `example` content or leave `content_kind` unset
- self-playing preset graphs remain examples unless they are explicitly rebuilt as midi-playable instruments
- each primary role should have:
  - one hero instrument
  - one reference instrument where it helps clarity
- avoid duplicate entries where a self-playing demo and a midi instrument are functionally the same experience

Retained example anchors should stay in place for listening and documentation:

- `moving_vocal_pad.json`
- `fm_glass_keys.json`
- `driven_hybrid_bass.json`
- `spectral_interaction_texture.json`
- the four existing module demo graphs

The new instrument graphs are the browseable "library" layer. The older self-playing graphs remain the "examples and references" layer.

### 5. Add asset-backed smoke and package validation
Pass 4 needs package-owned validation for both factory assets and graph metadata.

Add package tests and fixtures for:

- manifest assertions
  - `tests/test_package_manifest.cpp` should assert the `assets` block and `assets/wavetables`
- factory wavetable asset loading
  - add a package C++ test that enumerates `assets/wavetables/*.wav`
  - load each asset through `load_wavetable_from_wav()`
  - assert non-null, finite sampling, and at least one frame
- instrument metadata audit
  - add a package test that scans retained `content_kind="instrument"` graphs
  - require `category`, `family`, `role`, `playability`, and non-empty `preview_controls`
  - require `MidiInput` presence for `playability="midi"`
- graph smoke
  - add at least one asset-backed self-playing graph under `graphs/core/`
  - use package factory wavetable assets through `wavetable_source=Custom` and `wav_file`
  - keep MIDI-driven instrument graphs out of `graphs/core/` smoke coverage

Validation workflow for imported user assets should be explicit but non-committed:

- verify runtime `import_asset` succeeds for at least one external wavetable file
- verify the imported workspace asset appears in merged asset listing beside package assets
- verify one instrument graph or module can be retargeted to the imported asset by writing the canonical `wav_file` path
- do not commit workspace-imported asset paths into repo-tracked graphs

### 6. Refresh docs around the package's final library story
Update docs so the package explains the new split between assets, instruments, and examples.

Update:

- `README.md`
  - add `assets/wavetables/` to package layout
  - add a short `Instrument Library` section
  - explain factory package assets vs imported workspace assets
- `docs/showcase-library.md`
  - keep the retained example map
  - add a new instrument-library section listing the Pass 4 instrument graphs by role
- `vivid-package.json`
  - add a second guide entry for a new asset workflow doc
- add `docs/wavetable-asset-workflow.md`
  - explain:
    - package factory assets
    - user-imported workspace assets
    - `wavetable_source` + `wav_file`
    - which content is safe to commit and which is local-only
- `docs/expansion-april04/PASS-4.md`
  - store this pass doc as the authoritative breakdown

## Public Interfaces / Types
Public-facing additions in this pass should be:

- `vivid-package.json`
  - add `assets.wavetables`
- package filesystem
  - add `assets/wavetables/`
- module param surface
  - add source/file params on the four module types as listed above
- graph metadata adoption on selected library graphs
  - `meta.content_kind`
  - `meta.category`
  - `meta.family`
  - `meta.role`
  - `meta.playability`
  - `meta.preview_controls`

Public-facing behavior that should stay stable:

- `WavetableOsc` continues to persist custom selection through `wavetable_source` + `wav_file`
- graphs continue storing file paths, not `asset_id`
- modules remain additive wrappers over the package voice graphs
- self-playing examples remain available beside the new instrument entries

No custom wavetable editor.
No package-specific browser.
No new asset reference object.
No attempt to make every retained graph instrument-tagged.

## Test Plan
Baseline validation remains required:

- configure and build package operators successfully
- run package tests successfully
- run `vivid` link / rebuild / uninstall successfully
- run `test_demo_graphs` against `graphs/core/` successfully

Pass-specific validation:

- manifest parsing recognizes the package `assets.wavetables` declaration
- every factory wavetable asset loads successfully through the package wavetable loader
- at least one `graphs/core/` smoke graph proves a package factory asset can drive `WavetableOsc` through the custom-file path
- instrument graphs with `content_kind="instrument"` have complete browser metadata and valid preview-control references
- the package proves both scopes in practice:
  - package read-only wavetable assets
  - imported workspace wavetable assets
- module-backed instrument graphs load without missing param, missing module, or missing asset failures after link/rebuild

Listening acceptance:

- `hybrid_keys_instrument` reads as a straightforward playable keys instrument
- `dual_wavetable_pad_instrument` feels like a coherent playable pad, not a demo wrapper
- `rooted_sub_bass_instrument` stays grounded and controlled
- `glass_interaction_instrument` preserves the metallic/interaction identity while feeling operable
- `motion_texture_instrument` keeps audible movement without turning into a cluttered showcase patch
- the final browseable library feels curated, with obvious separation between playable instruments and self-playing examples

## Assumptions and Defaults
- Pass 4 builds on the assumption that Passes 1-3 have established the module surface and performance conventions already intended by the April 4 roadmap
- committed package content should reference package-shipped factory assets only; imported workspace assets are for validation and user workflow, not repo-tracked defaults
- module defaults remain builtin-first; asset-backed behavior is shown through curated graph content
- the final library should stay compact; do not create instrument duplicates for every retained example
- chosen default taxonomy:
  - `content_kind`: `instrument` for the new library graphs, otherwise example/unset
  - `role`: `hero`, `reference`, `utility`
  - `playability`: `midi` for the new instrument set
- chosen default asset scope policy:
  - package assets are read-only and declared in the manifest
  - user assets are imported into the workspace library and consumed through canonical `wav_file` paths
