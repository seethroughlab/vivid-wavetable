# Pass 2 — DualFilter Adoption and Tone-Shaping Recipes

> Historical note: this pass document records the April 4 adoption plan and completion notes. The current package now includes the later WavetableLayer / LayerPad production path; use the README and active guides for current user-facing behavior.

**Status: Completed 2026-04-05**

### Completion Summary

DualFilter was adopted in three of four instrument modules and six preset anchor graphs:

- **DualWavetablePad** — topology restructured to pre-mix both oscillator layers through a single DualFilter (Serial A→B: LP24 body + LP12 top rolloff). Removed separate mix_b and post-filter sum nodes. This is the canonical warm body recipe.
- **HybridKeys** — Filter swapped to DualFilter (Serial A→B: LP24 + HP12 rumble cut). Topology unchanged. Bright keys contour recipe.
- **SubAirPad** — Filter swapped to DualFilter (Parallel: LP24 body + BP clarity band). Topology unchanged. Body + air recipe.
- **GlassInteractionKeys** — kept on plain Filter by explicit decision. The interaction identity is better served by the single-stage path.

Preset anchors retuned: warm_dual_pad, airy_keys, halo_hybrid_brass, driven_hybrid_bass, moving_vocal_pad (DualFilter inserted — previously filterless), formant_choir.

Module param names and external ports remain unchanged. No new module types. No new user-facing filter pages.

---

## Summary
This pass should make `vivid-wavetable` sound more like a finished instrument library by selectively adopting `DualFilter` in the places where a two-stage tone path actually improves the voice. The goal is not to replace every existing `Filter` node. The goal is to establish a small set of package filter recipes, prove them in real module and preset content, and keep the public instrument surfaces compact.

Pass 2 should center on four recipe families:

- warm subtractive body
- bright animated keys / brass contour
- bass bite with controlled upper edge
- vocal or formant-adjacent movement where it helps a voice read more musically

## Key Changes
### 1. Adopt `DualFilter` selectively instead of globally
Use this decision rule across the package:

- adopt `DualFilter` when a voice needs separate body and edge shaping, staged contour, or more intentional vocal-ish movement
- keep plain `Filter` where one filter remains clearer, smaller, and easier to tune
- treat `GlassInteractionKeys` as the control case: keep its current single-filter path unless A/B listening proves `DualFilter` improves brightness control without blurring the interaction identity

### 2. Retrofit the most relevant module internals
Update these module internals first:

- `DualWavetablePad`
  - become the canonical warm/body-first `DualFilter` module
  - use the new filter path to glue the two wavetable layers into one pad voice instead of sounding like two independent layers
- `HybridKeys`
  - adopt a brighter, more articulated keys contour recipe
  - use `DualFilter` only if it makes the keys voice more compact and instrument-like than the current wavetable-plus-analog stack
- `SubAirPad`
  - adopt a body-plus-air recipe where the main wavetable body gets the richer filter treatment and the sub / air layers remain simple support elements
- `GlassInteractionKeys`
  - stay on plain `Filter` by default in this pass
  - only switch if listening tests show a real gain in controllable brightness without losing attack clarity

Module boundaries stay the same:

- no new module types
- no new external ports
- keep current exposed param names stable where possible
- do not add new user-facing filter pages or large second-filter control surfaces in this pass
- remap existing tone / cutoff / body controls onto the adopted internals rather than expanding the UI

### 3. Define the package's first filter recipe anchors
Use existing curated content as the primary recipe anchors instead of creating a large parallel demo set:

- warm pad anchor: `warm_dual_pad.json` plus `dual_wavetable_pad_module_demo.json`
- bright keys / brass anchor: `airy_keys.json`, `halo_hybrid_brass.json`, and `hybrid_keys_module_demo.json`
- bass bite anchor: `driven_hybrid_bass.json`
- vocal / formant-adjacent anchor: `moving_vocal_pad.json` or `formant_choir.json`

Pass 2 should prefer retuning these retained graphs and module demos over adding many new presets. Add a new focused graph only if one of the recipe families cannot be demonstrated cleanly with the current curated set.

### 4. Tighten the package tone-shaping conventions
Document one compact package rule set for filter-forward instrument voicing:

- pads should read as one blended body, not as stacked oscillators with a filter pasted on top
- keys and brass should have a readable attack-to-body contour
- basses should keep weight centered while letting upper bite stay controlled
- vocal-ish movement should remain supportive and musical, not turn into a separate experimental effect path

This pass should also refresh the affected module demo defaults so the first-load experience reflects the new tone story without requiring extra tweaking.

### 5. Update docs without expanding the surface area
Update package docs to match the adoption:

- `README.md`
  - keep the existing single-`Filter` beginner build path
  - add a short note that `DualFilter` is the package's advanced tone-shaping option for selected finished voices
- `docs/showcase-library.md`
  - annotate which retained graphs now act as the filter recipe anchors for Pass 2
- `docs/expansion-april04/PASS-2.md`
  - capture this plan as the authoritative pass breakdown

## Public Interfaces / Types
Public-facing changes in this pass should stay intentionally small:

- module external ports remain unchanged
- current module type names remain unchanged
- current exposed module param names should remain stable unless a rename is required for clarity
- module defaults and internal bindings may change to reflect the new filter recipes
- some retained preset graphs and module demo graphs will be retuned around `DualFilter`

No new operators.
No new package content categories.
No performance-page or `mod_assignments` adoption yet.

## Test Plan
Baseline validation remains required:

- configure and build package operators successfully
- run package tests successfully
- run `vivid` link / rebuild / uninstall successfully
- run `test_demo_graphs` against `graphs/core/` successfully

Pass-specific validation:

- updated module demo graphs compile and load without missing-node or missing-type failures
- the `DualFilter` retrofits do not break per-note frequency or envelope-driven tone behavior
- the selected filter anchors remain readable members of the curated library rather than turning into a new preset sprawl

Listening acceptance:

- `DualWavetablePad` sounds more unified and intentional than the current single-filter version
- the chosen keys / brass example gains a clearer contour without becoming harsher or more complicated
- the bass example keeps grounded low-end while adding controlled bite
- any vocal / formant-adjacent recipe sounds supportive and musical rather than gimmicky
- `GlassInteractionKeys` either improves clearly with `DualFilter` or stays on plain `Filter` by explicit decision

## Assumptions and Defaults
- Pass 2 builds on the Pass 1 module surface that already exists in the package
- this is a selective adoption pass, not a blanket filter rewrite
- the curated preset library should stay compact; revise existing anchors before adding new graphs
- `GlassInteractionKeys` is intentionally allowed to remain single-filter if that preserves its identity better
- performance pages, expressive-lane UX, and module-local modulation remain deferred to Pass 3
