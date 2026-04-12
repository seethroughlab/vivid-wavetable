## Instrument Adoption Plan for `vivid-wavetable`

> Historical note: this roadmap records the April 4 adoption plan. For the current WavetableLayer-era package surface, use `README.md`, `docs/showcase-library.md`, `docs/wavetable-asset-workflow.md`, and `docs/wavetable-operator-validation-guide.md`.

### Summary
Passes 1-5 made `vivid-wavetable` a strong modular wavetable engine with a curated showcase set, lane-aware voice architecture, better oscillator interaction, and clearer validation/docs.

The next phase is still about **instrument coherence**, but the framing changes. The package is no longer mainly blocked on new oscillator ideas first, and it is no longer fully blocked on missing core primitives. The relevant Vivid-core surfaces now exist in usable form:

- subgraph modules with exposed params/ports and flattening,
- module-local `mod_assignments`,
- `DualFilter`,
- asset-library/import/query plumbing with package asset support,
- expressive `MidiInput` lane outputs plus performance-page metadata,
- graph `content_kind="instrument"` and instrument-aware browser affordances.

So this roadmap is no longer a speculative wish list for core support. It is an **adoption plan**: use the core primitives that now exist, pressure-test them through real package usage, and refine both package and platform where adoption reveals rough edges.

The comparison target remains closer to Ableton Live Wavetable in terms of coherence, playability, and the feeling of a finished instrument voice. It is not a claim of Serum/Pigments breadth, and it is not a claim that the current Vivid implementations are already fully finished just because the primitives exist.

What this phase is not:

- not a full modulation matrix program,
- not a new synthesis-engine expansion,
- not a plan to move `vivid-wavetable` operators back into Vivid core,
- not a replacement of graph routing as the primary synthesis model,
- not an assumption that the new Vivid-core features are already fully proven.

### Pass 1 — Module-First Instrument Architectures
This pass should convert the package's best voice designs into actual instrument-facing module patterns using the current subgraph-module system.

- Define 3-4 canonical `vivid-wavetable` instrument modules based on the current voice patterns:
  - dual wavetable,
  - wavetable + analog,
  - wavetable + sub + air,
  - interaction-led metallic/glass voice.
- Keep the existing operators as the internal voice engine.
- Expose a small, curated control surface instead of the full graph internals.
- Treat this as an adoption and authoring pass for the current module system, not a redesign of module runtime behavior.

#### Public Interfaces / Graph Surface
- Add package-owned module definitions alongside the existing operator set.
- Preserve plain graphs and plain operators; modules are an additional instrument-facing surface, not a replacement.
- Keep lane architecture and internal graph routing intact.

#### Test and Listening Plan
- Keep package build/tests green.
- Keep `vivid` link/rebuild/uninstall green.
- Keep `graphs/core/` smoke green.
- Add at least 2-3 module-backed reference instruments and compare them against the current plain-graph equivalents.
- Acceptance should include listening for:
  - a keys module that is easier to understand than the raw graph,
  - a readable layered pad module,
  - an interaction-led module that still sounds intentional.

#### Not In This Pass
- No new oscillator operators.
- No package-specific module runtime hacks.
- No assumption that every retained showcase graph must become a module.

### Pass 2 — DualFilter Adoption and Tone-Shaping Recipes
This pass should adopt the current `DualFilter` primitive where it materially improves the package's instrument voices, while also discovering whether the current filter platform is sufficient as-is.

- Define package-level filter recipes built around `DualFilter` for:
  - warm subtractive body,
  - bright animated keys,
  - brass contour,
  - bass bite,
  - vocal/formant-adjacent movement where useful.
- Decide where `DualFilter` should replace current single-filter graph recipes and where the plain `Filter` remains simpler or clearer.
- Pressure-test whether the current `DualFilter` surface is sufficient for instrument use or whether small focused Vivid follow-ups remain justified.

#### Public Interfaces / Graph Surface
- More filter-forward package reference graphs and module internals.
- Possible package preset/default retunes around filter routing.
- No package-owned filter wrapper unless package usage proves the current core primitive is still awkward.

#### Test and Listening Plan
- Keep build/tests/smoke green.
- Add filter-focused listening coverage for:
  - one warm pad,
  - one animated keys/brass patch,
  - one bass patch with controlled bite.
- Acceptance should explicitly judge:
  - whether `DualFilter` makes package voices smaller and clearer,
  - whether the tone-shaping story now feels more instrument-like.

#### Not In This Pass
- No broad filter-bank expansion.
- No replacement of Vivid core filters.
- No effect-suite detour.

### Pass 3 — Module-Local Modulation, Performance Pages, and Expressive Play
This pass should adopt the new core modulation/performance surfaces that already exist and use package content to discover where they are still rough.

- Establish one canonical modulation/performance convention for package instruments:
  - motion,
  - brightness,
  - air,
  - body,
  - interaction.
- Use module-local modulation assignment instead of reopening module internals for every common macro move.
- Build at least a few package instruments that respond meaningfully to expressive lane inputs and/or curated performance pages.
- Treat this as a package UX pass for the current core features, not permission to invent a giant new modulation system.

#### Public Interfaces / Graph Surface
- Module definitions should expose curated performance pages.
- Package instruments may start consuming expressive lane inputs where musically justified.
- Factory presets and reference graphs should start reflecting the new macro/performance conventions.

#### Test and Listening Plan
- Keep build/tests/smoke green.
- Add instrument-ready validation for:
  - one pad with clear motion control,
  - one keys/lead voice with obvious brightness/body control,
  - one interaction patch with a useful performance surface,
  - one expressive-play scenario using the new `MidiInput` lanes.
- Acceptance should include:
  - modulation assignment remains readable,
  - performance pages feel worth using,
  - expressive inputs are stable and musically useful, not only technically connected.

#### Not In This Pass
- No full matrix expansion.
- No package-specific MPE transport.
- No separate instrument-state model outside current params, presets, and variations.

### Pass 4 — Asset-Driven Wavetable Workflow and Instrument Library
This pass should adopt the current asset-library and instrument-browser features so the package behaves more like an actual instrument library.

- Move the package's custom wavetable story onto the current asset-library workflow instead of treating it as a future platform idea.
- Define how package factory assets and user-imported wavetable assets should coexist.
- Mark retained module/graph content as instrument-ready where appropriate.
- End with a final library that feels like a coherent synth package:
  - bread-and-butter playable instruments,
  - a few hero sounds,
  - cleaner browser semantics than “everything is just a demo graph.”

#### Public Interfaces / Graph Surface
- Update package manifest/assets layout only as needed to align with current Vivid asset discovery.
- Add instrument-oriented graph metadata consistently.
- Expand factory presets and module/reference content around actual instrument roles.

#### Test and Listening Plan
- Keep build/tests/smoke green.
- Validate asset-library flows against the package:
  - package-owned wavetable assets,
  - imported user assets,
  - package graphs/modules that consume them.
- Validate final content coverage for:
  - playable keys,
  - expressive pads,
  - grounded basses,
  - one metallic/interaction voice,
  - one motion-heavy but readable patch.
- Acceptance should include a content-pruning pass so the final library stays curated.

#### Not In This Pass
- No custom wavetable editor.
- No cloud/content-service work.
- No regression into a large uncurated preset pile.

### Public Interfaces / Expected Changes
Likely public-facing changes in this program are now:

- package-owned module definitions built on the current Vivid subgraph-module system,
- more package reference graphs and factory presets organized around instrument roles,
- adoption of `DualFilter` where it improves synth voices,
- adoption of module-local modulation assignment and performance pages,
- package use of expressive `MidiInput` lanes where musically justified,
- package alignment with the current asset library and instrument browser metadata.

The following should stay stable unless package adoption proves a real blocker:

- the lane architecture,
- the Pass 1-5 package operators as the core sound engine,
- graph routing as the primary synthesis model,
- package ownership of synth voicing decisions,
- Vivid core as the provider of reusable primitives rather than synth-specific hard-coded paths.

### Validation Strategy
Every pass in this roadmap should meet the same baseline:

- package configure/build stays green,
- package tests stay green,
- `vivid` link/rebuild/uninstall stays green,
- `test_demo_graphs` against `graphs/core/` stays green.

Every pass must also prove real package adoption of the current core primitives:

- Pass 1: module-backed instruments are clearer than their raw graphs,
- Pass 2: `DualFilter` materially improves at least a few voice recipes,
- Pass 3: local modulation/performance surfaces feel useful and not patch-fragile,
- Pass 4: asset workflow and instrument metadata make the package easier to browse and use.

Overall acceptance should include repeated listening passes centered on:

- playable keys,
- expressive pads,
- grounded basses,
- one metallic/interaction voice,
- one motion-heavy but still readable patch.

If a core primitive proves awkward during package adoption, that is not a package failure; it is evidence for a focused Vivid follow-up.

No pass is complete if it only “uses the new feature” without making the package feel more like a coherent instrument.

### Assumptions and Defaults
- This roadmap starts from the current curated package after Pass 5.
- The package plan is now an adoption plan for existing and emerging Vivid core instrument primitives, not a speculative wishlist for missing core features.
- The comparison target remains closer to Ableton Wavetable as a coherent instrument, not exact cloning and not Serum/Pigments breadth.
- The relevant Vivid core capabilities are present today, but this document does not overclaim that they are already fully proven or fully finished until the package has adopted them successfully.
- This file should remain a standalone `PLAN.md`; do not split it into per-pass docs yet.
