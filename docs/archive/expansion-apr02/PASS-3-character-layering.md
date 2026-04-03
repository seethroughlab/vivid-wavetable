## Goal

Add the missing source and character blocks that make layered patches feel expensive, dimensional, and finished rather than merely louder.

This pass should address the current gap between “good oscillator source” and “complete modern patch.” The focus is on air, body, glue, and controllable density.

## Engine Changes

- Add one dedicated polyphonic noise or air layer operator.
  - Design it as a reusable source block for breath, attack, fizz, and texture.
  - Keep the surface focused on musical usefulness rather than large synthetic-noise feature scope.
- Add one lightweight per-voice drive or character operator that sits before summing.
  - It should add body and cohesion to layered voices without immediately collapsing into harsh distortion.
  - Favor controllable soft-drive behavior over novelty distortion modes.
- Improve `SubOsc` only where it materially helps layering:
  - cleaner gain staging,
  - better defaults for blend and weight,
  - optional shape bias only if it clearly improves patch usefulness.
- Improve `VoiceMixer` only where the current implementation is limiting lush sounds:
  - more graceful summing under dense unison,
  - optional soft glue or output cohesion,
  - better width/pan handling for stacked voices if needed.
- Keep these additions modular.
  - Do not build an all-in-one mega voice strip.
  - Each new block must justify itself as a reusable primitive across multiple patch families.

## Graph / Demo Work

- Add 5-6 new graphs that explicitly depend on the new character path.
- The graph set must include:
  - airy keys,
  - noisy pluck,
  - low-brass pad,
  - analog-stack lead,
  - hybrid bass,
  - wide ambient bed.
- Use the new graphs to prove different jobs for the new blocks:
  - transient noise for articulation,
  - air layer for openness,
  - gentle drive for body,
  - mixer glue for width and cohesion.
- Keep the graphs musical rather than technical.
  - A graph should demonstrate a patch family, not merely expose a new node.
- Pair each new engine block with at least one graph where its contribution is obvious enough to hear in isolation.

## Tests and Listening

- Add stability tests for the new operators.
  - Output must remain finite under typical polyphonic use and reasonable parameter extremes.
- Add level and saturation sanity checks for dense layered graphs so voice summing does not explode or collapse.
- Run the standard package build/test/link/rebuild/uninstall/smoke workflow.
- Audition the new graphs with a focus on:
  - whether noise adds air rather than hiss,
  - whether drive adds body rather than fizz,
  - whether dense stacks remain legible and controllable,
  - whether mixer behavior remains predictable at high voice counts and moderate unison.

## Acceptance Criteria

- At least one new operator proves broadly useful across multiple graph families.
- The new graphs sound fuller and more dimensional without merely becoming louder or harsher.
- `VoiceMixer` remains predictable under unison-heavy patches.
- The new character path increases palette breadth in a way that is easy to hear from the listening set.
- The package continues to pass build, test, and smoke validation.

## Explicit Non-Goals

- No giant all-in-one voice strip.
- No gratuitous distortion feature set that does not broaden the palette.
- No speculative operator additions that lack clear graph-level justification.
- No promotion of redundant demo graphs that only differ trivially in settings.
