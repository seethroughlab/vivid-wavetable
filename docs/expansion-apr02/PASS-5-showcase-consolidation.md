## Goal

Turn the expanded package into a coherent sound-design library rather than a pile of experiments, while preserving only the engine changes that proved their value in earlier passes.

This pass is about consolidation, curation, and polish. It should make the package easy to understand, easy to audition, and obviously broader than it was before the expansion work started.

## Engine Changes

- Limit engine work to cleanup and polish that was proven necessary by Passes 1-4.
- Fix rough edges that block library consistency:
  - naming cleanup where existing parameter names or graph labels became confusing,
  - small default retunes justified by repeated listening feedback,
  - light behavioral polish for operators added in earlier passes.
- Do not add speculative new synth primitives in this phase unless they unblock a clearly missing showcase family that the existing engine still cannot produce.
- Keep the focus on finishing and stabilizing what already earned its place.

## Graph / Demo Work

- Reorganize the graph library into clearer sound families.
- Ensure naming is consistent across the library.
  - Graph names should communicate sound family and role.
  - Avoid overlapping names that make curation harder later.
- Build a final balanced listening set that covers:
  - pads,
  - keys,
  - plucks,
  - leads,
  - basses,
  - textures,
  - arp/sequence patches,
  - cinematic hybrids.
- Ensure every major engine addition from earlier passes has:
  - one hero demo that shows the feature at full strength,
  - one simpler reference demo that teaches the idea more plainly.
- Prune weak, redundant, or confusing graphs.
  - Do not keep graphs solely to increase count.
  - If two graphs occupy the same role, keep the stronger one.

## Tests and Listening

- Re-run the full package validation flow:
  - configure + build package operators,
  - run package tests,
  - run the `vivid` link/rebuild/uninstall cycle,
  - run `test_demo_graphs` against `graphs/core/`.
- Audition the final listening library by family instead of only patch-by-patch.
  - Confirm that each family has a distinct role and emotional lane.
  - Confirm that the package feels intentionally curated rather than randomly expanded.
- Do a final quality pass on all retained hero/reference graphs.
  - Remove or rework anything that sounds underwhelming relative to the rest of the library.

## Acceptance Criteria

- The package reads as intentionally broader, not merely larger.
- Presets and graphs tell a coherent story about what the engine can now do.
- Every retained graph has a clear reason to exist.
- The final library includes strong representatives across the targeted sound families.
- No late-cycle cleanup regresses build, test, or smoke coverage.

## Explicit Non-Goals

- No late-cycle engine bloat.
- No keeping redundant graphs just to increase graph count.
- No new feature work that is not justified by the existing showcase gap.
- No reshuffling of smoke-safe `graphs/core/` content into expressive demos unless that improves both roles without adding risk.
