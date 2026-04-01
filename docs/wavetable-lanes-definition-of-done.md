# vivid-wavetable Lanes Migration: Definition Of Done

Use this checklist to decide when `vivid-wavetable` is fully migrated to the lanes core.

## Definition Of Done

- [ ] **Active package surface is modular-only**
  - `wavetable_synth` is absent from [`vivid-package.json`](/Users/jeff/Developer/vivid-wavetable/vivid-package.json)
  - `wavetable_synth` is absent from [`CMakeLists.txt`](/Users/jeff/Developer/vivid-wavetable/CMakeLists.txt)
  - active operators are only:
    - `poly_voice_allocator`
    - `wavetable_osc`
    - `voice_mixer`
    - `analog_osc`
    - `sub_osc`

- [ ] **Active operator code is fully lane-native**
  - no active file under `src/` uses:
    - `VividSpreadPort`
    - `VIVID_PORT_SPREAD`
    - `input_spreads`
    - `output_spreads`
  - active code uses lane-native core surfaces instead

- [ ] **The monolith is archive-only**
  - [`archive/src/wavetable_synth.cpp`](/Users/jeff/Developer/vivid-wavetable/archive/src/wavetable_synth.cpp) is the only code home for `WavetableSynth`
  - monolith graphs, tests, and presets live only under `archive/`
  - no active graph, test, manifest, or build target depends on `WavetableSynth`

- [ ] **Active docs teach only the modular story**
  - [`README.md`](/Users/jeff/Developer/vivid-wavetable/README.md) describes the package as modular
  - [`docs/synth-building-tutorial.md`](/Users/jeff/Developer/vivid-wavetable/docs/synth-building-tutorial.md) teaches the modular chain
  - no active doc presents `WavetableSynth` as the supported path

- [ ] **The migration plan doc is no longer stale**
  - either delete [`docs/wavetable-lanes-migration-plan.md`](/Users/jeff/Developer/vivid-wavetable/docs/wavetable-lanes-migration-plan.md)
  - or rewrite it so it no longer claims:
    - the package fails to build
    - active operators still use spread-era API

- [ ] **Build and tests pass from the active tree**
  - `cmake --build build` succeeds
  - package tests pass
  - smoke coverage uses the modular graph only
  - key active validation files are:
    - [`tests/test_package_manifest.cpp`](/Users/jeff/Developer/vivid-wavetable/tests/test_package_manifest.cpp)
    - [`tests/cpp/test_audio_correctness.cpp`](/Users/jeff/Developer/vivid-wavetable/tests/cpp/test_audio_correctness.cpp)
    - [`graphs/core/wavetable_modular_demo.json`](/Users/jeff/Developer/vivid-wavetable/graphs/core/wavetable_modular_demo.json)

- [ ] **Final grep gates are clean**
  - active surfaces have no hits for:
    - `VividSpreadPort`
    - `VIVID_PORT_SPREAD`
    - `input_spreads`
    - `output_spreads`
    - active `wavetable_synth` references
  - allowed remaining hits are only in:
    - `archive/`
    - explicit historical notes, if kept

- [ ] **The worktree is settled**
  - no large migration diff remains uncommitted
  - the migrated modular-only structure is committed, not just locally in progress

## Suggested PR Summary

`vivid-wavetable` is fully migrated to the lanes core and now exposes a modular-only active package surface. The legacy `WavetableSynth` monolith is archived, active operators and tests use lane-native core API, active docs and graphs teach the modular chain, and no spread-era core API remains in active package surfaces.
