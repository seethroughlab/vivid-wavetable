# Phase 5 Benchmark Results

> **Status: PRIMARY 256-FRAME GATES PASS WITH LARGE RELEASE-BUILD MARGIN.**
> After the sample-axis Highway recovery and macOS Accelerate backend pass,
> `WavetableLayer` beats the legacy `WavetableOsc` comparison at the primary
> 256-frame runtime setting with repeated-run margin well above the Phase 5
> target.

## Run Configuration

| Field | Value |
|-------|-------|
| Date | 2026-04-10 |
| Machine | arm64 / Darwin 25.3.0 |
| OS | macOS |
| Buffer size | 256 |
| Sample rate | 48000 |
| Build type | Release |
| Backend (build/config inference) | Accelerate preferred on macOS; Highway fallback |
| Vivid version | 0.1.0 |
| vivid-wavetable version | 0.2.0 |

## Single Instance (wavetable_layer_pad_demo)

| Metric | Value |
|--------|-------|
| mean audio_load | 0.0127 ± 0.0038 |
| max audio_load | 0.0376 |
| xruns | 0 |
| top node | instrument.__wt (WavetableLayer) ema=26-52us budget_pct=0.4-1.6% |
| **Primary 256-frame gate: mean load <= 0.30** | **PASS** (0.0127) |

## Four Instance WavetableLayer (wavetable_layer_stress)

| Metric | Value |
|--------|-------|
| mean audio_load | 0.0337 ± 0.0045 |
| max audio_load | 0.0441 |
| xruns | 0 |
| top nodes | wt_1..wt_4 (WavetableLayer) ema=27-33us budget_pct=0.5-0.7% |
| **Primary 256-frame gate: xruns == 0** | **PASS** |

## Four Instance Legacy (wavetable_osc_stress)

| Metric | Value |
|--------|-------|
| mean audio_load | 0.8550 ± 0.0195 |
| max audio_load | 0.9161 |
| xruns | 0 |
| top nodes | osc_1..osc_4 (WavetableOsc) ema=1079-1116us budget_pct=19.3-23.2% |

## Comparison

| Metric | WavetableLayer 4x | Legacy 4x |
|--------|-------------------|-----------|
| mean audio_load | 0.0337 ± 0.0045 | 0.8550 ± 0.0195 |
| Improvement | 96.1% ± 0.5% | baseline |
| **Primary 256-frame gate: WavetableLayer < Legacy** | **PASS** | |

## Analysis

The initial failure showed WavetableLayer was ~52% slower per-node than WavetableOsc (6.7ms vs 4.4ms). The recovery pass replaced the primary slot-axis SIMD renderer, which paid for per-sample horizontal reduction, with a sample-axis renderer that accumulates directly into contiguous stereo output spans. A stable-position fast path also avoids per-sample frame-plan work in the 4x stress fixture.

The macOS Accelerate backend was added as a narrow hot-path backend for no-warp/no-drift/no-pitch production rendering. In Release builds it is slightly ahead of the Highway-preferred run on the same benchmark fixture (`4x mean audio_load 0.0337` in the latest Accelerate-preferred run vs `0.0378` observed in the Release Highway-preferred comparison), so macOS builds now prefer Accelerate while retaining Highway fallback for unsupported paths and for future Windows portability.

Remaining observations:

- the repeated 256-frame live runtime run now passes all three primary gates
- the improvement over legacy is well above the preferred 15% Phase 5 margin
- backend label is still build/config inference in this report; package-local renderer telemetry distinguishes scalar, Highway, and Accelerate internally, but the control server does not yet expose operator-specific backend fields to the benchmark harness

## Phase 5 Closure Notes

- backend attribution via build/config inference is accepted for Phase 5; no runtime/control-server telemetry change is required before Phase 6
- keep Highway as the portable SIMD fallback and evaluate a Windows-native counterpart only when the Windows port begins
- treat the Release build type as mandatory for performance acceptance runs; unoptimized package builds produce misleading callback timings

## Cross-Platform Status

| Platform | Build | Tests | Benchmarks |
|----------|-------|-------|------------|
| macOS (ARM64) | PASS | PASS | PASS at 256 primary; large repeated Release margin |
| Windows (x64) | deferred | deferred | deferred |

## Notes

- Backend label is a build/config heuristic, not authoritative for the running instance. The control server does not currently expose which renderer backend was selected.
- Windows validation is deferred to a future Windows-port gate and does not block Phase 6, provided the public API and renderer boundary remain portable.
- Buffer size 256 is the primary Phase 5 reference for this run.
