# Wavetable Asset Workflow

`vivid-wavetable` supports two scopes of wavetable content: **package factory assets** that ship with the package, and **user-imported workspace assets** that live in your local Vivid workspace library. Both use the same `wavetable_source=Custom` + file-path workflow on the package modules.

## Package Factory Assets

The package declares a factory wavetable root in `vivid-package.json`:

```json
"assets": {
  "wavetables": ["assets/wavetables"]
}
```

Factory assets ship under `assets/wavetables/` and are read-only. Each wavetable is synthesized with original harmonic content (not exported from the builtin bank) so the factory set provides timbres distinct from the builtin families.

The current factory set (12 wavetables, covering all 6 timbral families):

- `warm-keys-core.wav` — fundamental-heavy with gentle odd harmonics, morphing to richer blend
- `analog-soft.wav` — rounded saw-like spectrum that thins to near-sine
- `rooted-bass-edge.wav` — heavy fundamental morphing to aggressive odd-harmonic edge
- `bright-pluck-edge.wav` — harmonically rich attack character morphing to metallic edge
- `digital-glass.wav` — bright spectrum with phase offsets creating glassy interference
- `vocal-pad-sweep.wav` — formant-like resonance peaks that shift across the morph axis
- `vocal-air.wav` — breathy, formant-adjacent texture morphing from nasal to airy
- `glass-motion.wav` — bell-like partials with slight inharmonicity, morphing to dense shimmer
- `metallic-hollow.wav` — suppressed even harmonics creating hollow metallic tone, morphing to full
- `harmonic-rich.wav` — dense evenly weighted harmonic stack that thins progressively
- `spectral-sweep.wav` — emphasis peak sweeps up through the harmonic series
- `texture-tide.wav` — complex evolving texture with phase-shifted interference patterns

These paths are safe to reference in committed graphs and instrument presets.

## User-Imported Workspace Assets

Users can import their own wavetable files into the Vivid workspace library using `import_asset`. Imported assets can then be targeted through the same module file params used for package assets.

Workspace asset paths should **not** be committed to repo-tracked graphs, since they resolve to local workspace storage and will not exist for other users.

## Validation Scope

Automated package validation covers:

- manifest `assets.wavetables` declaration
- factory `.wav` asset loading from `assets/wavetables/`
- package-relative asset-backed graph smoke coverage

The workspace-import flow is supported, but it is **not** automatically exercised by package CI today.

## Manual Validation For Workspace Assets

To validate a workspace-imported wavetable end to end in a live Vivid runtime:

1. Run `import_asset` on an external wavetable WAV file.
2. Refresh or list assets and confirm the imported entry appears beside package assets in the merged wavetable listing.
3. Set a module or instrument graph to `wavetable_source=Custom` and write the imported asset's canonical `wav_file` path.
4. Confirm the graph loads and the custom wavetable is audible.

Keep those imported workspace paths out of committed graphs and presets.

## Switching Between Builtin and Custom Sources

All four package modules expose `wavetable_source` and file params:

| Module | Source Param | File Param |
|--------|-------------|------------|
| HybridKeys | `wavetable_source` | `wavetable_file` |
| GlassInteractionKeys | `wavetable_source` | `wavetable_file` |
| SubAirPad | `wavetable_source` | `wavetable_file` |
| DualWavetablePad | `osc_a_source` / `osc_b_source` | `osc_a_file` / `osc_b_file` |

Set `wavetable_source` to `Custom` (1) and provide a path in the file param to use a custom wavetable. Set it back to `Builtin` (0) to return to the builtin bank selected by `wavetable_family` and `wavetable_member`.

Module defaults remain `Builtin` — the custom-file path is opt-in.

## WAV File Format

Custom wavetable WAV files must be:

- **Mono** (single channel)
- **PCM format** (16-bit or 32-bit float)
- Samples are divided into frames of **2048 samples** each
- Minimum: 1 frame (2048 samples)
- Maximum: 256 frames (524,288 samples)
- Any standard sample rate works; the loader does not resample

The total sample count determines the number of wavetable frames. Samples beyond `256 * 2048` are ignored.

## What to Commit

- **Safe to commit:** references to package factory asset paths (`assets/wavetables/...`)
- **Do not commit:** references to workspace-imported asset paths (these are local to each user's workspace)
