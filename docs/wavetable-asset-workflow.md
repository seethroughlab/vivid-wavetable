# Wavetable Asset Workflow

`vivid-wavetable` supports two scopes of wavetable content: **package factory assets** that ship with the package, and **user-imported workspace assets** that live in your local Vivid workspace library.

## Package Factory Assets

The package declares a factory wavetable root in `vivid-package.json`:

```json
"assets": {
  "wavetables": ["assets/wavetables"]
}
```

Factory assets ship under `assets/wavetables/` and are read-only. The current factory set:

- `warm-keys-core.wav` — warm analog-family core wavetable
- `vocal-pad-sweep.wav` — vocal formant sweep for pads
- `glass-motion.wav` — metallic glass texture for keys and leads
- `rooted-bass-edge.wav` — warm analog edge for bass voices
- `texture-tide.wav` — texture motion core for beds and drones
- `bright-pluck-edge.wav` — bright digital edge for plucks and attacks

These paths are safe to reference in committed graphs and instrument presets.

## User-Imported Workspace Assets

Users can import their own wavetable files into the Vivid workspace library using `import_asset`. Imported assets appear alongside package factory assets in the merged asset listing.

Workspace asset paths should **not** be committed to repo-tracked graphs, since they resolve to local workspace storage and will not exist for other users.

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
