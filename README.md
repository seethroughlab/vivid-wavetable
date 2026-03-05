# vivid-wavetable

`vivid-wavetable` is an extracted Vivid package that provides the `WavetableSynth` audio operator.

## Contents

- `src/wavetable_synth.cpp`
- `factory_presets/wavetable_synth.json`
- `graphs/wavetable_demo.json`
- `graphs/wavetable_position_env_demo.json`
- `tests/test_package_manifest.cpp`
- `tests/test_wavetable_position_env.cpp`
- `vivid-package.json`

## Local development

From vivid-core:

```bash
./build/vivid link ../vivid-wavetable
./build/vivid rebuild vivid-wavetable
```

## CI smoke coverage

The package CI workflow:

1. Clones and builds vivid-core (`test_demo_graphs` + core operators).
2. Builds package operators and package tests.
3. Runs package tests.
4. Runs graph smoke tests against this package's `graphs/` directory.

## License

MIT (see `LICENSE`).
