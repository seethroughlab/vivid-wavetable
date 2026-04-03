## Multi-Pass Richness Plan for `vivid-wavetable`

### Summary
We should treat this as a staged hybrid program: every pass adds at least one meaningful engine improvement and a small set of new demo graphs so we can listen immediately, while still pushing the synth architecture forward.

The goal is not strict Serum/Pigments parity in one shot. The goal is to steadily raise the ceiling in four areas that matter most for “lush, complex, expensive” sound:
1. richer source material,
2. denser oscillator behavior,
3. more tone-shaping/character in the voice path,
4. better musical graph design that actually exploits the engine.

### Pass 1 — Stronger Baseline, Faster Listening
- Improve the existing oscillator path without changing the package shape:
  - add modulation smoothing where current movement sounds steppy or brittle,
  - tighten default level staging across oscillator, mixer, and common graph recipes,
  - review current unison defaults so “more voices” sounds wider and smoother, not harsher.
- Build a new preset-graph baseline that deliberately uses the current engine harder:
  - 2-osc layered patches,
  - wavetable + analog hybrids,
  - wavetable + sub foundations,
  - movement-driven pads and keys using `LfoAu`, `EnvelopeAu`, chorus, delay, and reverb.
- Add 6-8 new musical demo graphs split across:
  - lush pad,
  - glassy digital pad,
  - hybrid poly keys,
  - warm bass,
  - animated arp,
  - cinematic texture.
- Use `vivid-sequencers` in the musical demos where sequence content materially helps the listening pass.
- Keep `graphs/core/` conservative and smoke-safe; put the more expressive listening graphs in `graphs/presets/` or `graphs/extended/`.

### Pass 2 — Wavetable Content and Motion Depth
- Expand the wavetable library substantially instead of relying on the current small built-in bank.
- Add organized wavetable families with clear tonal roles:
  - analog warm,
  - bright digital,
  - vocal/formant,
  - metallic,
  - harmonic/spectral,
  - texture/noise-adjacent.
- Improve `WavetableOsc` so it has better “living motion” before extra FX:
  - start-phase control,
  - random phase per note,
  - slow drift/slop per voice,
  - optional smoothing on wavetable position and warp modulation,
  - stereo phase/spread behavior that creates width without brittle combing.
- Keep this as an improvement to the existing operator rather than creating a separate “advanced wavetable osc.”
- Add 4-6 new graphs specifically designed to audition these new motion behaviors:
  - lush supersaw-like pad,
  - moving vocal pad,
  - evolving texture bed,
  - animated pluck,
  - modern bright lead.

### Pass 3 — Voice Character and Layering Operators
- Add the missing source/character blocks that make flagship synth patches feel finished:
  - a dedicated polyphonic noise/air layer operator for breath, attack, and texture,
  - a lightweight per-voice drive/saturation operator before summing,
  - optional stereo spread/width behavior in the voice path if the existing mixer remains too plain.
- Improve `SubOsc` only where it materially helps layering:
  - cleaner gain staging,
  - better blend defaults,
  - optional slight saturation or shape bias if needed for weight.
- Improve `VoiceMixer` only where it is limiting lush sounds:
  - more graceful summing under dense unison,
  - optional soft saturation or output glue,
  - better width/pan handling for stacked voices.
- Add 5-6 new graphs that explicitly use these new blocks:
  - cinematic low brass pad,
  - airy keys,
  - noisy pluck,
  - warm analog-stack lead,
  - aggressive hybrid bass,
  - wide ambient bed.

### Pass 4 — Richer Oscillator Interaction
- Deepen oscillator-to-oscillator interaction so patches do not rely only on static layering:
  - improve FM/RM/AM usability and gain staging,
  - add at least one or two new musically useful warp/interact modes if current ones are too narrow,
  - make cross-mod patches stable and easy to voice in polyphonic graphs.
- Prioritize interactions that produce audible richness without demanding fragile graph wiring.
- Do not add a full modulation-matrix system in this pass; use the graph model for routing and keep the operator surface focused.
- Add 4-5 demo graphs built around interaction rather than static stacking:
  - FM glass keys,
  - metallic but controlled digital lead,
  - motion-rich hybrid arp,
  - aggressive growl bass,
  - spectral texture patch.

### Pass 5 — Consolidation and Showcase Library
- Normalize the package into a clearer sound-design library:
  - group graphs by family,
  - keep naming consistent,
  - ensure each important engine feature has at least one “hero” demo and one simpler reference graph.
- Refresh factory presets and graph presets together so the operator-level presets and graph-level listening assets tell the same story.
- Build a final balanced listening set that covers:
  - pads,
  - keys,
  - plucks,
  - leads,
  - basses,
  - textures,
  - arp/sequence patches,
  - cinematic hybrids.
- End with a package-owned showcase set that is clearly broader than today, not just larger.

### Public APIs / Interfaces
- Improve existing operator interfaces first:
  - `WavetableOsc` gains additional motion/phase/spread controls,
  - `VoiceMixer` may gain optional glue/width behavior,
  - `SubOsc` may gain small layering-oriented improvements.
- Add new package-owned operators only where the current surface is genuinely missing a sound-design primitive:
  - one dedicated noise/air layer operator,
  - one lightweight per-voice drive/character operator.
- No operator renames, no manifest identity changes, and no requirement to move functionality back into core Vivid.

### Test and Listening Plan
- After every pass:
  - build package operators,
  - run package tests,
  - run `vivid` link/rebuild/uninstall,
  - run `test_demo_graphs` on `graphs/core/`,
  - separately audition the new non-core demo graphs by family.
- Add focused behavior tests when engine work lands:
  - wavetable bank generation/import behavior,
  - unison/stereo spread sanity,
  - level/stability checks for modulation and cross-mod paths,
  - saturation/character operators staying finite and stable.
- Treat the listening library as part of acceptance:
  - every pass must produce at least 3 clearly improved “keeper” sounds,
  - each new operator or major new parameter must have at least one graph that proves why it exists,
  - no pass is complete if it adds engine complexity without an audible demo.

### Assumptions and Defaults
- We will optimize for a hybrid result: new graphs every pass, but engine work is mandatory in multiple passes.
- Musical showcase graphs can use `vivid-sequencers`; smoke fixtures in `graphs/core/` stay stable and headless-friendly.
- We will prefer broad, reusable sound-design primitives over one-off “mega operator” features.
- If a potential engine idea does not clearly unlock a new patch family, it should be deferred rather than added speculatively.
