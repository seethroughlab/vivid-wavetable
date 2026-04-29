// OPT-IN STUB — currently always passes with a SKIP message.
//
// Pins down a known but unfixed bug: Filter (in the vivid-core repo at
// operators/audio/filter/filter.cpp) reads `input_lanes[0]` and
// `input_lanes[1]` as if they were `cutoff_mod` and `frequencies`, but
// VividAudioContext::input_lanes is sized and indexed by overall input
// port position (graph_compiler.cpp:228). Filter's port order is:
//   [0] input (audio_buffer)
//   [1] cutoff_cv (scalar)
//   [2] resonance_cv (scalar)
//   [3] cutoff_mod (lane_array)        ← actual cutoff_mod port
//   [4] frequencies (lane_array)        ← actual frequencies port
// So Filter currently reads the wrong slots. Per-voice keytracking from
// `NoteBreakout/voice_freqs → Filter/frequencies` produces no effect (no
// audible filter motion across the chord) — exactly what we observed
// when listening to dream_keys with a bypassed reverb.
//
// This test is a placeholder. Properly testing Filter from this package
// requires either (a) reaching across repos to dlopen filter.dylib from
// the Vivid.app bundle (path-fragile), or (b) building a JSON graph
// fixture that runs through the live runtime (out of scope for ctest).
// The right home for the actual test is the vivid-core repo's
// tests/audio/filter test target, paired with the Filter source fix.
//
// When the Filter input-lane indexing is fixed in vivid core (start at
// input_lanes[3] / input_lanes[4], not input_lanes[0] / input_lanes[1]),
// either replace this stub with a real cross-operator test or move the
// test to the vivid-core repo and delete this file.

#include <cstdio>

int main() {
    std::fprintf(stderr,
                 "SKIP: test_filter_lane_input_routing — Filter input-lane "
                 "indexing bug is identified but unfixed. The fix and its "
                 "regression test belong in the vivid-core repo "
                 "(operators/audio/filter/filter.cpp). See "
                 "docs/wavetable-operator-validation-guide.md (Bottom-up "
                 "isolation, Step 6) for the symptom.\n");
    return 0;
}
