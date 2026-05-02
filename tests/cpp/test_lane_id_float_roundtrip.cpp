// Pin the precondition that synthetic per-voice lane ids must survive a
// float32 round-trip. WavetableLayer/WavetableOsc pass lane ids into their
// renderers through a `float[]` lane buffer (synth_lane_ids[]) and read them
// back via static_cast<uint32_t>. If kMidiLaneIdBase + slot exceeds the 24-bit
// float mantissa, every (base + i) collapses to the same float, which then
// makes vivid_lane_state(ctx, lid, Voice) return the *same* Voice for every
// polyphonic voice — voices clobber each other's persistent state every audio
// block, producing a 1024-sample-period click at the block boundary. Diagnosed
// 2026-04-29; see project_buzz_diagnostic_state.md for the long story.
//
// The values below are mirrored from src/wavetable_layer_internal.h and
// src/wavetable_osc_internal.h. They are duplicated here on purpose: the
// production headers pull in heavy operator_api / WebGPU dependencies that the
// test runner doesn't have, and the goal of this test is to fail loudly when
// someone bumps the base back into a region where float can't represent
// (base + slot) exactly. If you change the production constants, change them
// here too.

#include <cstdint>
#include <cstdio>

namespace {

// Mirrors WavetableLayer::kMidiLaneIdBase and ::kMaxVoices.
constexpr uint32_t kLayerLaneIdBase = 0x100000u;
constexpr int kLayerMaxVoices = 16;

// Mirrors WavetableOsc::kMidiLaneIdBase and ::kMaxVoices.
constexpr uint32_t kOscLaneIdBase = 0x110000u;
constexpr int kOscMaxVoices = 16;

int failures = 0;

void check_roundtrip(const char* who, uint32_t base, int max_voices) {
    int local_failures = 0;
    for (int slot = 0; slot < max_voices; ++slot) {
        uint32_t expected = base + static_cast<uint32_t>(slot);
        float as_float = static_cast<float>(expected);
        uint32_t roundtripped = static_cast<uint32_t>(as_float);
        if (roundtripped != expected) {
            std::fprintf(stderr,
                         "  FAIL: %s lane id 0x%x (slot %d) does not survive float roundtrip"
                         " (got 0x%x). The float32 mantissa is 24 bits;"
                         " pick a base such that (base + max_voices - 1) < 2^24.\n",
                         who, expected, slot, roundtripped);
            ++local_failures;
        }
    }
    if (local_failures == 0) {
        std::fprintf(stderr, "  PASS: %s base=0x%x round-trips through float for all %d slots\n",
                     who, base, max_voices);
    }
    failures += local_failures;
}

}  // namespace

int main() {
    std::fprintf(stderr, "\n--- Lane id float roundtrip ---\n");

    check_roundtrip("WavetableLayer", kLayerLaneIdBase, kLayerMaxVoices);
    check_roundtrip("WavetableOsc",   kOscLaneIdBase,   kOscMaxVoices);

    if (failures > 0) {
        std::fprintf(stderr, "\nFAIL: %d lane id roundtrip failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "\nOK\n");
    return 0;
}
