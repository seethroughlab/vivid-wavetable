// Codifies the lesson: WavetableLayer's voice_*/control breakout lanes
// must publish the right data to the right port. The bug we fixed:
// emit_voice_breakouts was indexing ctx->output_lanes[0..3] as if those
// were the lane-array slots, but output_lanes[] is sized by overall
// OUTPUT port position (graph_compiler.cpp:207). With `output` (audio)
// at index 0, the first lane port is at index 1. The pre-fix code shifted
// every lane down by one — voice_freqs went silent and voice_velocities
// reported frequency Hz instead of velocity values.
//
// This test binds capturing lane outputs to all four voice_* ports on a
// WavetableLayer, runs a single-note render, and asserts:
//   - voice_ids has a non-zero note_id
//   - voice_gates is 1.0 (note is on)
//   - voice_velocities is in [0, 1] (looks like a velocity, not a frequency)
//   - voice_freqs is the actual Hz of the played note
//
// If the indexing regresses, voice_freqs will be empty or zero and
// voice_velocities will report Hz — the exact signature we observed
// before the fix.

#include "test_support.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef VIVID_PLUGIN_SUFFIX_STR
#if defined(_WIN32)
#define VIVID_PLUGIN_SUFFIX_STR ".dll"
#elif defined(__APPLE__)
#define VIVID_PLUGIN_SUFFIX_STR ".dylib"
#else
#define VIVID_PLUGIN_SUFFIX_STR ".so"
#endif
#endif

namespace {

int failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "  FAIL: %s\n", msg);
        ++failures;
    } else {
        std::fprintf(stderr, "  PASS: %s\n", msg);
    }
}

// Captures lane writes for assertion. resize() returns the pre-allocated
// backing storage; commit() records the written length. The capture is
// flat (single block) — fine for these per-block assertions.
struct CapturedLane {
    static constexpr uint32_t kCapacity = 64;
    float buffer[kCapacity] = {};
    uint32_t length = 0;
    bool ever_written = false;

    static void* resize_fn(void* handle, uint32_t length) {
        auto* self = static_cast<CapturedLane*>(handle);
        self->length = (length <= kCapacity) ? length : kCapacity;
        return self->buffer;
    }
    static void commit_fn(void* handle, uint32_t length) {
        auto* self = static_cast<CapturedLane*>(handle);
        self->length = (length <= kCapacity) ? length : kCapacity;
        self->ever_written = true;
    }
    void reset() {
        length = 0;
        ever_written = false;
        std::memset(buffer, 0, sizeof(buffer));
    }
};

}  // namespace

int main() {
    MiniLoader loader;
    std::string path = std::string("./wavetable_layer") + VIVID_PLUGIN_SUFFIX_STR;
    if (!loader.load(path.c_str())) {
        std::fprintf(stderr, "FAIL: could not load %s\n", path.c_str());
        return 1;
    }
    const VividOperatorDescriptor* desc = loader.descriptor();
    if (!desc) return 1;

    // Resolve param indices for setup
    int idx_amplitude = -1, idx_position = -1;
    for (uint32_t p = 0; p < desc->param_count; ++p) {
        const char* n = desc->params[p].name;
        if (!std::strcmp(n, "amplitude")) idx_amplitude = p;
        else if (!std::strcmp(n, "position")) idx_position = p;
    }

    std::vector<float> params(desc->param_count);
    for (uint32_t p = 0; p < desc->param_count; ++p)
        params[p] = desc->params[p].default_value;
    if (idx_amplitude >= 0) params[idx_amplitude] = 0.3f;
    if (idx_position >= 0)  params[idx_position]  = 0.5f;

    // WavetableLayer output port layout (per src/wavetable_layer.cpp:140–171):
    //   [0] output (audio buffer)
    //   [1] voice_ids
    //   [2] voice_gates
    //   [3] voice_velocities
    //   [4] voice_freqs
    static constexpr int kOutputPortCount = 5;
    CapturedLane captured[kOutputPortCount];
    VividValueOutput lane_outputs[kOutputPortCount] = {};
    for (int i = 0; i < kOutputPortCount; ++i) {
        lane_outputs[i].handle = &captured[i];
        lane_outputs[i].resize = CapturedLane::resize_fn;
        lane_outputs[i].commit = CapturedLane::commit_fn;
    }

    void* inst = loader.create_instance();
    PolyTestContext tc;
    tc.set_output_channels(2);
    tc.ctx.param_values = params.data();
    tc.ctx.value_outputs = lane_outputs;
    tc.setup_wavetable_layer_voice(261.63f);  // C4

    std::fprintf(stderr, "\n--- voice_*/control breakouts publish correct data ---\n");

    // Burn in a couple blocks so the voice is firmly in sustain — note_id is
    // assigned on note-on and persists.
    for (int b = 0; b < 2; ++b) {
        for (auto& c : captured) c.reset();
        tc.clear_output();
        loader.process_audio(inst, &tc.ctx);
        tc.ctx.time += static_cast<double>(tc.kFrames) / tc.kSampleRate;
        tc.ctx.frame++;
        tc.clear_notes();
    }

    // Final block — the one we assert on.
    for (auto& c : captured) c.reset();
    tc.clear_output();
    loader.process_audio(inst, &tc.ctx);

    // Indices match the comment above.
    const auto& voice_ids        = captured[1];
    const auto& voice_gates      = captured[2];
    const auto& voice_velocities = captured[3];
    const auto& voice_freqs      = captured[4];

    std::fprintf(stderr, "    voice_ids:        len=%u first=%.3f\n", voice_ids.length,        voice_ids.length        ? voice_ids.buffer[0]        : 0.0f);
    std::fprintf(stderr, "    voice_gates:      len=%u first=%.3f\n", voice_gates.length,      voice_gates.length      ? voice_gates.buffer[0]      : 0.0f);
    std::fprintf(stderr, "    voice_velocities: len=%u first=%.3f\n", voice_velocities.length, voice_velocities.length ? voice_velocities.buffer[0] : 0.0f);
    std::fprintf(stderr, "    voice_freqs:      len=%u first=%.3f\n", voice_freqs.length,      voice_freqs.length      ? voice_freqs.buffer[0]      : 0.0f);

    check(voice_ids.length        == 1u, "voice_ids has one entry for the single voice");
    check(voice_gates.length      == 1u, "voice_gates has one entry");
    check(voice_velocities.length == 1u, "voice_velocities has one entry");
    check(voice_freqs.length      == 1u, "voice_freqs has one entry (would be 0 under the pre-fix shift bug)");

    if (voice_ids.length        > 0) check(voice_ids.buffer[0]        > 0.0f,  "voice_ids carries a non-zero note_id");
    if (voice_gates.length      > 0) check(std::abs(voice_gates.buffer[0] - 1.0f) < 1e-4f, "voice_gates is 1.0 while gated");
    if (voice_velocities.length > 0) check(voice_velocities.buffer[0] >= 0.0f && voice_velocities.buffer[0] <= 1.0f,
                                            "voice_velocities is in [0,1] (would be ~261 Hz under the pre-fix shift bug)");
    if (voice_freqs.length      > 0) check(std::abs(voice_freqs.buffer[0] - 261.63f) < 1.0f,
                                            "voice_freqs reports the actual Hz of C4");

    loader.destroy_instance(inst);

    std::fprintf(stderr, "\n%s: %d failure(s)\n", failures == 0 ? "OK" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
