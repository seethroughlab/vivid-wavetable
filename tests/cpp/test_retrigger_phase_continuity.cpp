// Codifies the lesson: when the same pitch is retriggered while the
// previous instance is still in release, WavetableLayer carries phase
// (and envelope state) across the boundary instead of resetting to
// phase=0. This was the fix for the audible click on every retriggered
// chord root — see src/wavetable_layer.cpp:487–516 (the carry_slot
// path in process_audio_midi).
//
// Test approach: render two scenarios and compare the maximum
// sample-to-sample delta in a window around the retrigger boundary:
//   A) sustained: one note held continuously
//   B) retriggered: same note gets note_off + note_on (different note_id)
//      while the previous instance is still ringing in release
// With the fix, both should look smooth (low max delta). Without the
// fix, scenario B would show a sharp discontinuity at the boundary
// (phase jumps from somewhere mid-cycle back to 0 → DC pulse → audible
// pop, even with the output DC blocker downstream).
//
// We use a generous comparison threshold because exact bit-equality
// across a retrigger isn't expected even with the fix (the new note
// gets a fresh declick ramp).

#include "test_support.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

// Maximum sample-to-sample absolute delta in [start, start+window).
float max_delta(const float* buf, uint32_t start, uint32_t window, uint32_t total) {
    if (start + window > total) window = total - start;
    if (window < 2) return 0.0f;
    float m = 0.0f;
    for (uint32_t i = start + 1; i < start + window; ++i) {
        m = std::max(m, std::abs(buf[i] - buf[i - 1]));
    }
    return m;
}

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

    int idx_amplitude = -1, idx_position = -1,
        idx_attack = -1, idx_decay = -1, idx_sustain = -1, idx_release = -1,
        idx_phase_reset_mode = -1;
    for (uint32_t p = 0; p < desc->param_count; ++p) {
        const char* n = desc->params[p].name;
        if (!std::strcmp(n, "amplitude"))         idx_amplitude = p;
        else if (!std::strcmp(n, "position"))     idx_position = p;
        else if (!std::strcmp(n, "attack"))       idx_attack = p;
        else if (!std::strcmp(n, "decay"))        idx_decay = p;
        else if (!std::strcmp(n, "sustain"))      idx_sustain = p;
        else if (!std::strcmp(n, "release"))      idx_release = p;
        else if (!std::strcmp(n, "phase_reset_mode")) idx_phase_reset_mode = p;
    }

    std::vector<float> params(desc->param_count);
    for (uint32_t p = 0; p < desc->param_count; ++p)
        params[p] = desc->params[p].default_value;
    if (idx_amplitude >= 0) params[idx_amplitude] = 0.3f;
    if (idx_position >= 0)  params[idx_position]  = 0.5f;  // avoid degenerate frame 0
    if (idx_attack >= 0)    params[idx_attack]    = 0.005f;
    if (idx_decay >= 0)     params[idx_decay]     = 0.05f;
    if (idx_sustain >= 0)   params[idx_sustain]   = 0.85f;
    if (idx_release >= 0)   params[idx_release]   = 0.5f;  // long release so the
                                                            // first instance is
                                                            // still ringing when
                                                            // the second triggers
    if (idx_phase_reset_mode >= 0) params[idx_phase_reset_mode] = 0.0f;  // FreeRun

    auto run_scenario = [&](bool retrigger_midway) {
        void* inst = loader.create_instance();
        PolyTestContext tc;
        tc.set_output_channels(2);
        tc.ctx.param_values = params.data();
        tc.setup_wavetable_layer_voice(261.63f);  // C4

        // Burn in 2 blocks to reach sustain.
        for (int b = 0; b < 2; ++b) {
            tc.clear_output();
            loader.process_audio(inst, &tc.ctx);
            tc.ctx.time += static_cast<double>(tc.kFrames) / tc.kSampleRate;
            tc.ctx.frame++;
            tc.clear_notes();
        }

        if (retrigger_midway) {
            // Release the held note (id 100, the first push_note_on assignment)
            // and immediately retrigger the same pitch. The previous voice is
            // still in release, so the carry-slot path in process_audio_midi
            // should fire and preserve phase + envelope state.
            tc.clear_notes();
            tc.push_note_off(100);
            tc.push_note_on(static_cast<uint8_t>(PolyTestContext::freq_to_midi(261.63f)));
        }

        // Render the boundary block — this is the one we measure.
        tc.clear_output();
        loader.process_audio(inst, &tc.ctx);

        std::vector<float> capture(tc.kFrames);
        std::memcpy(capture.data(), tc.output_buf, tc.kFrames * sizeof(float));

        loader.destroy_instance(inst);
        return capture;
    };

    std::fprintf(stderr, "\n--- Same-note retrigger phase continuity ---\n");
    auto sustained   = run_scenario(/*retrigger_midway=*/false);
    auto retriggered = run_scenario(/*retrigger_midway=*/true);

    // The retrigger boundary is at sample 0 of this block (the new note-on
    // arrives with frame_offset 0). Examine the first 256 samples — that
    // covers the declick window plus a few cycles of the wavetable.
    constexpr uint32_t kWindow = 256;
    float sustained_max   = max_delta(sustained.data(),   0, kWindow, sustained.size());
    float retriggered_max = max_delta(retriggered.data(), 0, kWindow, retriggered.size());
    std::fprintf(stderr, "    max sample-to-sample delta: sustained=%.5f retriggered=%.5f\n",
                 sustained_max, retriggered_max);

    // Threshold: retriggered shouldn't be more than 4× the sustained baseline.
    // Without the carry-slot fix, the phase reset would produce a much larger
    // jump (on the order of full wavetable peak-to-peak per sample, vs the
    // smooth ~0.01–0.05 per-sample slope of a normal sine in sustain).
    check(retriggered_max < std::max(sustained_max * 4.0f, 0.05f),
          "retriggered first-block max delta is comparable to sustained baseline");

    std::fprintf(stderr, "\n%s: %d failure(s)\n", failures == 0 ? "OK" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
