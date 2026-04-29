// Codifies the lesson that WavetableLayer's stereo output bus must be
// DC-blocked: a sustained note across many blocks should produce a mean
// near zero, not accumulate any subsonic component. Regression for the
// "low wub on retriggered chord roots" symptom that the output_rumble_dc_
// stage in src/wavetable_layer.cpp:416–425 was added to fix.

#include "test_support.h"

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

float channel_mean(const float* buf, uint32_t count) {
    double sum = 0.0;
    for (uint32_t i = 0; i < count; ++i) sum += buf[i];
    return static_cast<float>(sum / count);
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

    int idx_amplitude = -1, idx_position = -1, idx_attack = -1, idx_decay = -1,
        idx_sustain = -1, idx_release = -1;
    for (uint32_t p = 0; p < desc->param_count; ++p) {
        const char* n = desc->params[p].name;
        if (!std::strcmp(n, "amplitude")) idx_amplitude = p;
        else if (!std::strcmp(n, "position")) idx_position = p;
        else if (!std::strcmp(n, "attack")) idx_attack = p;
        else if (!std::strcmp(n, "decay")) idx_decay = p;
        else if (!std::strcmp(n, "sustain")) idx_sustain = p;
        else if (!std::strcmp(n, "release")) idx_release = p;
    }

    std::vector<float> params(desc->param_count);
    for (uint32_t p = 0; p < desc->param_count; ++p)
        params[p] = desc->params[p].default_value;
    if (idx_amplitude >= 0) params[idx_amplitude] = 0.3f;
    if (idx_position >= 0)  params[idx_position]  = 0.5f;  // avoid the position=0 degenerate frame
    if (idx_attack >= 0)    params[idx_attack]    = 0.005f;
    if (idx_decay >= 0)     params[idx_decay]     = 0.05f;
    if (idx_sustain >= 0)   params[idx_sustain]   = 0.85f;
    if (idx_release >= 0)   params[idx_release]   = 0.5f;

    void* inst = loader.create_instance();
    PolyTestContext tc;
    tc.set_output_channels(2);
    tc.ctx.param_values = params.data();
    tc.setup_wavetable_layer_voice(261.63f);  // C4

    // Warm-up: render a few blocks so attack/decay finish and we're in sustain.
    // After that, the note continues to be gated on (push_note_on persists in
    // the runtime's allocator, not in our notes_buf — but the synth's gate
    // state holds until note_off).
    std::fprintf(stderr, "\n--- DC offset under sustained note ---\n");
    for (int b = 0; b < 4; ++b) {
        tc.clear_output();
        loader.process_audio(inst, &tc.ctx);
        tc.ctx.time += static_cast<double>(tc.kFrames) / tc.kSampleRate;
        tc.ctx.frame++;
        tc.clear_notes();  // don't re-trigger; allocator already holds the note
    }

    // Now measure DC over a fresh sustain block.
    tc.clear_output();
    loader.process_audio(inst, &tc.ctx);

    float left_mean  = channel_mean(tc.output_buf,                tc.kFrames);
    float right_mean = channel_mean(tc.output_buf + tc.kFrames,   tc.kFrames);
    std::fprintf(stderr, "    left_mean=%.6f right_mean=%.6f\n", left_mean, right_mean);

    // The DC blocker is a leaky integrator with coefficient 0.9948 — it
    // doesn't take the mean to literal zero, but it should keep it well
    // under any audible threshold. 1e-3 is generous (-60 dBFS).
    check(std::abs(left_mean)  < 1.0e-3f, "left channel mean is DC-blocked");
    check(std::abs(right_mean) < 1.0e-3f, "right channel mean is DC-blocked");

    loader.destroy_instance(inst);

    std::fprintf(stderr, "\n%s: %d failure(s)\n", failures == 0 ? "OK" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
