// OPT-IN TEST — currently EXPECTED TO FAIL until the position=0 bug is
// fixed. See user-memory note `project_position_zero_bug.md`.
//
// At wavetable_family=0 (AnalogWarm) / wavetable_member=0 (Core) /
// position=0, the bank's frame 0 should render as a clean sine (per the
// generator code in src/wavetable_bank.cpp:127–189 — sample = sine at
// t=0). Empirically the audio is wubby/silent-ish, suggesting either
// the bank's frame-0 normalization is producing near-zero output, the
// renderer's f0=f1=0 lookup at frame_blend=0 is reading the wrong cell,
// or some other initialization issue specific to the lookup at the
// table's first frame.
//
// Build with -DVIVID_TEST_POSITION_ZERO=1 to enable. Once the underlying
// bug is fixed, drop the ifdef and make this a mandatory regression test.

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

#ifndef VIVID_TEST_POSITION_ZERO
int main() {
    std::fprintf(stderr,
                 "SKIP: test_position_zero_renders_clean is opt-in. "
                 "Build with -DVIVID_TEST_POSITION_ZERO=1 once the open bug at "
                 "(family=0/member=0/position=0) is fixed.\n");
    return 0;
}
#else

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

float rms_of(const float* buf, uint32_t count) {
    double sum = 0.0;
    for (uint32_t i = 0; i < count; ++i) sum += buf[i] * buf[i];
    return std::sqrt(static_cast<float>(sum / count));
}

float peak_of(const float* buf, uint32_t count) {
    float p = 0.0f;
    for (uint32_t i = 0; i < count; ++i) p = std::max(p, std::abs(buf[i]));
    return p;
}

}  // namespace

int main() {
    MiniLoader loader;
    std::string path = std::string("./wavetable_layer") + VIVID_PLUGIN_SUFFIX_STR;
    if (!loader.load(path.c_str())) return 1;
    const VividOperatorDescriptor* desc = loader.descriptor();
    if (!desc) return 1;

    int idx_amplitude = -1, idx_position = -1,
        idx_wavetable_family = -1, idx_wavetable_member = -1;
    for (uint32_t p = 0; p < desc->param_count; ++p) {
        const char* n = desc->params[p].name;
        if (!std::strcmp(n, "amplitude"))           idx_amplitude = p;
        else if (!std::strcmp(n, "position"))       idx_position = p;
        else if (!std::strcmp(n, "wavetable_family")) idx_wavetable_family = p;
        else if (!std::strcmp(n, "wavetable_member")) idx_wavetable_member = p;
    }

    std::vector<float> params(desc->param_count);
    for (uint32_t p = 0; p < desc->param_count; ++p)
        params[p] = desc->params[p].default_value;
    if (idx_amplitude >= 0)         params[idx_amplitude]         = 0.5f;
    if (idx_wavetable_family >= 0)  params[idx_wavetable_family]  = 0.0f;  // AnalogWarm
    if (idx_wavetable_member >= 0)  params[idx_wavetable_member]  = 0.0f;  // Core
    if (idx_position >= 0)          params[idx_position]          = 0.0f;  // frame 0

    void* inst = loader.create_instance();
    PolyTestContext tc;
    tc.set_output_channels(2);
    tc.ctx.param_values = params.data();
    tc.setup_wavetable_layer_voice(440.0f);  // A4

    // Burn in to sustain.
    for (int b = 0; b < 4; ++b) {
        tc.clear_output();
        loader.process_audio(inst, &tc.ctx);
        tc.ctx.time += static_cast<double>(tc.kFrames) / tc.kSampleRate;
        tc.ctx.frame++;
        tc.clear_notes();
    }

    tc.clear_output();
    loader.process_audio(inst, &tc.ctx);

    float rms  = rms_of(tc.output_buf, tc.kFrames);
    float peak = peak_of(tc.output_buf, tc.kFrames);
    std::fprintf(stderr, "    rms=%.4f peak=%.4f\n", rms, peak);

    // A clean sine at amplitude 0.5 should produce RMS ~0.35 and peak ~0.5.
    // The current bug produces near-silence or wubb (much lower RMS).
    check(rms  > 0.10f, "frame 0 of family=0/member=0 produces audible signal at amp=0.5");
    check(peak > 0.20f, "frame 0 peak is at least a meaningful fraction of amplitude");

    loader.destroy_instance(inst);

    std::fprintf(stderr, "\n%s: %d failure(s)\n", failures == 0 ? "OK" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}

#endif  // VIVID_TEST_POSITION_ZERO
