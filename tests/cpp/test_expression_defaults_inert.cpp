// Codifies the lesson: WavetableLayer's expression params (pressure_to_amp,
// timbre_to_position) MUST be inert when the note source emits no MPE
// pressure/timbre data. The math at src/wavetable_layer.cpp:532–536 is
// (1 + depth × slot.pressure); for slot.pressure = 0 (the non-MPE default)
// the multiplier is 1 regardless of depth. This test catches any future
// bug where slot.pressure is left in stale/uninitialized state, which
// would silently affect every voice in every preset.
//
// Concretely: render the same chord twice, once with pressure_to_amp=0
// and once with pressure_to_amp=0.5, using a non-MPE note source. The
// resulting audio buffers must be bit-identical. Same for timbre_to_position.

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

// Render a fixed scenario with the given param overrides and return the
// final block of audio data. Each call uses a fresh instance + fresh
// PolyTestContext so the runs are independent.
std::vector<float> render_scenario(MiniLoader& loader,
                                   const VividOperatorDescriptor* desc,
                                   int idx_pressure_to_amp,
                                   int idx_timbre_to_position,
                                   int idx_amplitude,
                                   int idx_position,
                                   float pressure_depth,
                                   float timbre_depth) {
    std::vector<float> params(desc->param_count);
    for (uint32_t p = 0; p < desc->param_count; ++p)
        params[p] = desc->params[p].default_value;
    if (idx_amplitude >= 0)            params[idx_amplitude]            = 0.2f;
    if (idx_position >= 0)             params[idx_position]             = 0.5f;
    if (idx_pressure_to_amp >= 0)      params[idx_pressure_to_amp]      = pressure_depth;
    if (idx_timbre_to_position >= 0)   params[idx_timbre_to_position]   = timbre_depth;

    void* inst = loader.create_instance();
    PolyTestContext tc;
    tc.set_output_channels(2);
    tc.ctx.param_values = params.data();
    tc.setup_wavetable_layer_voice(261.63f);  // C4

    // Burn in 4 blocks so the envelope is in sustain and any boot-time
    // transients have settled.
    for (int b = 0; b < 4; ++b) {
        tc.clear_output();
        loader.process_audio(inst, &tc.ctx);
        tc.ctx.time += static_cast<double>(tc.kFrames) / tc.kSampleRate;
        tc.ctx.frame++;
        tc.clear_notes();
    }

    // Capture a final sustain block as the comparison.
    tc.clear_output();
    loader.process_audio(inst, &tc.ctx);

    std::vector<float> capture(2 * tc.kFrames);
    std::memcpy(capture.data(), tc.output_buf, 2 * tc.kFrames * sizeof(float));

    loader.destroy_instance(inst);
    return capture;
}

bool buffers_identical(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::abs(a[i] - b[i]));
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
        idx_pressure_to_amp = -1, idx_timbre_to_position = -1;
    for (uint32_t p = 0; p < desc->param_count; ++p) {
        const char* n = desc->params[p].name;
        if (!std::strcmp(n, "amplitude"))           idx_amplitude = p;
        else if (!std::strcmp(n, "position"))       idx_position = p;
        else if (!std::strcmp(n, "pressure_to_amp")) idx_pressure_to_amp = p;
        else if (!std::strcmp(n, "timbre_to_position")) idx_timbre_to_position = p;
    }

    std::fprintf(stderr, "\n--- pressure_to_amp inert at non-MPE input ---\n");
    auto baseline = render_scenario(loader, desc, idx_pressure_to_amp, idx_timbre_to_position,
                                    idx_amplitude, idx_position,
                                    /*pressure_depth=*/0.0f, /*timbre_depth=*/0.0f);
    auto pressure_on = render_scenario(loader, desc, idx_pressure_to_amp, idx_timbre_to_position,
                                       idx_amplitude, idx_position,
                                       /*pressure_depth=*/0.5f, /*timbre_depth=*/0.0f);

    float diff_p = max_abs_diff(baseline, pressure_on);
    std::fprintf(stderr, "    max_abs_diff (pressure_to_amp 0 vs 0.5) = %.6e\n", diff_p);
    check(buffers_identical(baseline, pressure_on),
          "pressure_to_amp toggling produces bit-identical output for non-MPE input");

    std::fprintf(stderr, "\n--- timbre_to_position inert at non-MPE input ---\n");
    auto timbre_on = render_scenario(loader, desc, idx_pressure_to_amp, idx_timbre_to_position,
                                     idx_amplitude, idx_position,
                                     /*pressure_depth=*/0.0f, /*timbre_depth=*/0.5f);

    float diff_t = max_abs_diff(baseline, timbre_on);
    std::fprintf(stderr, "    max_abs_diff (timbre_to_position 0 vs 0.5) = %.6e\n", diff_t);
    check(buffers_identical(baseline, timbre_on),
          "timbre_to_position toggling produces bit-identical output for non-MPE input");

    std::fprintf(stderr, "\n%s: %d failure(s)\n", failures == 0 ? "OK" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
