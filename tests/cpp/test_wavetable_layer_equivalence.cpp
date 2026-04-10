// Scalar vs SIMD equivalence tests for WavetableLayer renderer.
// Calls both backends directly with identical input and compares stereo output.
// Only meaningful when VIVID_HAS_HIGHWAY is defined.

#include "wavetable_layer_renderer.h"
#include "wavetable_voice_utils.h"
#include "wavetable_bank.h"
#include "wavetable_dsp.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace vivid_wavetable::layer;
using namespace vivid_wavetable::bank;

static int failures = 0;

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "  FAIL: %s\n", msg);
        ++failures;
    } else {
        std::fprintf(stderr, "  PASS: %s\n", msg);
    }
}

static float rms_diff(const float* a, const float* b, uint32_t count) {
    double sum = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(static_cast<float>(sum / count));
}

static float max_diff(const float* a, const float* b, uint32_t count) {
    float mx = 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
        float d = std::abs(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static float rms_of(const float* buf, uint32_t count) {
    double sum = 0.0;
    for (uint32_t i = 0; i < count; ++i) sum += buf[i] * buf[i];
    return std::sqrt(static_cast<float>(sum / count));
}

// Build a simple test wavetable (single sine frame)
static Wavetable make_test_wavetable() {
    Wavetable wt;
    wt.allocate(4); // 4 frames for position interpolation
    for (uint32_t fr = 0; fr < wt.frame_count; ++fr) {
        float* frame = wt.frame_ptr(fr);
        float harmonic_scale = 1.0f + static_cast<float>(fr) * 0.5f;
        for (uint32_t i = 0; i < kSamplesPerFrame; ++i) {
            float phase = static_cast<float>(i) / static_cast<float>(kSamplesPerFrame);
            frame[i] = std::sin(2.0f * static_cast<float>(M_PI) * phase * harmonic_scale);
        }
    }
    wt.build_mipmaps();
    return wt;
}

struct EquivTestCase {
    const char* name;
    int voice_count;
    int num_unison;
    float freq;
    int warp_mode;
    float warp_amount;
    bool drift;
    float position;
    bool has_voice_gain;
    bool has_pitch_mod;
};

#ifdef VIVID_HAS_HIGHWAY

static void run_equivalence(const EquivTestCase& tc, const Wavetable& wt) {
    std::fprintf(stderr, "\n  [%s]\n", tc.name);

    constexpr uint32_t kFrames = 256;
    constexpr float kSampleRate = 48000.0f;

    PreparedWavetable pwt;
    pwt.prepare_from(wt);

    RenderParams params{};
    params.warp_mode = tc.warp_mode;
    params.amplitude = 0.3f;
    params.position_base = tc.position;
    params.warp_base = tc.warp_amount;
    params.drift_amount = tc.drift ? 0.5f : 0.0f;
    params.drift_rate_hz = 1.0f;
    params.drift_enabled = tc.drift;
    params.num_unison = tc.num_unison;
    params.unison_spread = 20.0f;
    params.unison_stereo = 1.0f;
    params.phase_reset_mode = 1; // Reset for determinism
    params.stereo_phase_offset = 0.25f;

    float unison_gain = params.amplitude / std::sqrt(static_cast<float>(params.num_unison));

    // Build identical render units for both backends
    auto build_ru = [&]() {
        RenderUnit ru{};
        ru.clear();
        int slot = 0;
        for (int vi = 0; vi < tc.voice_count; ++vi) {
            float base_freq = tc.freq * (1.0f + vi * 0.25f); // slightly different freqs
            for (int ui = 0; ui < tc.num_unison; ++ui) {
                float det_offset = vivid_wavetable::voice::unison_detune_offset(
                    ui, tc.num_unison, params.unison_spread, 0, vi + 1);
                float detune_ratio = vivid_wavetable::dsp::cents_to_ratio(det_offset);
                float unit_freq = base_freq * detune_ratio;

                ru.phase[slot] = 0.0f;
                ru.phase_inc[slot] = unit_freq / kSampleRate;
                ru.drift_phase[slot] = vivid_wavetable::voice::hash01(
                    static_cast<uint32_t>(vi + 1 + ui * 211)) * 2.0f * static_cast<float>(M_PI);
                ru.drift_phase_inc[slot] = params.drift_rate_hz * 2.0f * static_cast<float>(M_PI) / kSampleRate;
                ru.mip_level[slot] = quantized_mip_level(unit_freq, kSampleRate);

                float pan_pos = vivid_wavetable::voice::unison_pan_position(
                    ui, tc.num_unison, params.unison_stereo);
                float angle = (pan_pos * 0.5f + 0.5f) * static_cast<float>(M_PI) * 0.5f;
                ru.pan_l[slot] = std::cos(angle);
                ru.pan_r[slot] = std::sin(angle);
                ru.gain[slot] = unison_gain;
                ru.voice_idx[slot] = vi;
                ++slot;
            }
        }
        ru.active_count = slot;
        // Zero-pad
        for (int i = slot; i < slot + 16 && i < kMaxRenderUnits; ++i) {
            ru.phase[i] = 0.0f; ru.phase_inc[i] = 0.0f;
            ru.gain[i] = 0.0f; ru.pan_l[i] = 0.0f; ru.pan_r[i] = 0.0f;
        }
        return ru;
    };

    // Build identical voice blocks
    float voice_gain_buf[kFrames];
    float pitch_mod_buf[kFrames];
    for (uint32_t i = 0; i < kFrames; ++i) {
        voice_gain_buf[i] = 0.5f + 0.5f * static_cast<float>(i) / static_cast<float>(kFrames);
        pitch_mod_buf[i] = std::sin(static_cast<float>(i) * 0.01f) * 0.5f;
    }

    auto build_vb = [&]() {
        VoiceBlock vb{};
        vb.voice_count = tc.voice_count;
        for (int v = 0; v < tc.voice_count; ++v) {
            vb.pos_from[v] = params.position_base;
            vb.pos_to[v] = params.position_base;
            vb.warp_from[v] = params.warp_base;
            vb.warp_to[v] = params.warp_base;
            vb.declick_remaining[v] = 0;
            if (tc.has_voice_gain) vb.voice_gain_audio[v] = voice_gain_buf;
            if (tc.has_pitch_mod) vb.pitch_mod_audio[v] = pitch_mod_buf;
        }
        return vb;
    };

    // Run scalar
    float out_scalar[2 * kFrames] = {};
    {
        RenderUnit ru = build_ru();
        VoiceBlock vb = build_vb();
        render_block_scalar(out_scalar, kFrames, kSampleRate, ru, vb, pwt, params);
    }

    // Run SIMD
    float out_simd[2 * kFrames] = {};
    {
        RenderUnit ru = build_ru();
        VoiceBlock vb = build_vb();
        render_block_simd(out_simd, kFrames, kSampleRate, ru, vb, pwt, params);
    }

    // Compare
    float rms_l_scalar = rms_of(out_scalar, kFrames);
    float rms_l_simd = rms_of(out_simd, kFrames);
    float rd_l = rms_diff(out_scalar, out_simd, kFrames);
    float rd_r = rms_diff(out_scalar + kFrames, out_simd + kFrames, kFrames);
    float md_l = max_diff(out_scalar, out_simd, kFrames);
    float md_r = max_diff(out_scalar + kFrames, out_simd + kFrames, kFrames);

    std::fprintf(stderr, "    scalar_rms=%.6f simd_rms=%.6f\n", rms_l_scalar, rms_l_simd);
    std::fprintf(stderr, "    rms_diff L=%.8f R=%.8f  max_diff L=%.8f R=%.8f\n",
                 rd_l, rd_r, md_l, md_r);

    check(rms_l_scalar > 0.001f, "scalar produces signal");
    check(rms_l_simd > 0.001f, "simd produces signal");

    // Tolerance: float rounding from operation reordering
    // ReduceSum in SIMD may accumulate in different order than scalar
    check(rd_l < 1e-4f, "left channel RMS difference < 1e-4");
    check(rd_r < 1e-4f, "right channel RMS difference < 1e-4");
    check(md_l < 1e-3f, "left channel max sample difference < 1e-3");
    check(md_r < 1e-3f, "right channel max sample difference < 1e-3");
}

#endif // VIVID_HAS_HIGHWAY

int main() {
#ifdef VIVID_HAS_HIGHWAY
    std::fprintf(stderr, "--- Scalar/SIMD equivalence tests ---\n");

    Wavetable wt = make_test_wavetable();

    EquivTestCase cases[] = {
        {"1 voice, no mod",       1, 1, 440.0f, 0, 0.0f, false, 0.0f,  false, false},
        {"4 voices, 1 unison",    4, 1, 440.0f, 0, 0.0f, false, 0.3f,  false, false},
        {"2 voices, 4 unison",    2, 4, 330.0f, 0, 0.0f, false, 0.5f,  false, false},
        {"4 voices, 4 unison",    4, 4, 261.0f, 0, 0.0f, false, 0.2f,  false, false},
        {"warp sync",             2, 2, 440.0f, 1, 0.5f, false, 0.0f,  false, false},
        {"warp quantize",         2, 2, 440.0f, 6, 0.5f, false, 0.0f,  false, false},
        {"warp flip",             2, 2, 440.0f, 7, 0.5f, false, 0.0f,  false, false},
        {"warp mirror",           2, 2, 440.0f, 4, 0.5f, false, 0.0f,  false, false},
        {"warp asym",             2, 2, 440.0f, 5, 0.5f, false, 0.0f,  false, false},
        {"warp bend+",            2, 2, 440.0f, 2, 0.5f, false, 0.0f,  false, false},
        {"warp bend-",            2, 2, 440.0f, 3, 0.5f, false, 0.0f,  false, false},
        {"simd kernel no-warp+drift",  2, 2, 440.0f, 0, 0.0f, true,  0.0f, false, false},
        {"simd kernel warp+no-drift",  2, 2, 440.0f, 1, 0.5f, false, 0.2f, false, false},
        {"simd kernel warp+drift",     2, 2, 440.0f, 1, 0.5f, true,  0.2f, false, false},
        {"drift enabled",         2, 2, 440.0f, 0, 0.0f, true,  0.0f,  false, false},
        {"voice_gain_audio",      2, 2, 440.0f, 0, 0.0f, false, 0.0f,  true,  false},
        {"pitch_mod_audio",       2, 2, 440.0f, 0, 0.0f, false, 0.0f,  false, true},
        {"all mods + warp + drift", 4, 4, 261.0f, 1, 0.5f, true, 0.4f, true, true},
    };

    for (const auto& tc : cases) {
        run_equivalence(tc, wt);
    }

    std::fprintf(stderr, "\n");
    if (failures == 0) {
        std::printf("Scalar/SIMD equivalence: ALL PASS (%zu cases)\n",
                    sizeof(cases) / sizeof(cases[0]));
    } else {
        std::fprintf(stderr, "Scalar/SIMD equivalence: %d FAILURES\n", failures);
    }
    return failures > 0 ? 1 : 0;

#else
    std::printf("Highway not available — scalar/SIMD equivalence tests skipped: PASS\n");
    return 0;
#endif
}
