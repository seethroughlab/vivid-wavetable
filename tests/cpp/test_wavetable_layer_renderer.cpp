// Renderer correctness tests for WavetableLayer.
// Validates the scalar backend against spectral, amplitude, and behavioral properties.

#include "test_support.h"

#include <array>
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

static int failures = 0;

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "  FAIL: %s\n", msg);
        ++failures;
    } else {
        std::fprintf(stderr, "  PASS: %s\n", msg);
    }
}

static void check_float(float actual, float expected, float tol, const char* msg) {
    if (std::abs(actual - expected) > tol) {
        std::fprintf(stderr, "  FAIL: %s (actual=%.6f, expected=%.6f, tol=%.6f)\n",
                     msg, actual, expected, tol);
        ++failures;
    } else {
        std::fprintf(stderr, "  PASS: %s (%.6f)\n", msg, actual);
    }
}

static float rms_of(const float* buf, uint32_t count) {
    double sum = 0.0;
    for (uint32_t i = 0; i < count; ++i) sum += buf[i] * buf[i];
    return std::sqrt(static_cast<float>(sum / count));
}

static float peak_of(const float* buf, uint32_t count) {
    float peak = 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
        float a = std::abs(buf[i]);
        if (a > peak) peak = a;
    }
    return peak;
}

// Count zero crossings to estimate fundamental frequency
static float estimate_frequency(const float* buf, uint32_t count, float sample_rate) {
    int crossings = 0;
    for (uint32_t i = 1; i < count; ++i) {
        if ((buf[i - 1] >= 0.0f) != (buf[i] >= 0.0f)) ++crossings;
    }
    return static_cast<float>(crossings) * sample_rate / (2.0f * static_cast<float>(count));
}

// Check if two buffers are bit-identical
static bool buffers_identical(const float* a, const float* b, uint32_t count) {
    return std::memcmp(a, b, count * sizeof(float)) == 0;
}

static float average_abs_diff(const float* a, const float* b, uint32_t count) {
    double sum = 0.0;
    for (uint32_t i = 0; i < count; ++i) sum += std::abs(a[i] - b[i]);
    return static_cast<float>(sum / count);
}

struct LayerTestHarness {
    MiniLoader loader;
    const VividOperatorDescriptor* desc = nullptr;

    // Param indices (resolved by name)
    int idx_amplitude = -1;
    int idx_position = -1;
    int idx_warp_mode = -1;
    int idx_warp_amount = -1;
    int idx_unison_voices = -1;
    int idx_unison_spread = -1;
    int idx_unison_stereo = -1;
    int idx_drift_amount = -1;
    int idx_drift_rate_hz = -1;
    int idx_phase_reset_mode = -1;
    int idx_detune = -1;
    int idx_portamento = -1;
    int idx_position_smooth_ms = -1;
    int idx_warp_smooth_ms = -1;

    bool load() {
        std::string path = std::string("./wavetable_layer") + VIVID_PLUGIN_SUFFIX_STR;
        if (!loader.load(path.c_str())) {
            std::fprintf(stderr, "FAIL: could not load %s\n", path.c_str());
            return false;
        }
        desc = loader.descriptor();
        if (!desc) return false;

        for (uint32_t p = 0; p < desc->param_count; ++p) {
            const char* n = desc->params[p].name;
            if (!std::strcmp(n, "amplitude")) idx_amplitude = p;
            else if (!std::strcmp(n, "position")) idx_position = p;
            else if (!std::strcmp(n, "warp_mode")) idx_warp_mode = p;
            else if (!std::strcmp(n, "warp_amount")) idx_warp_amount = p;
            else if (!std::strcmp(n, "unison_voices")) idx_unison_voices = p;
            else if (!std::strcmp(n, "unison_spread")) idx_unison_spread = p;
            else if (!std::strcmp(n, "unison_stereo")) idx_unison_stereo = p;
            else if (!std::strcmp(n, "drift_amount")) idx_drift_amount = p;
            else if (!std::strcmp(n, "drift_rate_hz")) idx_drift_rate_hz = p;
            else if (!std::strcmp(n, "phase_reset_mode")) idx_phase_reset_mode = p;
            else if (!std::strcmp(n, "detune")) idx_detune = p;
            else if (!std::strcmp(n, "portamento")) idx_portamento = p;
            else if (!std::strcmp(n, "position_smooth_ms")) idx_position_smooth_ms = p;
            else if (!std::strcmp(n, "warp_smooth_ms")) idx_warp_smooth_ms = p;
        }
        return true;
    }

    std::vector<float> default_params() {
        std::vector<float> params(desc->param_count);
        for (uint32_t p = 0; p < desc->param_count; ++p)
            params[p] = desc->params[p].default_value;
        return params;
    }

    // Run N blocks, return metrics from last block.
    // Caller sets up tc (voice, params, audio inputs) before calling.
    vivid::AudioMetrics run_blocks(void* inst, PolyTestContext& tc, int blocks = 6) {
        for (int b = 0; b < blocks; ++b) {
            tc.clear_output();
            loader.process_audio(inst, &tc.ctx);
            tc.ctx.time += static_cast<double>(tc.kFrames) / tc.kSampleRate;
            tc.ctx.frame++;
        }
        return tc.analyze_output(2);
    }
};

// ===========================================================================
// Test: Single voice fundamental frequency
// ===========================================================================
static void test_single_voice_fundamental(LayerTestHarness& h) {
    std::fprintf(stderr, "\n--- Single voice fundamental ---\n");

    void* inst = h.loader.create_instance();
    auto params = h.default_params();
    if (h.idx_amplitude >= 0) params[h.idx_amplitude] = 0.5f;

    PolyTestContext tc;
    tc.set_output_channels(2);
    tc.ctx.param_values = params.data();
    tc.setup_wavetable_layer_voice(440.0f);

    auto m = h.run_blocks(inst, tc);
    std::fprintf(stderr, "    rms=%.4f centroid=%.1fHz\n", m.rms, m.spectral_centroid_hz);

    check(m.rms > 0.01f, "produces signal");
    // Spectral centroid reflects energy-weighted harmonic content, not just the fundamental.
    // For a wavetable with harmonics, centroid is typically above the fundamental.
    check(m.spectral_centroid_hz > 300.0f, "spectral centroid above 300Hz");
    check(m.spectral_centroid_hz < 3000.0f, "spectral centroid below 3000Hz");

    // Zero-crossing gives a tighter fundamental estimate
    float est_freq = estimate_frequency(tc.output_buf, tc.kFrames, tc.kSampleRate);
    check_float(est_freq, 440.0f, 40.0f, "zero-crossing frequency near 440Hz");

    h.loader.destroy_instance(inst);
}

// ===========================================================================
// Test: Stereo pan with unison
// ===========================================================================
static void test_stereo_pan(LayerTestHarness& h) {
    std::fprintf(stderr, "\n--- Stereo pan with unison ---\n");

    void* inst = h.loader.create_instance();
    auto params = h.default_params();
    if (h.idx_amplitude >= 0) params[h.idx_amplitude] = 0.5f;
    if (h.idx_unison_voices >= 0) params[h.idx_unison_voices] = 4.0f;
    if (h.idx_unison_spread >= 0) params[h.idx_unison_spread] = 20.0f;
    if (h.idx_unison_stereo >= 0) params[h.idx_unison_stereo] = 1.0f;

    PolyTestContext tc;
    tc.set_output_channels(2);
    tc.ctx.param_values = params.data();
    tc.setup_wavetable_layer_voice(440.0f);

    h.run_blocks(inst, tc);

    float left_rms = rms_of(tc.output_buf, tc.kFrames);
    float right_rms = rms_of(tc.output_buf + tc.kFrames, tc.kFrames);

    std::fprintf(stderr, "    left_rms=%.4f right_rms=%.4f\n", left_rms, right_rms);

    check(left_rms > 0.01f, "left channel has signal");
    check(right_rms > 0.01f, "right channel has signal");

    // With stereo spread, L and R should differ
    bool lr_differ = !buffers_identical(tc.output_buf, tc.output_buf + tc.kFrames, tc.kFrames);
    check(lr_differ, "L and R channels differ with stereo unison");

    h.loader.destroy_instance(inst);
}

// ===========================================================================
// Test: Gate-on phase reset
// ===========================================================================
static void test_gate_on_reset(LayerTestHarness& h) {
    std::fprintf(stderr, "\n--- Gate-on phase reset ---\n");

    auto params = h.default_params();
    if (h.idx_amplitude >= 0) params[h.idx_amplitude] = 0.5f;
    if (h.idx_phase_reset_mode >= 0) params[h.idx_phase_reset_mode] = 1.0f; // Reset mode

    // Run two separate instances, both gating on from scratch at 440Hz.
    // With phase reset, they should produce identical output.
    float out_a[PolyTestContext::kMaxAudioChannels * PolyTestContext::kFrames] = {};
    float out_b[PolyTestContext::kMaxAudioChannels * PolyTestContext::kFrames] = {};

    for (int trial = 0; trial < 2; ++trial) {
        void* inst = h.loader.create_instance();
        PolyTestContext tc;
        tc.set_output_channels(2);
        tc.ctx.param_values = params.data();
        tc.setup_wavetable_layer_voice(440.0f);

        tc.clear_output();
        h.loader.process_audio(inst, &tc.ctx);

        float* dst = (trial == 0) ? out_a : out_b;
        std::memcpy(dst, tc.output_buf, 2 * tc.kFrames * sizeof(float));
        h.loader.destroy_instance(inst);
    }

    bool identical = buffers_identical(out_a, out_b, 2 * PolyTestContext::kFrames);
    check(identical, "two fresh instances with phase reset produce identical first block");
}

// ===========================================================================
// Test: voice_gain_audio envelope
// ===========================================================================
static void test_voice_gain_audio(LayerTestHarness& h) {
    std::fprintf(stderr, "\n--- voice_gain_audio envelope ---\n");

    void* inst = h.loader.create_instance();
    auto params = h.default_params();
    if (h.idx_amplitude >= 0) params[h.idx_amplitude] = 0.5f;

    // Create a gain ramp: 0→1 over the buffer
    float gain_ramp[PolyTestContext::kFrames];
    for (int i = 0; i < PolyTestContext::kFrames; ++i)
        gain_ramp[i] = static_cast<float>(i) / static_cast<float>(PolyTestContext::kFrames - 1);

    PolyTestContext tc;
    tc.set_output_channels(2);
    tc.ctx.param_values = params.data();
    tc.setup_wavetable_layer_voice(440.0f);
    tc.clear_audio_inputs();
    // WavetableLayer port layout: midi_in=0, lanes=1-7, pitch_mod_audio=8,
    // position_mod_audio=9, warp_mod_audio=10, voice_gain_audio=11.
    tc.bind_audio_input(11, gain_ramp, 1);

    // Run a few blocks to stabilize
    for (int b = 0; b < 4; ++b) {
        tc.clear_output();
        h.loader.process_audio(inst, &tc.ctx);
        tc.ctx.time += static_cast<double>(tc.kFrames) / tc.kSampleRate;
        tc.ctx.frame++;
    }

    // Check: first quarter should be quieter than last quarter
    uint32_t quarter = tc.kFrames / 4;
    float rms_start = rms_of(tc.output_buf, quarter);
    float rms_end = rms_of(tc.output_buf + 3 * quarter, quarter);

    std::fprintf(stderr, "    rms_start=%.4f rms_end=%.4f\n", rms_start, rms_end);
    check(rms_end > rms_start * 1.5f, "output amplitude follows voice_gain_audio ramp");

    h.loader.destroy_instance(inst);
}

// ===========================================================================
// Test: Multi-voice mixing
// ===========================================================================
static void test_multi_voice(LayerTestHarness& h) {
    std::fprintf(stderr, "\n--- Multi-voice mixing ---\n");

    void* inst = h.loader.create_instance();
    auto params = h.default_params();
    if (h.idx_amplitude >= 0) params[h.idx_amplitude] = 0.3f;

    PolyTestContext tc;
    tc.set_output_channels(2);
    tc.ctx.param_values = params.data();

    // Set up 4 voices at different frequencies
    tc.clear_lane_ports();
    float freqs[4] = {261.63f, 329.63f, 392.0f, 523.25f}; // C4, E4, G4, C5
    float gates[4] = {1, 1, 1, 1};
    float vels[4] = {1, 1, 1, 1};
    float lids[4] = {1, 2, 3, 4};
    float pitch_mod[4] = {0, 0, 0, 0};
    float pos_mod[4] = {0, 0, 0, 0};
    float warp_mod[4] = {0, 0, 0, 0};
    // WavetableLayer port layout: midi_in=0, frequencies=1, gates=2,
    // velocities=3, lane_ids=4, pitch_mod=5, position_mod=6, warp_mod=7.
    tc.bind_lane(1, freqs, 4);
    tc.bind_lane(2, gates, 4);
    tc.bind_lane(3, vels, 4);
    tc.bind_lane(4, lids, 4);
    tc.bind_lane(5, pitch_mod, 4);
    tc.bind_lane(6, pos_mod, 4);
    tc.bind_lane(7, warp_mod, 4);

    auto m = h.run_blocks(inst, tc);
    std::fprintf(stderr, "    rms=%.4f centroid=%.1fHz\n", m.rms, m.spectral_centroid_hz);

    check(m.rms > 0.01f, "multi-voice produces signal");
    // Centroid should be somewhere between 261Hz and 523Hz (weighted average)
    check(m.spectral_centroid_hz > 200.0f, "centroid above lowest note");
    check(m.spectral_centroid_hz < 800.0f, "centroid below 2x highest note");

    // Compare single voice vs 4 voices — multi should be louder
    void* inst_single = h.loader.create_instance();
    PolyTestContext tc_single;
    tc_single.set_output_channels(2);
    tc_single.ctx.param_values = params.data();
    tc_single.setup_wavetable_layer_voice(261.63f);
    auto m_single = h.run_blocks(inst_single, tc_single);

    check(m.rms > m_single.rms * 0.8f, "4 voices at least as loud as 1 voice");

    h.loader.destroy_instance(inst_single);
    h.loader.destroy_instance(inst);
}

// ===========================================================================
// Test: Drift causes variation
// ===========================================================================
static void test_drift(LayerTestHarness& h) {
    std::fprintf(stderr, "\n--- Drift variation ---\n");

    auto params = h.default_params();
    if (h.idx_amplitude >= 0) params[h.idx_amplitude] = 0.5f;
    if (h.idx_drift_amount >= 0) params[h.idx_drift_amount] = 0.5f;
    if (h.idx_drift_rate_hz >= 0) params[h.idx_drift_rate_hz] = 1.0f;

    void* inst = h.loader.create_instance();
    PolyTestContext tc;
    tc.set_output_channels(2);
    tc.ctx.param_values = params.data();
    tc.setup_wavetable_layer_voice(440.0f);

    // Run 10 blocks, capture block 5 and block 10
    float block5[PolyTestContext::kFrames] = {};
    float block10[PolyTestContext::kFrames] = {};
    for (int b = 0; b < 10; ++b) {
        tc.clear_output();
        h.loader.process_audio(inst, &tc.ctx);
        tc.ctx.time += static_cast<double>(tc.kFrames) / tc.kSampleRate;
        tc.ctx.frame++;
        if (b == 4) std::memcpy(block5, tc.output_buf, tc.kFrames * sizeof(float));
        if (b == 9) std::memcpy(block10, tc.output_buf, tc.kFrames * sizeof(float));
    }

    bool differ = !buffers_identical(block5, block10, PolyTestContext::kFrames);
    check(differ, "drift causes variation between blocks");

    h.loader.destroy_instance(inst);
}

// ===========================================================================
// Test: lane-rate pitch modulation affects rendered pitch
// ===========================================================================
static void test_lane_pitch_mod(LayerTestHarness& h) {
    std::fprintf(stderr, "\n--- Lane pitch modulation ---\n");

    auto run_with_pitch_mod = [&](float pitch_mod_semitones) -> float {
        void* inst = h.loader.create_instance();
        auto params = h.default_params();
        if (h.idx_amplitude >= 0) params[h.idx_amplitude] = 0.5f;

        PolyTestContext tc;
        tc.set_output_channels(2);
        tc.ctx.param_values = params.data();
        tc.setup_wavetable_layer_voice(220.0f);
        tc.pitch_mod_lane_data[0] = pitch_mod_semitones;

        h.run_blocks(inst, tc, 4);
        float est_freq = estimate_frequency(tc.output_buf, tc.kFrames, tc.kSampleRate);
        h.loader.destroy_instance(inst);
        return est_freq;
    };

    float base = run_with_pitch_mod(0.0f);
    float octave = run_with_pitch_mod(12.0f);
    std::fprintf(stderr, "    base=%.1fHz octave=%.1fHz\n", base, octave);

    check(octave > base * 1.6f, "lane pitch modulation raises pitch meaningfully");
}

// ===========================================================================
// Test: lane-rate position modulation affects timbre
// ===========================================================================
static void test_lane_position_mod(LayerTestHarness& h) {
    std::fprintf(stderr, "\n--- Lane position modulation ---\n");

    auto run_with_position_mod = [&](float position_mod) -> float {
        void* inst = h.loader.create_instance();
        auto params = h.default_params();
        if (h.idx_amplitude >= 0) params[h.idx_amplitude] = 0.5f;

        PolyTestContext tc;
        tc.set_output_channels(2);
        tc.ctx.param_values = params.data();
        tc.setup_wavetable_layer_voice(220.0f);
        tc.position_mod_lane_data[0] = position_mod;

        auto m = h.run_blocks(inst, tc, 4);
        h.loader.destroy_instance(inst);
        return m.spectral_centroid_hz;
    };

    float c0 = run_with_position_mod(0.0f);
    float c1 = run_with_position_mod(0.7f);
    std::fprintf(stderr, "    base=%.1fHz modded=%.1fHz\n", c0, c1);

    check(std::abs(c1 - c0) > 10.0f, "lane position modulation changes timbre");
}

// ===========================================================================
// Test: lane-rate warp modulation affects timbre
// ===========================================================================
static void test_lane_warp_mod(LayerTestHarness& h) {
    std::fprintf(stderr, "\n--- Lane warp modulation ---\n");

    auto run_with_warp_mod = [&](float warp_mod) -> float {
        void* inst = h.loader.create_instance();
        auto params = h.default_params();
        if (h.idx_amplitude >= 0) params[h.idx_amplitude] = 0.5f;
        if (h.idx_warp_mode >= 0) params[h.idx_warp_mode] = 1.0f; // Sync

        PolyTestContext tc;
        tc.set_output_channels(2);
        tc.ctx.param_values = params.data();
        tc.setup_wavetable_layer_voice(220.0f);
        tc.warp_mod_lane_data[0] = warp_mod;

        auto m = h.run_blocks(inst, tc, 4);
        h.loader.destroy_instance(inst);
        return m.spectral_centroid_hz;
    };

    float c0 = run_with_warp_mod(0.0f);
    float c1 = run_with_warp_mod(0.7f);
    std::fprintf(stderr, "    base=%.1fHz modded=%.1fHz\n", c0, c1);

    check(std::abs(c1 - c0) > 10.0f, "lane warp modulation changes timbre");
}

// ===========================================================================
// Test: smoothing params affect transition shape
// ===========================================================================
static void test_smoothing_params(LayerTestHarness& h) {
    std::fprintf(stderr, "\n--- Smoothing params ---\n");

    auto first_transition_block = [&](bool position_smoothing) -> float {
        auto run_transition = [&](float smooth_ms) {
            std::array<float, PolyTestContext::kFrames> out{};
            void* inst = h.loader.create_instance();
            auto params = h.default_params();
            if (h.idx_amplitude >= 0) params[h.idx_amplitude] = 0.5f;
            if (h.idx_warp_mode >= 0) params[h.idx_warp_mode] = 1.0f; // Sync for clear warp behavior
            if (position_smoothing) {
                if (h.idx_position_smooth_ms >= 0) params[h.idx_position_smooth_ms] = smooth_ms;
            } else if (h.idx_warp_smooth_ms >= 0) {
                params[h.idx_warp_smooth_ms] = smooth_ms;
            }

            PolyTestContext tc;
            tc.set_output_channels(2);
            tc.ctx.param_values = params.data();
            tc.setup_wavetable_layer_voice(220.0f);

            h.run_blocks(inst, tc, 2);

            tc.clear_output();
            if (position_smoothing) {
                tc.position_mod_lane_data[0] = 0.8f;
            } else {
                tc.warp_mod_lane_data[0] = 0.8f;
            }
            h.loader.process_audio(inst, &tc.ctx);
            std::memcpy(out.data(), tc.output_buf, sizeof(float) * PolyTestContext::kFrames);
            h.loader.destroy_instance(inst);
            return out;
        };

        auto smoothed = run_transition(30.0f);
        auto instant = run_transition(0.0f);
        return average_abs_diff(smoothed.data(), instant.data(), 64);
    };

    float pos_diff = first_transition_block(true);
    float warp_diff = first_transition_block(false);
    std::fprintf(stderr, "    pos_transition_diff=%.6f warp_transition_diff=%.6f\n", pos_diff, warp_diff);

    check(pos_diff > 1e-4f, "position_smooth_ms changes early transition shape");
    check(warp_diff > 1e-4f, "warp_smooth_ms changes early transition shape");
}

// ===========================================================================
// Test: Warp modes produce different output
// ===========================================================================
static void test_warp_modes(LayerTestHarness& h) {
    std::fprintf(stderr, "\n--- Warp modes ---\n");

    // Warp modes: 0=None, 1=Sync, 2=BendPlus, 3=BendMinus, 4=Mirror,
    //             5=Asym, 6=Quantize, 7=Flip
    const char* mode_names[] = {
        "None", "Sync", "BendPlus", "BendMinus", "Mirror", "Asym", "Quantize", "Flip"
    };

    // Capture output with warp_mode=0 (None) as reference
    auto run_with_warp = [&](int warp_mode_val, float warp_amount_val) -> float {
        void* inst = h.loader.create_instance();
        auto params = h.default_params();
        if (h.idx_amplitude >= 0) params[h.idx_amplitude] = 0.5f;
        if (h.idx_warp_mode >= 0) params[h.idx_warp_mode] = static_cast<float>(warp_mode_val);
        if (h.idx_warp_amount >= 0) params[h.idx_warp_amount] = warp_amount_val;
        if (h.idx_phase_reset_mode >= 0) params[h.idx_phase_reset_mode] = 1.0f; // Reset for consistency

        PolyTestContext tc;
        tc.set_output_channels(2);
        tc.ctx.param_values = params.data();
        tc.setup_wavetable_layer_voice(220.0f);

        auto m = h.run_blocks(inst, tc, 4);
        h.loader.destroy_instance(inst);
        return m.spectral_centroid_hz;
    };

    float centroid_none = run_with_warp(0, 0.7f);
    std::fprintf(stderr, "    None centroid=%.1fHz\n", centroid_none);

    for (int mode = 1; mode <= 7; ++mode) {
        float centroid = run_with_warp(mode, 0.7f);
        std::fprintf(stderr, "    %s centroid=%.1fHz\n", mode_names[mode], centroid);

        // Each warp mode should produce a different spectral character than None.
        // Use a relative threshold since some modes (Quantize, Flip) have subtler effects.
        float diff = std::abs(centroid - centroid_none);
        bool differs = diff > 2.0f || (diff / std::max(centroid_none, 1.0f)) > 0.005f;
        char msg[128];
        std::snprintf(msg, sizeof(msg), "%s warp alters spectral centroid vs None", mode_names[mode]);
        check(differs, msg);
    }
}

// ===========================================================================
// Test: Position affects timbre
// ===========================================================================
static void test_position_timbre(LayerTestHarness& h) {
    std::fprintf(stderr, "\n--- Position affects timbre ---\n");

    auto run_at_position = [&](float pos) -> float {
        void* inst = h.loader.create_instance();
        auto params = h.default_params();
        if (h.idx_amplitude >= 0) params[h.idx_amplitude] = 0.5f;
        if (h.idx_position >= 0) params[h.idx_position] = pos;

        PolyTestContext tc;
        tc.set_output_channels(2);
        tc.ctx.param_values = params.data();
        tc.setup_wavetable_layer_voice(440.0f);

        auto m = h.run_blocks(inst, tc, 4);
        h.loader.destroy_instance(inst);
        return m.spectral_centroid_hz;
    };

    float c0 = run_at_position(0.0f);
    float c50 = run_at_position(0.5f);
    float c100 = run_at_position(1.0f);

    std::fprintf(stderr, "    pos=0.0 centroid=%.1f  pos=0.5 centroid=%.1f  pos=1.0 centroid=%.1f\n",
                 c0, c50, c100);

    // Different positions in the wavetable should produce different timbres
    bool varies = (std::abs(c0 - c50) > 10.0f) || (std::abs(c50 - c100) > 10.0f);
    check(varies, "different wavetable positions produce different timbres");
}

// ===========================================================================
// Test: Amplitude scaling
// ===========================================================================
static void test_amplitude_scaling(LayerTestHarness& h) {
    std::fprintf(stderr, "\n--- Amplitude scaling ---\n");

    auto run_at_amplitude = [&](float amp) -> float {
        void* inst = h.loader.create_instance();
        auto params = h.default_params();
        if (h.idx_amplitude >= 0) params[h.idx_amplitude] = amp;

        PolyTestContext tc;
        tc.set_output_channels(2);
        tc.ctx.param_values = params.data();
        tc.setup_wavetable_layer_voice(440.0f);

        h.run_blocks(inst, tc, 4);
        float rms = rms_of(tc.output_buf, tc.kFrames);
        h.loader.destroy_instance(inst);
        return rms;
    };

    float rms_low = run_at_amplitude(0.1f);
    float rms_high = run_at_amplitude(0.8f);

    std::fprintf(stderr, "    amp=0.1 rms=%.4f  amp=0.8 rms=%.4f\n", rms_low, rms_high);

    check(rms_high > rms_low * 2.0f, "higher amplitude produces louder output");
    check(rms_low > 0.001f, "low amplitude still produces signal");
}

// ===========================================================================
// Main
// ===========================================================================
int main() {
    LayerTestHarness h;
    if (!h.load()) return 1;

    test_single_voice_fundamental(h);
    test_stereo_pan(h);
    test_gate_on_reset(h);
    test_voice_gain_audio(h);
    test_multi_voice(h);
    test_drift(h);
    test_lane_pitch_mod(h);
    test_lane_position_mod(h);
    test_lane_warp_mod(h);
    test_smoothing_params(h);
    test_warp_modes(h);
    test_position_timbre(h);
    test_amplitude_scaling(h);

    std::fprintf(stderr, "\n");
    if (failures == 0) {
        std::printf("WavetableLayer renderer correctness: ALL PASS\n");
    } else {
        std::fprintf(stderr, "WavetableLayer renderer correctness: %d FAILURES\n", failures);
    }
    return failures > 0 ? 1 : 0;
}
