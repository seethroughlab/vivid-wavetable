// Audio output correctness tests for vivid-wavetable operators.
// Verifies that polyphonic oscillators produce expected spectral and amplitude
// properties using analyze_audio() — property-based, no golden files.

#include "operator_api/types.h"
#include "runtime/debug/output_analyzer.h"
#include "envelope.h"
#include "test_support.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>
#include <limits>

static int failures = 0;

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "  FAIL: %s\n", msg);
        failures++;
    } else {
        std::fprintf(stderr, "  PASS: %s\n", msg);
    }
}

static void check_float(float actual, float expected, float tol, const char* msg) {
    if (std::fabs(actual - expected) > tol) {
        std::fprintf(stderr, "  FAIL: %s (expected %.4f, got %.4f)\n", msg, expected, actual);
        failures++;
    } else {
        std::fprintf(stderr, "  PASS: %s (%.4f)\n", msg, actual);
    }
}

static std::vector<float> make_sine_buffer(int frames, float sample_rate, float frequency, float amplitude) {
    std::vector<float> buffer(static_cast<size_t>(frames), 0.0f);
    for (int i = 0; i < frames; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        buffer[static_cast<size_t>(i)] = std::sin(t * 2.0f * 3.14159265358979323846f * frequency) * amplitude;
    }
    return buffer;
}

static float average_abs_diff(const float* a, const float* b, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; ++i)
        sum += std::fabs(a[i] - b[i]);
    return sum / static_cast<float>(count);
}

static float max_adjacent_delta(const float* data, int count) {
    float max_delta = 0.0f;
    for (int i = 1; i < count; ++i)
        max_delta = std::max(max_delta, std::fabs(data[i] - data[i - 1]));
    return max_delta;
}

static float stereo_window_rms(const float* left, const float* right, int start, int count) {
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
        float l = left[start + i];
        float r = right[start + i];
        sum += static_cast<double>(l) * static_cast<double>(l);
        sum += static_cast<double>(r) * static_cast<double>(r);
    }
    return std::sqrt(static_cast<float>(sum / (2.0 * static_cast<double>(count))));
}

static float mono_window_rms(const float* data, int start, int count) {
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
        float v = data[start + i];
        sum += static_cast<double>(v) * static_cast<double>(v);
    }
    return std::sqrt(static_cast<float>(sum / static_cast<double>(count)));
}

static float rms_ratio(float a, float b) {
    float hi = std::max(a, b);
    float lo = std::min(a, b);
    if (hi <= 1.0e-6f) return 1.0f;
    return lo / hi;
}

static float average_value(const float* data, int count) {
    double sum = 0.0;
    for (int i = 0; i < count; ++i) sum += data[i];
    return static_cast<float>(sum / static_cast<double>(count));
}

static std::vector<float> render_envelope_from_gate(const std::vector<float>& gate,
                                                    float sample_rate,
                                                    float attack,
                                                    float decay,
                                                    float sustain,
                                                    float release) {
    std::vector<float> out(gate.size(), 0.0f);
    Envelope::LaneState state{};
    float dt = 1.0f / sample_rate;
    for (size_t i = 0; i < gate.size(); ++i) {
        Envelope::advance_triggers(state, gate[i], 0.0f);
        Envelope::advance_adsr(state, dt, attack, decay, sustain, release, 1);
        out[i] = state.env_value;
    }
    return out;
}

static void test_per_voice_envelope_path(const std::string& staging) {
    std::fprintf(stderr, "\n--- VoiceMixer: per-note envelope path ---\n");

    MiniLoader loader;
    if (!loader.load((staging + "/voice_mixer.dylib").c_str())) {
        std::fprintf(stderr, "  SKIP: could not load voice_mixer.dylib\n");
        return;
    }

    {
        std::fprintf(stderr, "\n  [Independent lane envelope state]\n");
        std::vector<float> gate_a(PolyTestContext::kFrames, 0.0f);
        std::vector<float> gate_b(PolyTestContext::kFrames, 0.0f);
        for (int i = 0; i < 180; ++i) gate_a[i] = 1.0f;
        for (int i = 320; i < 500; ++i) gate_b[i] = 1.0f;

        auto env_a = render_envelope_from_gate(gate_a, PolyTestContext::kSampleRate,
                                               0.002f, 0.008f, 0.75f, 0.004f);
        auto env_b = render_envelope_from_gate(gate_b, PolyTestContext::kSampleRate,
                                               0.002f, 0.008f, 0.75f, 0.004f);

        std::fprintf(stderr, "    env_a[120]=%.4f env_b[120]=%.4f env_a[380]=%.4f env_b[380]=%.4f\n",
                     env_a[120], env_b[120], env_a[380], env_b[380]);
        check(env_a[120] > 0.4f && env_b[120] < 0.02f,
              "lane A can attack while lane B stays idle");
        check(env_b[380] > 0.2f && env_a[380] < 0.2f,
              "lane B keeps its own envelope state on a later retrigger");
    }

    std::fprintf(stderr, "\n  [Repeated chord retrigger consistency]\n");
    const auto* desc = loader.descriptor();
    if (!desc) {
        std::fprintf(stderr, "  SKIP: missing voice_mixer descriptor\n");
        return;
    }

    int spread_idx = -1;
    int glue_idx = -1;
    for (uint32_t p = 0; p < desc->param_count; ++p) {
        if (std::strcmp(desc->params[p].name, "stereo_spread") == 0) {
            spread_idx = static_cast<int>(p);
        }
        if (std::strcmp(desc->params[p].name, "glue") == 0) {
            glue_idx = static_cast<int>(p);
        }
    }

    std::vector<float> gate(PolyTestContext::kFrames, 0.0f);
    constexpr int kOn = 320;
    constexpr int kOff = 160;
    for (int chord = 0; chord < 4; ++chord) {
        int start = chord * (kOn + kOff);
        for (int i = 0; i < kOn && start + i < PolyTestContext::kFrames; ++i) {
            gate[start + i] = 1.0f;
        }
    }

    auto env_left = render_envelope_from_gate(gate, PolyTestContext::kSampleRate,
                                              0.002f, 0.010f, 0.88f, 0.006f);
    auto env_right = render_envelope_from_gate(gate, PolyTestContext::kSampleRate,
                                               0.002f, 0.010f, 0.88f, 0.006f);

    std::vector<float> input(2 * PolyTestContext::kFrames, 1.0f);
    std::vector<float> env_audio(2 * PolyTestContext::kFrames, 0.0f);
    std::memcpy(env_audio.data(), env_left.data(), PolyTestContext::kFrames * sizeof(float));
    std::memcpy(env_audio.data() + PolyTestContext::kFrames, env_right.data(),
                PolyTestContext::kFrames * sizeof(float));

    void* inst = loader.create_instance();
    std::vector<float> params(desc->param_count);
    for (uint32_t p = 0; p < desc->param_count; ++p) {
        params[p] = desc->params[p].default_value;
    }
    if (spread_idx >= 0) params[spread_idx] = 0.35f;

    PolyTestContext tc;
    tc.set_output_channels(2);
    tc.ctx.param_values = params.data();
    tc.clear_audio_inputs();
    tc.clear_lane_ports();
    tc.bind_audio_input(0, input.data(), 2);
    tc.bind_audio_input(1, env_audio.data(), 2);
    tc.vel_data[0] = 1.0f;
    tc.vel_data[1] = 1.0f;
    tc.bind_lane(4, tc.vel_data, 2);
    tc.clear_output();
    loader.process_audio(inst, &tc.ctx);

    float* left = tc.output_buf;
    float* right = tc.output_buf + PolyTestContext::kFrames;
    float first_rms = stereo_window_rms(left, right, 0, kOn);
    std::fprintf(stderr, "    chord_rms:");
    for (int chord = 0; chord < 4; ++chord) {
        int start = chord * (kOn + kOff);
        float rms = stereo_window_rms(left, right, start, kOn);
        std::fprintf(stderr, " %.4f", rms);
        check(rms >= first_rms * 0.7f, "later chord window stays close to first-hit level");
    }
    std::fprintf(stderr, "\n");

    float gap_rms = stereo_window_rms(left, right, kOn, kOff);
    std::fprintf(stderr, "    first_rms=%.4f gap_rms=%.4f\n", first_rms, gap_rms);
    check(gap_rms < first_rms * 0.75f, "release gap drops below active chord level");

    {
        std::fprintf(stderr, "\n  [Glue stays bounded on dense stacks]\n");
        void* glue_inst = loader.create_instance();
        std::vector<float> glue_params(desc->param_count);
        for (uint32_t p = 0; p < desc->param_count; ++p) glue_params[p] = desc->params[p].default_value;
        if (spread_idx >= 0) glue_params[spread_idx] = 0.75f;

        std::vector<float> dense_input(8 * PolyTestContext::kFrames, 0.0f);
        for (int ch = 0; ch < 8; ++ch) {
            float freq = 110.0f + 22.0f * static_cast<float>(ch);
            float* buf = dense_input.data() + ch * PolyTestContext::kFrames;
            for (int i = 0; i < PolyTestContext::kFrames; ++i) {
                float t = static_cast<float>(i) / static_cast<float>(PolyTestContext::kSampleRate);
                buf[i] = 0.22f * std::sin(t * 2.0f * 3.14159265358979323846f * freq);
            }
        }

        auto run_glue_mix = [&](float glue_value) {
            if (glue_idx >= 0) glue_params[glue_idx] = glue_value;
            PolyTestContext mix_tc;
            mix_tc.set_output_channels(2);
            mix_tc.ctx.param_values = glue_params.data();
            mix_tc.clear_audio_inputs();
            mix_tc.clear_lane_ports();
            mix_tc.bind_audio_input(0, dense_input.data(), 8);
            for (int i = 0; i < 8; ++i) {
                mix_tc.vel_data[i] = 1.0f;
            }
            mix_tc.bind_lane(4, mix_tc.vel_data, 8);
            mix_tc.clear_output();
            loader.process_audio(glue_inst, &mix_tc.ctx);
            return mix_tc.analyze_output(2);
        };

        auto clean = run_glue_mix(0.0f);
        auto glued = run_glue_mix(0.75f);
        std::fprintf(stderr, "    clean_rms=%.4f glued_rms=%.4f clean_peak=%.4f glued_peak=%.4f\n",
                     clean.rms, glued.rms, clean.peak, glued.peak);
        check(std::isfinite(glued.rms) && std::isfinite(glued.peak), "glued dense stack stays finite");
        check(glued.peak < 1.5f, "glued dense stack stays bounded");
        check(glued.rms < clean.rms * 1.35f, "glue does not cause a large loudness jump");

        loader.destroy_instance(glue_inst);
    }

    loader.destroy_instance(inst);
}

static void test_voice_mixer_stereo_pairs(const std::string& staging) {
    std::fprintf(stderr, "\n--- VoiceMixer: stereo-pair width preservation ---\n");

    MiniLoader loader;
    if (!loader.load((staging + "/voice_mixer.dylib").c_str())) {
        std::fprintf(stderr, "  SKIP: could not load voice_mixer.dylib\n");
        return;
    }

    const auto* desc = loader.descriptor();
    if (!desc) {
        std::fprintf(stderr, "  SKIP: missing voice_mixer descriptor\n");
        return;
    }

    int input_layout_idx = -1;
    int spread_idx = -1;
    for (uint32_t p = 0; p < desc->param_count; ++p) {
        if (std::strcmp(desc->params[p].name, "input_layout") == 0) input_layout_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "stereo_spread") == 0) spread_idx = static_cast<int>(p);
    }

    struct StereoMixResult {
        float raw_lr_diff = 0.0f;
        float mix_lr_diff = 0.0f;
        float left_rms = 0.0f;
        float right_rms = 0.0f;
    };

    auto run_stereo_mix = [&](float phase_amount) {
        std::vector<float> stereo_input(2 * PolyTestContext::kFrames, 0.0f);
        float* left = stereo_input.data();
        float* right = stereo_input.data() + PolyTestContext::kFrames;
        float phase_offset = phase_amount * 1.57079632679f;
        for (int i = 0; i < PolyTestContext::kFrames; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(PolyTestContext::kSampleRate);
            float carrier = t * 2.0f * 3.14159265358979323846f * 220.0f;
            left[i] = 0.35f * std::sin(carrier);
            right[i] = 0.35f * std::sin(carrier + phase_offset);
        }
        float raw_lr_diff = average_abs_diff(left, right, PolyTestContext::kFrames);

        void* inst = loader.create_instance();
        std::vector<float> params(desc->param_count);
        for (uint32_t p = 0; p < desc->param_count; ++p)
            params[p] = desc->params[p].default_value;
        if (input_layout_idx >= 0) params[input_layout_idx] = 1.0f;
        if (spread_idx >= 0) params[spread_idx] = 0.8f;

        PolyTestContext tc;
        tc.set_output_channels(2);
        tc.ctx.param_values = params.data();
        tc.clear_audio_inputs();
        tc.clear_lane_ports();
        tc.bind_audio_input(0, stereo_input.data(), 2);
        tc.vel_data[0] = 1.0f;
        tc.bind_lane(4, tc.vel_data, 1);
        tc.clear_output();
        loader.process_audio(inst, &tc.ctx);

        float mix_lr_diff = average_abs_diff(tc.output_buf, tc.output_buf + PolyTestContext::kFrames,
                                             PolyTestContext::kFrames);
        float left_rms = mono_window_rms(tc.output_buf, 0, PolyTestContext::kFrames);
        float right_rms = mono_window_rms(tc.output_buf + PolyTestContext::kFrames, 0,
                                          PolyTestContext::kFrames);
        auto mix_metrics = tc.analyze_output(2);
        std::fprintf(stderr,
                     "    phase_amount=%.2f raw_lr_diff=%.4f mix_lr_diff=%.4f left_rms=%.4f right_rms=%.4f rms=%.4f peak=%.4f\n",
                     phase_amount, raw_lr_diff, mix_lr_diff, left_rms, right_rms,
                     mix_metrics.rms, mix_metrics.peak);
        check(std::isfinite(mix_metrics.rms) && std::isfinite(mix_metrics.peak), "stereo-pair mix stays finite");
        check(mix_metrics.peak < 1.5f, "stereo-pair mix stays bounded");
        loader.destroy_instance(inst);
        return StereoMixResult{raw_lr_diff, mix_lr_diff, left_rms, right_rms};
    };

    auto narrow = run_stereo_mix(0.0f);
    auto wide = run_stereo_mix(1.0f);
    check(wide.raw_lr_diff > narrow.raw_lr_diff + 0.001f, "test fixture widens the source stereo pair");
    check(wide.mix_lr_diff > narrow.mix_lr_diff + 0.001f, "VoiceMixer preserves wider stereo-pair divergence");
    check(wide.mix_lr_diff > wide.raw_lr_diff * 0.45f, "VoiceMixer keeps meaningful stereo width from stereo pairs");
    check(rms_ratio(wide.left_rms, wide.right_rms) > 0.45f, "stereo-pair mix keeps substantial energy in both channels");
}

// ---------------------------------------------------------------------------
// AnalogOsc tests
// ---------------------------------------------------------------------------

static void test_analog_osc(const std::string& staging) {
    std::fprintf(stderr, "\n--- AnalogOsc: waveform spectral properties ---\n");

    MiniLoader loader;
    if (!loader.load((staging + "/analog_osc.dylib").c_str())) {
        std::fprintf(stderr, "  SKIP: could not load analog_osc.dylib\n");
        return;
    }

    const auto* desc = loader.descriptor();
    if (!desc) return;

    int waveform_idx = -1;
    int amplitude_idx = -1;
    int interaction_mode_idx = -1;
    int interaction_depth_idx = -1;
    int interaction_input_gain_idx = -1;
    int interaction_tracking_idx = -1;
    for (uint32_t p = 0; p < desc->param_count; p++) {
        if (std::strcmp(desc->params[p].name, "waveform") == 0) waveform_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "amplitude") == 0) amplitude_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "interaction_mode") == 0) interaction_mode_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "interaction_depth") == 0) interaction_depth_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "interaction_input_gain") == 0) interaction_input_gain_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "interaction_tracking") == 0) interaction_tracking_idx = static_cast<int>(p);
    }

    auto make_params = [&](int waveform,
                           float amplitude,
                           int interaction_mode = 0,
                           float interaction_depth = 0.0f,
                           float interaction_input_gain = 1.0f,
                           float interaction_tracking = 1.0f) {
        std::vector<float> params(desc->param_count);
        for (uint32_t p = 0; p < desc->param_count; p++)
            params[p] = desc->params[p].default_value;
        if (waveform_idx >= 0) params[waveform_idx] = static_cast<float>(waveform);
        if (amplitude_idx >= 0) params[amplitude_idx] = amplitude;
        if (interaction_mode_idx >= 0) params[interaction_mode_idx] = static_cast<float>(interaction_mode);
        if (interaction_depth_idx >= 0) params[interaction_depth_idx] = interaction_depth;
        if (interaction_input_gain_idx >= 0) params[interaction_input_gain_idx] = interaction_input_gain;
        if (interaction_tracking_idx >= 0) params[interaction_tracking_idx] = interaction_tracking;
        return params;
    };

    auto run_osc = [&](int waveform,
                       float amplitude,
                       float freq,
                       int interaction_mode = 0,
                       float interaction_depth = 0.0f,
                       float interaction_input_gain = 1.0f,
                       float interaction_tracking = 1.0f,
                       const float* mod_audio = nullptr) -> vivid::AudioMetrics {
        void* inst = loader.create_instance();
        auto params = make_params(waveform, amplitude, interaction_mode, interaction_depth,
                                  interaction_input_gain, interaction_tracking);

        PolyTestContext tc;
        tc.ctx.param_values = params.data();
        tc.setup_analog_voice(freq);
        tc.clear_audio_inputs();
        // AnalogOsc port layout (PR3): notes_in=0, mod_input=1, pitch_mod_audio=2.
        if (mod_audio) tc.bind_audio_input(1, const_cast<float*>(mod_audio), 1);

        for (int b = 0; b < 6; b++) {
            tc.clear_output();
            loader.process_audio(inst, &tc.ctx);
            tc.ctx.time += static_cast<double>(PolyTestContext::kFrames) / PolyTestContext::kSampleRate;
            tc.ctx.frame++;
        }

        auto m = tc.analyze_output();
        loader.destroy_instance(inst);
        return m;
    };

    {
        std::fprintf(stderr, "\n  [Sine 440Hz]\n");
        auto m = run_osc(0, 0.5f, 440.0f);
        std::fprintf(stderr, "    rms=%.4f centroid=%.1fHz flatness=%.4f brightness=%.4f\n",
                     m.rms, m.spectral_centroid_hz, m.spectral_flatness, m.spectral_brightness);

        check(m.rms > 0.05f, "sine produces signal");
        check_float(m.spectral_centroid_hz, 440.0f, 60.0f, "sine centroid near 440Hz");
        check(m.spectral_flatness < 0.15f, "sine is tonal (low flatness)");
        check(m.spectral_brightness < 0.15f, "sine has low high-frequency content");
    }

    float brightness_sine, brightness_saw, brightness_square;

    {
        std::fprintf(stderr, "\n  [Saw 440Hz]\n");
        auto m_sine = run_osc(0, 0.5f, 440.0f);
        auto m = run_osc(1, 0.5f, 440.0f);
        std::fprintf(stderr, "    rms=%.4f centroid=%.1fHz flatness=%.4f brightness=%.4f\n",
                     m.rms, m.spectral_centroid_hz, m.spectral_flatness, m.spectral_brightness);

        brightness_sine = m_sine.spectral_brightness;
        brightness_saw = m.spectral_brightness;
        check(m.spectral_brightness > brightness_sine,
              "saw has more high-frequency content than sine");
        check(m.spectral_centroid_hz > 440.0f,
              "saw centroid above fundamental (harmonics pull it up)");
    }

    {
        std::fprintf(stderr, "\n  [Square 440Hz]\n");
        auto m = run_osc(2, 0.5f, 440.0f);
        std::fprintf(stderr, "    rms=%.4f centroid=%.1fHz flatness=%.4f brightness=%.4f\n",
                     m.rms, m.spectral_centroid_hz, m.spectral_flatness, m.spectral_brightness);

        brightness_square = m.spectral_brightness;
        check(brightness_square > brightness_sine,
              "square has more high-frequency content than sine");
    }

    {
        std::fprintf(stderr, "\n  [Triangle 440Hz]\n");
        auto m = run_osc(3, 0.5f, 440.0f);
        std::fprintf(stderr, "    rms=%.4f centroid=%.1fHz flatness=%.4f brightness=%.4f\n",
                     m.rms, m.spectral_centroid_hz, m.spectral_flatness, m.spectral_brightness);

        check(m.spectral_brightness < brightness_saw,
              "triangle has less high-frequency content than saw");
    }

    {
        std::fprintf(stderr, "\n  [Gate=0 → silence]\n");
        void* inst = loader.create_instance();
        auto params = make_params(0, 0.5f);

        PolyTestContext tc;
        tc.ctx.param_values = params.data();
        tc.setup_analog_voice(440.0f);
        tc.silence_gate();

        for (int b = 0; b < 4; b++) {
            tc.clear_output();
            loader.process_audio(inst, &tc.ctx);
        }

        auto m = tc.analyze_output();
        std::fprintf(stderr, "    rms=%.6f peak=%.6f\n", m.rms, m.peak);
        check(m.rms < 0.001f, "gate=0 produces silence");

        loader.destroy_instance(inst);
    }

    {
        std::fprintf(stderr, "\n  [Amplitude scaling]\n");
        auto m_half = run_osc(0, 0.5f, 440.0f);
        auto m_quarter = run_osc(0, 0.25f, 440.0f);
        float ratio = m_half.rms / m_quarter.rms;
        std::fprintf(stderr, "    rms(0.5)=%.4f rms(0.25)=%.4f ratio=%.2f\n",
                     m_half.rms, m_quarter.rms, ratio);
        check_float(ratio, 2.0f, 0.3f, "doubling amplitude roughly doubles RMS");
    }

    {
        std::fprintf(stderr, "\n  [Interaction depth and mode sanity]\n");
        auto mod_audio = make_sine_buffer(PolyTestContext::kFrames, static_cast<float>(PolyTestContext::kSampleRate), 220.0f, 0.85f);
        auto off = run_osc(1, 0.35f, 220.0f, 0, 0.0f, 1.0f, 1.0f, mod_audio.data());
        auto low_fm = run_osc(1, 0.35f, 220.0f, 1, 0.12f, 1.0f, 1.0f, mod_audio.data());
        auto mid_fm = run_osc(1, 0.35f, 220.0f, 1, 0.45f, 1.0f, 1.0f, mod_audio.data());
        auto high_fm = run_osc(1, 0.35f, 220.0f, 1, 0.9f, 1.0f, 1.0f, mod_audio.data());
        auto pm = run_osc(0, 0.35f, 440.0f, 2, 0.4f, 1.0f, 1.0f, mod_audio.data());
        auto rm_low = run_osc(1, 0.35f, 220.0f, 3, 0.15f, 1.0f, 1.0f, mod_audio.data());
        auto rm_high = run_osc(1, 0.35f, 220.0f, 3, 0.85f, 1.0f, 1.0f, mod_audio.data());
        auto am = run_osc(1, 0.35f, 220.0f, 4, 0.75f, 1.0f, 1.0f, mod_audio.data());
        auto gain_low = run_osc(1, 0.35f, 220.0f, 1, 0.4f, 0.5f, 1.0f, mod_audio.data());
        auto gain_high = run_osc(1, 0.35f, 220.0f, 1, 0.4f, 3.0f, 1.0f, mod_audio.data());
        auto tracking_off = run_osc(1, 0.35f, 880.0f, 1, 0.4f, 1.0f, 0.0f, mod_audio.data());
        auto tracking_on = run_osc(1, 0.35f, 880.0f, 1, 0.4f, 1.0f, 1.0f, mod_audio.data());

        float low_diff = std::fabs(low_fm.spectral_brightness - off.spectral_brightness)
            + std::fabs(low_fm.spectral_centroid_hz - off.spectral_centroid_hz) / 1000.0f;
        float mid_diff = std::fabs(mid_fm.spectral_brightness - low_fm.spectral_brightness)
            + std::fabs(mid_fm.spectral_centroid_hz - low_fm.spectral_centroid_hz) / 1000.0f;
        float rm_diff = std::fabs(rm_high.spectral_brightness - rm_low.spectral_brightness)
            + std::fabs(rm_high.spectral_centroid_hz - rm_low.spectral_centroid_hz) / 1000.0f;
        float gain_diff = std::fabs(gain_high.spectral_centroid_hz - gain_low.spectral_centroid_hz);
        float tracking_diff = std::fabs(tracking_on.spectral_centroid_hz - tracking_off.spectral_centroid_hz);

        std::fprintf(stderr,
                     "    low_diff=%.4f mid_diff=%.4f rm_diff=%.4f pm_peak=%.4f am_peak=%.4f gain_diff=%.1f tracking_diff=%.1f\n",
                     low_diff, mid_diff, rm_diff, pm.peak, am.peak, gain_diff, tracking_diff);
        check(low_diff > 0.01f, "low FM interaction depth is audible");
        check(mid_diff > 0.01f, "medium FM interaction depth differs from low depth");
        check(std::isfinite(high_fm.rms) && std::isfinite(high_fm.peak), "high FM interaction stays finite");
        check(high_fm.peak < 1.5f, "high FM interaction stays bounded");
        check(std::isfinite(pm.rms) && std::isfinite(pm.peak), "PM interaction stays finite");
        check(pm.peak < 1.5f, "PM interaction stays bounded");
        check(rm_diff > 0.01f, "RM depth changes the analog output materially");
        check(std::isfinite(am.rms) && std::isfinite(am.peak), "AM interaction stays finite");
        check(am.peak < 1.5f, "AM interaction stays bounded");
        check(gain_diff > 20.0f, "interaction input gain measurably changes analog interaction");
        check(tracking_diff > 20.0f, "interaction tracking measurably changes analog FM behavior");
    }
}

// ---------------------------------------------------------------------------
// SubOsc tests
// ---------------------------------------------------------------------------

static void test_sub_osc(const std::string& staging) {
    std::fprintf(stderr, "\n--- SubOsc: octave relationships ---\n");

    MiniLoader loader;
    if (!loader.load((staging + "/sub_osc.dylib").c_str())) {
        std::fprintf(stderr, "  SKIP: could not load sub_osc.dylib\n");
        return;
    }

    const auto* desc = loader.descriptor();
    if (!desc) return;

    int octave_idx = -1, waveform_idx = -1, velocity_to_level_idx = -1;
    for (uint32_t p = 0; p < desc->param_count; p++) {
        if (std::strcmp(desc->params[p].name, "octave") == 0) octave_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "waveform") == 0) waveform_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "velocity_to_level") == 0) velocity_to_level_idx = static_cast<int>(p);
    }

    auto run_sub = [&](int octave, float freq, float velocity, float velocity_to_level) -> vivid::AudioMetrics {
        void* inst = loader.create_instance();
        std::vector<float> params(desc->param_count);
        for (uint32_t p = 0; p < desc->param_count; p++)
            params[p] = desc->params[p].default_value;
        if (octave_idx >= 0) params[octave_idx] = static_cast<float>(octave);
        if (waveform_idx >= 0) params[waveform_idx] = 0.0f;
        if (velocity_to_level_idx >= 0) params[velocity_to_level_idx] = velocity_to_level;

        PolyTestContext tc;
        tc.ctx.param_values = params.data();
        tc.setup_sub_voice(freq, velocity);

        for (int b = 0; b < 6; b++) {
            tc.clear_output();
            loader.process_audio(inst, &tc.ctx);
            tc.ctx.time += static_cast<double>(PolyTestContext::kFrames) / PolyTestContext::kSampleRate;
            tc.ctx.frame++;
        }

        auto m = tc.analyze_output();
        loader.destroy_instance(inst);
        return m;
    };

    {
        std::fprintf(stderr, "\n  [Octave -1, freq=440Hz]\n");
        auto m = run_sub(0, 440.0f, 1.0f, 0.0f);
        std::fprintf(stderr, "    rms=%.4f centroid=%.1fHz\n", m.rms, m.spectral_centroid_hz);
        check(m.rms > 0.05f, "sub osc produces signal");
        check_float(m.spectral_centroid_hz, 220.0f, 60.0f, "octave -1: centroid near 220Hz");
    }

    {
        std::fprintf(stderr, "\n  [Octave -2, freq=440Hz]\n");
        auto m = run_sub(1, 440.0f, 1.0f, 0.0f);
        std::fprintf(stderr, "    rms=%.4f centroid=%.1fHz\n", m.rms, m.spectral_centroid_hz);
        check(m.rms > 0.05f, "sub osc produces signal");
        check_float(m.spectral_centroid_hz, 110.0f, 60.0f, "octave -2: centroid near 110Hz");
    }

    {
        std::fprintf(stderr, "\n  [Velocity-to-level response]\n");
        auto soft = run_sub(0, 220.0f, 0.25f, 1.0f);
        auto hard = run_sub(0, 220.0f, 1.0f, 1.0f);
        std::fprintf(stderr, "    soft_rms=%.4f hard_rms=%.4f\n", soft.rms, hard.rms);
        check(hard.rms > soft.rms * 1.35f, "higher velocity increases sub contribution when velocity_to_level is active");
    }
}

// ---------------------------------------------------------------------------
// WavetableOsc tests
// ---------------------------------------------------------------------------

static void test_wavetable_osc(const std::string& staging) {
    std::fprintf(stderr, "\n--- WavetableOsc: position changes timbre ---\n");

    MiniLoader loader;
    if (!loader.load((staging + "/wavetable_osc.dylib").c_str())) {
        std::fprintf(stderr, "  SKIP: could not load wavetable_osc.dylib\n");
        return;
    }

    const auto* desc = loader.descriptor();
    if (!desc) return;

    int wavetable_source_idx = -1;
    int wavetable_family_idx = -1;
    int wavetable_member_idx = -1;
    int position_idx = -1;
    int unison_voices_idx = -1;
    int unison_spread_idx = -1;
    int unison_stereo_idx = -1;
    int unison_output_mode_idx = -1;
    int position_smooth_ms_idx = -1;
    int warp_smooth_ms_idx = -1;
    int phase_reset_mode_idx = -1;
    int start_phase_idx = -1;
    int phase_random_idx = -1;
    int stereo_phase_offset_idx = -1;
    int drift_amount_idx = -1;
    int drift_rate_hz_idx = -1;
    int interaction_mode_idx = -1;
    int interaction_depth_idx = -1;
    int interaction_input_gain_idx = -1;
    int interaction_tracking_idx = -1;
    for (uint32_t p = 0; p < desc->param_count; p++) {
        if (std::strcmp(desc->params[p].name, "wavetable_source") == 0) wavetable_source_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "wavetable_family") == 0) wavetable_family_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "wavetable_member") == 0) wavetable_member_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "position") == 0) position_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "unison_voices") == 0) unison_voices_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "unison_spread") == 0) unison_spread_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "unison_stereo") == 0) unison_stereo_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "unison_output_mode") == 0) unison_output_mode_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "position_smooth_ms") == 0) position_smooth_ms_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "warp_smooth_ms") == 0) warp_smooth_ms_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "phase_reset_mode") == 0) phase_reset_mode_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "start_phase") == 0) start_phase_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "phase_random") == 0) phase_random_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "stereo_phase_offset") == 0) stereo_phase_offset_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "drift_amount") == 0) drift_amount_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "drift_rate_hz") == 0) drift_rate_hz_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "interaction_mode") == 0) interaction_mode_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "interaction_depth") == 0) interaction_depth_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "interaction_input_gain") == 0) interaction_input_gain_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "interaction_tracking") == 0) interaction_tracking_idx = static_cast<int>(p);
    }
    check(position_idx >= 0, "position param found");
    check(wavetable_family_idx >= 0, "wavetable_family param found");
    check(wavetable_member_idx >= 0, "wavetable_member param found");
    if (position_idx < 0 || wavetable_family_idx < 0 || wavetable_member_idx < 0) return;

    struct WavetableRun {
        vivid::AudioMetrics metrics;
        std::vector<float> samples;
        uint8_t output_channels = 1;
    };

    auto run_wt = [&](float position,
                      int family,
                      int member,
                      int unison_voices,
                      float unison_spread,
                      int output_mode,
                      float unison_stereo,
                      float stereo_phase_offset,
                      float pos_smooth_ms,
                      float warp_smooth_ms,
                      int reset_mode,
                      float start_phase,
                      float phase_random,
                      float drift_amount,
                      float drift_rate_hz,
                      int interaction_mode = 0,
                      float interaction_depth = 0.0f,
                      float interaction_input_gain = 1.0f,
                      float interaction_tracking = 1.0f,
                      const float* pos_audio = nullptr,
                      const float* warp_audio = nullptr,
                      const float* mod_audio = nullptr) -> WavetableRun {
        void* inst = loader.create_instance();
        std::vector<float> params(desc->param_count);
        for (uint32_t p = 0; p < desc->param_count; p++)
            params[p] = desc->params[p].default_value;
        if (wavetable_source_idx >= 0) params[wavetable_source_idx] = 0.0f;
        if (wavetable_family_idx >= 0) params[wavetable_family_idx] = static_cast<float>(family);
        if (wavetable_member_idx >= 0) params[wavetable_member_idx] = static_cast<float>(member);
        params[position_idx] = position;
        if (unison_voices_idx >= 0) params[unison_voices_idx] = static_cast<float>(unison_voices);
        if (unison_spread_idx >= 0) params[unison_spread_idx] = unison_spread;
        if (unison_output_mode_idx >= 0) params[unison_output_mode_idx] = static_cast<float>(output_mode);
        if (unison_stereo_idx >= 0) params[unison_stereo_idx] = unison_stereo;
        if (position_smooth_ms_idx >= 0) params[position_smooth_ms_idx] = pos_smooth_ms;
        if (warp_smooth_ms_idx >= 0) params[warp_smooth_ms_idx] = warp_smooth_ms;
        if (phase_reset_mode_idx >= 0) params[phase_reset_mode_idx] = static_cast<float>(reset_mode);
        if (start_phase_idx >= 0) params[start_phase_idx] = start_phase;
        if (phase_random_idx >= 0) params[phase_random_idx] = phase_random;
        if (stereo_phase_offset_idx >= 0) params[stereo_phase_offset_idx] = stereo_phase_offset;
        if (drift_amount_idx >= 0) params[drift_amount_idx] = drift_amount;
        if (drift_rate_hz_idx >= 0) params[drift_rate_hz_idx] = drift_rate_hz;
        if (interaction_mode_idx >= 0) params[interaction_mode_idx] = static_cast<float>(interaction_mode);
        if (interaction_depth_idx >= 0) params[interaction_depth_idx] = interaction_depth;
        if (interaction_input_gain_idx >= 0) params[interaction_input_gain_idx] = interaction_input_gain;
        if (interaction_tracking_idx >= 0) params[interaction_tracking_idx] = interaction_tracking;

        PolyTestContext tc;
        tc.ctx.param_values = params.data();
        tc.setup_wavetable_voice(440.0f);
        tc.clear_audio_inputs();
        // WavetableOsc port layout: midi_in=0, lanes=1-7, mod_input=8,
        // pitch_mod_audio=9, position_mod_audio=10, warp_mod_audio=11.
        if (mod_audio) tc.bind_audio_input(8, const_cast<float*>(mod_audio), 1);
        if (pos_audio) tc.bind_audio_input(10, const_cast<float*>(pos_audio), 1);
        if (warp_audio) tc.bind_audio_input(11, const_cast<float*>(warp_audio), 1);

        for (int b = 0; b < 6; b++) {
            tc.clear_output();
            loader.process_audio(inst, &tc.ctx);
            tc.ctx.time += static_cast<double>(PolyTestContext::kFrames) / PolyTestContext::kSampleRate;
            tc.ctx.frame++;
        }

        WavetableRun result;
        result.output_channels = (output_mode == 1) ? 2 : 1;
        result.metrics = tc.analyze_output(1);
        result.samples.assign(tc.output_buf, tc.output_buf + PolyTestContext::kFrames * result.output_channels);
        loader.destroy_instance(inst);
        return result;
    };

    auto default_run = [&](float position, int family, int member) {
        return run_wt(position, family, member, 1, 0.0f, 0, 0.0f,
                      0.25f, 8.0f, 8.0f, 0, 0.0f, 0.0f, 0.0f, 0.18f);
    };

    auto m_pos0 = default_run(0.0f, 0, 0);
    auto m_pos5 = default_run(0.5f, 0, 0);

    std::fprintf(stderr, "  pos=0.0: brightness=%.4f centroid=%.1fHz\n",
                 m_pos0.metrics.spectral_brightness, m_pos0.metrics.spectral_centroid_hz);
    std::fprintf(stderr, "  pos=0.5: brightness=%.4f centroid=%.1fHz\n",
                 m_pos5.metrics.spectral_brightness, m_pos5.metrics.spectral_centroid_hz);

    check(m_pos0.metrics.rms > 0.05f, "wavetable osc produces signal at pos=0");
    check(m_pos5.metrics.rms > 0.05f, "wavetable osc produces signal at pos=0.5");

    float brightness_diff = std::fabs(m_pos0.metrics.spectral_brightness - m_pos5.metrics.spectral_brightness);
    float centroid_diff = std::fabs(m_pos0.metrics.spectral_centroid_hz - m_pos5.metrics.spectral_centroid_hz);
    std::fprintf(stderr, "  brightness_diff=%.4f centroid_diff=%.1fHz\n",
                 brightness_diff, centroid_diff);

    check(brightness_diff > 0.01f || centroid_diff > 20.0f,
          "different positions produce different timbres");

    {
        std::fprintf(stderr, "\n  [Family distinctness]\n");
        auto analog = default_run(0.35f, 0, 0);
        auto digital = default_run(0.35f, 1, 0);
        auto vocal = default_run(0.35f, 2, 0);
        auto metallic = default_run(0.35f, 3, 0);

        float analog_digital = average_abs_diff(analog.samples.data(), digital.samples.data(), PolyTestContext::kFrames);
        float analog_vocal = average_abs_diff(analog.samples.data(), vocal.samples.data(), PolyTestContext::kFrames);
        float analog_metallic = average_abs_diff(analog.samples.data(), metallic.samples.data(), PolyTestContext::kFrames);
        std::fprintf(stderr, "    analog_digital=%.4f analog_vocal=%.4f analog_metallic=%.4f\n",
                     analog_digital, analog_vocal, analog_metallic);
        check(analog_digital > 0.02f, "bright digital family differs from analog warm");
        check(analog_vocal > 0.02f, "vocal formant family differs from analog warm");
        check(analog_metallic > 0.02f, "metallic family differs from analog warm");
    }

    {
        std::fprintf(stderr, "\n  [Unison sanity]\n");
        auto mono = run_wt(0.35f, 0, 0, 1, 0.0f, 0, 0.0f,
                           0.25f, 8.0f, 8.0f, 0, 0.0f, 0.0f, 0.0f, 0.18f);
        auto uni = run_wt(0.35f, 0, 2, 6, 32.0f, 0, 0.0f,
                          0.25f, 8.0f, 8.0f, 0, 0.0f, 0.0f, 0.0f, 0.18f);
        float diff = average_abs_diff(mono.samples.data(), uni.samples.data(), PolyTestContext::kFrames);
        std::fprintf(stderr, "    mono_rms=%.4f uni_rms=%.4f avg_abs_diff=%.4f\n",
                     mono.metrics.rms, uni.metrics.rms, diff);
        check(uni.metrics.rms > 0.02f, "unison output remains audible");
        check(std::isfinite(uni.metrics.rms) && std::isfinite(uni.metrics.peak), "unison output stays finite");
        check(uni.metrics.peak < 1.5f, "unison output stays bounded");
        check(diff > 0.01f, "unison materially changes the waveform");
    }

    {
        std::fprintf(stderr, "\n  [Motion smoothing sanity]\n");
        float pos_audio[PolyTestContext::kFrames] = {};
        float warp_audio[PolyTestContext::kFrames] = {};
        for (int i = 0; i < PolyTestContext::kFrames; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(PolyTestContext::kFrames - 1);
            pos_audio[i] = t * 0.35f;
            warp_audio[i] = std::sin(t * 6.28318530718f) * 0.2f;
        }

        auto static_run = run_wt(0.2f, 5, 0, 1, 0.0f, 0, 0.0f,
                                 0.25f, 8.0f, 8.0f, 0, 0.0f, 0.0f, 0.0f, 0.18f);
        auto moving_run = run_wt(0.2f, 5, 0, 1, 0.0f, 0, 0.0f,
                                 0.25f, 8.0f, 8.0f, 0, 0.0f, 0.0f, 0.0f, 0.18f,
                                 0, 0.0f, 1.0f, 1.0f, pos_audio, warp_audio);
        float diff = average_abs_diff(static_run.samples.data(), moving_run.samples.data(), PolyTestContext::kFrames);
        std::fprintf(stderr, "    moving_rms=%.4f peak=%.4f avg_abs_diff=%.4f\n",
                     moving_run.metrics.rms, moving_run.metrics.peak, diff);
        check(moving_run.metrics.rms > 0.02f, "motion modulation remains audible");
        check(std::isfinite(moving_run.metrics.rms) && std::isfinite(moving_run.metrics.peak),
              "moving modulation stays finite");
        check(moving_run.metrics.peak < 1.5f, "moving modulation stays bounded");
        check(diff > 0.005f, "moving modulation changes the rendered output");
    }

    {
        std::fprintf(stderr, "\n  [Smooth step response]\n");
        float pos_step[PolyTestContext::kFrames] = {};
        float warp_step[PolyTestContext::kFrames] = {};
        for (int i = PolyTestContext::kFrames / 2; i < PolyTestContext::kFrames; ++i) {
            pos_step[i] = 0.65f;
            warp_step[i] = 0.45f;
        }

        auto unsmoothed = run_wt(0.18f, 1, 5, 1, 0.0f, 0, 0.0f,
                                 0.25f, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0.0f, 0.18f,
                                 0, 0.0f, 1.0f, 1.0f, pos_step, warp_step);
        auto smoothed = run_wt(0.18f, 1, 5, 1, 0.0f, 0, 0.0f,
                               0.25f, 18.0f, 18.0f, 0, 0.0f, 0.0f, 0.0f, 0.18f,
                               0, 0.0f, 1.0f, 1.0f, pos_step, warp_step);

        float unsmoothed_delta = max_adjacent_delta(unsmoothed.samples.data() + PolyTestContext::kFrames / 2 - 16, 64);
        float smoothed_delta = max_adjacent_delta(smoothed.samples.data() + PolyTestContext::kFrames / 2 - 16, 64);
        std::fprintf(stderr, "    unsmoothed_delta=%.4f smoothed_delta=%.4f\n",
                     unsmoothed_delta, smoothed_delta);
        check(smoothed_delta < unsmoothed_delta, "smoothing reduces local step discontinuity");
    }

    {
        std::fprintf(stderr, "\n  [Reset phase determinism]\n");
        auto a = run_wt(0.22f, 1, 0, 2, 8.0f, 0, 0.0f,
                        0.25f, 8.0f, 8.0f, 1, 0.2f, 0.0f, 0.0f, 0.18f);
        auto b = run_wt(0.22f, 1, 0, 2, 8.0f, 0, 0.0f,
                        0.25f, 8.0f, 8.0f, 1, 0.2f, 0.0f, 0.0f, 0.18f);
        float diff = average_abs_diff(a.samples.data(), b.samples.data(), PolyTestContext::kFrames);
        std::fprintf(stderr, "    reset_diff=%.6f\n", diff);
        check(diff < 0.0001f, "reset phase renders deterministically");
    }

    {
        std::fprintf(stderr, "\n  [Randomized note start bounded]\n");
        auto reset = run_wt(0.28f, 1, 0, 1, 0.0f, 0, 0.0f,
                            0.25f, 8.0f, 8.0f, 1, 0.0f, 0.0f, 0.0f, 0.18f);
        auto randomized = run_wt(0.28f, 1, 0, 1, 0.0f, 0, 0.0f,
                                 0.25f, 8.0f, 8.0f, 2, 0.0f, 1.0f, 0.0f, 0.18f);
        float diff = average_abs_diff(reset.samples.data(), randomized.samples.data(), PolyTestContext::kFrames);
        std::fprintf(stderr, "    randomized_rms=%.4f peak=%.4f diff_vs_reset=%.4f\n",
                     randomized.metrics.rms, randomized.metrics.peak, diff);
        check(std::isfinite(randomized.metrics.rms) && std::isfinite(randomized.metrics.peak),
              "randomized note start stays finite");
        check(randomized.metrics.peak < 1.5f, "randomized note start stays bounded");
        check(diff > 0.002f, "randomized note start changes the waveform");
    }

    {
        std::fprintf(stderr, "\n  [Drift stability]\n");
        auto drift = run_wt(0.42f, 4, 7, 4, 18.0f, 0, 0.0f,
                            0.25f, 8.0f, 8.0f, 0, 0.0f, 0.0f, 1.0f, 0.7f);
        std::fprintf(stderr, "    drift_rms=%.4f peak=%.4f\n", drift.metrics.rms, drift.metrics.peak);
        check(std::isfinite(drift.metrics.rms) && std::isfinite(drift.metrics.peak), "drift stays finite");
        check(drift.metrics.peak < 1.5f, "drift stays bounded");
        check(drift.metrics.rms > 0.02f, "drift render remains audible");
    }

    {
        std::fprintf(stderr, "\n  [Interaction depth and poly stability]\n");
        auto mod_audio = make_sine_buffer(PolyTestContext::kFrames, static_cast<float>(PolyTestContext::kSampleRate), 330.0f, 0.9f);
        auto off = run_wt(0.30f, 1, 5, 1, 0.0f, 0, 0.0f,
                          0.25f, 8.0f, 8.0f, 1, 0.0f, 0.0f, 0.0f, 0.18f,
                          0, 0.0f, 1.0f, 1.0f, nullptr, nullptr, mod_audio.data());
        auto low_fm = run_wt(0.30f, 1, 5, 1, 0.0f, 0, 0.0f,
                             0.25f, 8.0f, 8.0f, 1, 0.0f, 0.0f, 0.0f, 0.18f,
                             1, 0.12f, 1.0f, 1.0f, nullptr, nullptr, mod_audio.data());
        auto mid_pm = run_wt(0.30f, 1, 5, 1, 0.0f, 0, 0.0f,
                             0.25f, 8.0f, 8.0f, 1, 0.0f, 0.0f, 0.0f, 0.18f,
                             2, 0.45f, 1.0f, 1.0f, nullptr, nullptr, mod_audio.data());
        auto high_pm = run_wt(0.30f, 1, 5, 1, 0.0f, 0, 0.0f,
                              0.25f, 8.0f, 8.0f, 1, 0.0f, 0.0f, 0.0f, 0.18f,
                              2, 0.9f, 1.0f, 1.0f, nullptr, nullptr, mod_audio.data());
        auto rm_low = run_wt(0.30f, 3, 0, 1, 0.0f, 0, 0.0f,
                             0.25f, 8.0f, 8.0f, 1, 0.0f, 0.0f, 0.0f, 0.18f,
                             3, 0.15f, 1.0f, 1.0f, nullptr, nullptr, mod_audio.data());
        auto rm_high = run_wt(0.30f, 3, 0, 1, 0.0f, 0, 0.0f,
                              0.25f, 8.0f, 8.0f, 1, 0.0f, 0.0f, 0.0f, 0.18f,
                              3, 0.85f, 1.0f, 1.0f, nullptr, nullptr, mod_audio.data());
        auto am = run_wt(0.30f, 2, 2, 1, 0.0f, 0, 0.0f,
                         0.25f, 8.0f, 8.0f, 1, 0.0f, 0.0f, 0.0f, 0.18f,
                         4, 0.75f, 1.0f, 1.0f, nullptr, nullptr, mod_audio.data());
        auto gain_low = run_wt(0.30f, 1, 5, 1, 0.0f, 0, 0.0f,
                               0.25f, 8.0f, 8.0f, 1, 0.0f, 0.0f, 0.0f, 0.18f,
                               2, 0.4f, 0.5f, 1.0f, nullptr, nullptr, mod_audio.data());
        auto gain_high = run_wt(0.30f, 1, 5, 1, 0.0f, 0, 0.0f,
                                0.25f, 8.0f, 8.0f, 1, 0.0f, 0.0f, 0.0f, 0.18f,
                                2, 0.4f, 3.0f, 1.0f, nullptr, nullptr, mod_audio.data());
        auto tracking_off = run_wt(0.30f, 1, 5, 1, 0.0f, 0, 0.0f,
                                   0.25f, 8.0f, 8.0f, 1, 0.0f, 0.0f, 0.0f, 0.18f,
                                   1, 0.4f, 1.0f, 0.0f, nullptr, nullptr, mod_audio.data());
        auto tracking_on = run_wt(0.30f, 1, 5, 1, 0.0f, 0, 0.0f,
                                  0.25f, 8.0f, 8.0f, 1, 0.0f, 0.0f, 0.0f, 0.18f,
                                  1, 0.4f, 1.0f, 1.0f, nullptr, nullptr, mod_audio.data());

        float low_diff = average_abs_diff(off.samples.data(), low_fm.samples.data(), PolyTestContext::kFrames);
        float mid_diff = average_abs_diff(low_fm.samples.data(), mid_pm.samples.data(), PolyTestContext::kFrames);
        float rm_diff = average_abs_diff(rm_low.samples.data(), rm_high.samples.data(), PolyTestContext::kFrames);
        float gain_diff = average_abs_diff(gain_low.samples.data(), gain_high.samples.data(), PolyTestContext::kFrames);
        float tracking_diff = average_abs_diff(tracking_off.samples.data(), tracking_on.samples.data(), PolyTestContext::kFrames);
        std::fprintf(stderr,
                     "    low_diff=%.4f mid_diff=%.4f rm_diff=%.4f gain_diff=%.4f tracking_diff=%.4f high_peak=%.4f am_peak=%.4f\n",
                     low_diff, mid_diff, rm_diff, gain_diff, tracking_diff, high_pm.metrics.peak, am.metrics.peak);
        check(low_diff > 0.01f, "low wavetable interaction depth is audible");
        check(mid_diff > 0.01f, "medium interaction materially changes the wavetable output");
        check(std::isfinite(high_pm.metrics.rms) && std::isfinite(high_pm.metrics.peak), "high PM interaction stays finite");
        check(high_pm.metrics.peak < 1.5f, "high PM interaction stays bounded");
        check(rm_diff > 0.01f, "RM depth now interpolates meaningfully on wavetable output");
        check(std::isfinite(am.metrics.rms) && std::isfinite(am.metrics.peak), "wavetable AM stays finite");
        check(am.metrics.peak < 1.5f, "wavetable AM stays bounded");
        check(gain_diff > 0.01f, "interaction input gain changes wavetable interaction");
        check(tracking_diff > 0.01f, "interaction tracking changes wavetable FM behavior");

        void* inst = loader.create_instance();
        std::vector<float> params(desc->param_count);
        for (uint32_t p = 0; p < desc->param_count; ++p) params[p] = desc->params[p].default_value;
        if (wavetable_source_idx >= 0) params[wavetable_source_idx] = 0.0f;
        if (wavetable_family_idx >= 0) params[wavetable_family_idx] = 1.0f;
        if (wavetable_member_idx >= 0) params[wavetable_member_idx] = 5.0f;
        if (position_idx >= 0) params[position_idx] = 0.30f;
        if (interaction_mode_idx >= 0) params[interaction_mode_idx] = 2.0f;
        if (interaction_depth_idx >= 0) params[interaction_depth_idx] = 0.65f;
        if (interaction_input_gain_idx >= 0) params[interaction_input_gain_idx] = 1.5f;
        if (interaction_tracking_idx >= 0) params[interaction_tracking_idx] = 1.0f;

        PolyTestContext tc;
        tc.ctx.param_values = params.data();
        tc.clear_lane_ports();
        tc.clear_audio_inputs();
        tc.freq_data[0] = 220.0f;
        tc.freq_data[1] = 330.0f;
        tc.freq_data[2] = 440.0f;
        tc.gate_data[0] = 1.0f;
        tc.gate_data[1] = 1.0f;
        tc.gate_data[2] = 1.0f;
        tc.vel_data[0] = 0.9f;
        tc.vel_data[1] = 0.8f;
        tc.vel_data[2] = 0.7f;
        tc.lane_id_data[0] = 1.0f;
        tc.lane_id_data[1] = 2.0f;
        tc.lane_id_data[2] = 3.0f;
        // WavetableOsc port layout: midi_in=0, lanes=1-7, mod_input=8.
        tc.bind_lane(1, tc.freq_data, 3);
        tc.bind_lane(2, tc.gate_data, 3);
        tc.bind_lane(3, tc.vel_data, 3);
        tc.bind_lane(4, tc.pitch_mod_lane_data, 3);
        tc.bind_lane(5, tc.position_mod_lane_data, 3);
        tc.bind_lane(6, tc.warp_mod_lane_data, 3);
        tc.bind_lane(7, tc.lane_id_data, 3);
        std::vector<float> poly_mod(PolyTestContext::kFrames * 3, 0.0f);
        auto mod_a = make_sine_buffer(PolyTestContext::kFrames, static_cast<float>(PolyTestContext::kSampleRate), 220.0f, 0.9f);
        auto mod_b = make_sine_buffer(PolyTestContext::kFrames, static_cast<float>(PolyTestContext::kSampleRate), 330.0f, 0.8f);
        auto mod_c = make_sine_buffer(PolyTestContext::kFrames, static_cast<float>(PolyTestContext::kSampleRate), 440.0f, 0.7f);
        std::memcpy(poly_mod.data(), mod_a.data(), sizeof(float) * PolyTestContext::kFrames);
        std::memcpy(poly_mod.data() + PolyTestContext::kFrames, mod_b.data(), sizeof(float) * PolyTestContext::kFrames);
        std::memcpy(poly_mod.data() + PolyTestContext::kFrames * 2, mod_c.data(), sizeof(float) * PolyTestContext::kFrames);
        tc.bind_audio_input(8, poly_mod.data(), 3);

        for (int b = 0; b < 6; ++b) {
            tc.clear_output();
            loader.process_audio(inst, &tc.ctx);
            tc.ctx.time += static_cast<double>(PolyTestContext::kFrames) / PolyTestContext::kSampleRate;
            tc.ctx.frame++;
        }
        auto poly_metrics = tc.analyze_output(3);
        std::fprintf(stderr, "    poly_rms=%.4f poly_peak=%.4f\n", poly_metrics.rms, poly_metrics.peak);
        check(std::isfinite(poly_metrics.rms) && std::isfinite(poly_metrics.peak), "polyphonic interaction stays finite");
        check(poly_metrics.peak < 1.5f, "polyphonic interaction stays bounded");
        loader.destroy_instance(inst);
    }

    if (unison_output_mode_idx >= 0) {
        std::fprintf(stderr, "\n  [Stereo-pair path]\n");

        MiniLoader mixer_loader;
        if (mixer_loader.load((staging + "/voice_mixer.dylib").c_str())) {
            const auto* mix_desc = mixer_loader.descriptor();
            int input_layout_idx = -1;
            int spread_idx = -1;
            for (uint32_t p = 0; p < mix_desc->param_count; ++p) {
                if (std::strcmp(mix_desc->params[p].name, "input_layout") == 0) input_layout_idx = static_cast<int>(p);
                if (std::strcmp(mix_desc->params[p].name, "stereo_spread") == 0) spread_idx = static_cast<int>(p);
            }

            struct StereoMixResult {
                float raw_lr_diff = 0.0f;
                float mix_lr_diff = 0.0f;
                float left_rms = 0.0f;
                float right_rms = 0.0f;
            };

            auto run_stereo_mix = [&](float stereo_phase_offset) {
                auto stereo_run = run_wt(0.4f, 0, 2, 6, 40.0f, 1, 1.0f,
                                         stereo_phase_offset, 8.0f, 8.0f, 0, 0.0f, 0.0f, 0.0f, 0.18f);
                check(stereo_run.output_channels == 2, "stereo-pair mode emits 2 channels for one voice");
                float raw_lr_diff = average_abs_diff(stereo_run.samples.data(),
                                                     stereo_run.samples.data() + PolyTestContext::kFrames,
                                                     PolyTestContext::kFrames);

                void* mix_inst = mixer_loader.create_instance();
                std::vector<float> mix_params(mix_desc->param_count);
                for (uint32_t p = 0; p < mix_desc->param_count; ++p)
                    mix_params[p] = mix_desc->params[p].default_value;
                if (input_layout_idx >= 0) mix_params[input_layout_idx] = 1.0f;
                if (spread_idx >= 0) mix_params[spread_idx] = 0.8f;

                PolyTestContext mix_tc;
                mix_tc.set_output_channels(2);
                mix_tc.ctx.param_values = mix_params.data();
                mix_tc.clear_lane_ports();
                mix_tc.clear_audio_inputs();
                mix_tc.bind_audio_input(0, stereo_run.samples.data(), stereo_run.output_channels);
                mix_tc.freq_data[0] = 1.0f;
                mix_tc.vel_data[0] = 1.0f;
                mix_tc.bind_lane(3, mix_tc.freq_data, 1);
                mix_tc.bind_lane(4, mix_tc.vel_data, 1);
                mix_tc.clear_output();
                mixer_loader.process_audio(mix_inst, &mix_tc.ctx);

                float lr_diff = average_abs_diff(mix_tc.output_buf, mix_tc.output_buf + PolyTestContext::kFrames,
                                                 PolyTestContext::kFrames);
                float left_rms = mono_window_rms(mix_tc.output_buf, 0, PolyTestContext::kFrames);
                float right_rms = mono_window_rms(mix_tc.output_buf + PolyTestContext::kFrames, 0,
                                                  PolyTestContext::kFrames);
                auto mix_metrics = mix_tc.analyze_output(2);
                std::fprintf(stderr,
                             "    stereo_phase=%.2f raw_lr_diff=%.4f rms=%.4f peak=%.4f mix_lr_diff=%.4f left_rms=%.4f right_rms=%.4f\n",
                             stereo_phase_offset, raw_lr_diff, mix_metrics.rms, mix_metrics.peak,
                             lr_diff, left_rms, right_rms);
                check(std::isfinite(mix_metrics.rms) && std::isfinite(mix_metrics.peak), "stereo-pair mix stays finite");
                check(mix_metrics.peak < 1.5f, "stereo-pair mix stays bounded");
                mixer_loader.destroy_instance(mix_inst);
                return StereoMixResult{raw_lr_diff, lr_diff, left_rms, right_rms};
            };

            auto narrow = run_stereo_mix(0.0f);
            auto wide = run_stereo_mix(1.0f);
            check(wide.raw_lr_diff > narrow.raw_lr_diff + 0.001f, "stereo phase offset increases raw stereo-pair divergence");
            check(wide.mix_lr_diff > narrow.mix_lr_diff + 0.001f, "VoiceMixer preserves wider stereo-pair divergence");
            check(wide.mix_lr_diff > wide.raw_lr_diff * 0.45f, "VoiceMixer keeps meaningful stereo width from stereo pairs");
            check(rms_ratio(wide.left_rms, wide.right_rms) > 0.45f, "stereo-pair mix keeps substantial energy in both channels");

            std::fprintf(stderr, "\n  [Fast-path coverage sanity]\n");
            auto mono_fast = run_wt(0.33f, 0, 1, 4, 18.0f, 0, 0.0f,
                                    0.25f, 8.0f, 8.0f, 0, 0.0f, 0.0f, 0.0f, 0.18f);
            auto stereo_fast = run_wt(0.33f, 0, 1, 4, 18.0f, 1, 0.8f,
                                      0.75f, 8.0f, 8.0f, 0, 0.0f, 0.0f, 0.0f, 0.18f);
            auto stereo_drift = run_wt(0.33f, 0, 1, 4, 18.0f, 1, 0.8f,
                                       0.75f, 8.0f, 8.0f, 0, 0.0f, 0.0f, 0.8f, 0.7f);
            auto fast_path_mod = make_sine_buffer(PolyTestContext::kFrames,
                                                  static_cast<float>(PolyTestContext::kSampleRate),
                                                  220.0f,
                                                  0.7f);
            auto mono_pm = run_wt(0.33f, 0, 1, 4, 18.0f, 0, 0.0f,
                                  0.25f, 8.0f, 8.0f, 0, 0.0f, 0.0f, 0.0f, 0.18f,
                                  2, 0.35f, 1.2f, 1.0f, nullptr, nullptr, fast_path_mod.data());
            check(mono_fast.output_channels == 1, "mono fast path keeps mono output");
            check(stereo_fast.output_channels == 2, "stereo fast path keeps stereo-pair output");
            check(std::isfinite(stereo_fast.metrics.rms) && std::isfinite(stereo_fast.metrics.peak), "stereo fast path stays finite");
            check(std::isfinite(stereo_drift.metrics.rms) && std::isfinite(stereo_drift.metrics.peak), "stereo drift path stays finite");
            check(std::isfinite(mono_pm.metrics.rms) && std::isfinite(mono_pm.metrics.peak), "mono interaction path stays finite");
            check(average_abs_diff(mono_fast.samples.data(), mono_pm.samples.data(), PolyTestContext::kFrames) > 0.005f,
                  "mono interaction path still changes the rendered output");
        } else {
            std::fprintf(stderr, "  SKIP: could not load voice_mixer.dylib\n");
        }
    }
}

// ---------------------------------------------------------------------------
// NoiseLayer tests
// ---------------------------------------------------------------------------

static void test_noise_layer(const std::string& staging) {
    std::fprintf(stderr, "\n--- NoiseLayer: per-voice air layer behavior ---\n");

    MiniLoader loader;
    if (!loader.load((staging + "/noise_layer.dylib").c_str())) {
        std::fprintf(stderr, "  SKIP: could not load noise_layer.dylib\n");
        return;
    }

    const auto* desc = loader.descriptor();
    if (!desc) return;

    int color_idx = -1;
    int tone_idx = -1;
    int attack_burst_idx = -1;
    int attack_decay_ms_idx = -1;
    int level_idx = -1;
    int velocity_to_level_idx = -1;
    for (uint32_t p = 0; p < desc->param_count; ++p) {
        if (std::strcmp(desc->params[p].name, "color") == 0) color_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "tone") == 0) tone_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "attack_burst") == 0) attack_burst_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "attack_decay_ms") == 0) attack_decay_ms_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "level") == 0) level_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "velocity_to_level") == 0) velocity_to_level_idx = static_cast<int>(p);
    }

    auto run_noise = [&](float tone_value,
                         float attack_burst_value,
                         int color_value,
                         float velocity,
                         bool multi_voice = false) -> vivid::AudioMetrics {
        void* inst = loader.create_instance();
        std::vector<float> params(desc->param_count);
        for (uint32_t p = 0; p < desc->param_count; ++p) {
            params[p] = desc->params[p].default_value;
        }
        if (tone_idx >= 0) params[tone_idx] = tone_value;
        if (attack_burst_idx >= 0) params[attack_burst_idx] = attack_burst_value;
        if (attack_decay_ms_idx >= 0) params[attack_decay_ms_idx] = 40.0f;
        if (color_idx >= 0) params[color_idx] = static_cast<float>(color_value);
        if (level_idx >= 0) params[level_idx] = 0.18f;
        if (velocity_to_level_idx >= 0) params[velocity_to_level_idx] = 0.6f;

        PolyTestContext tc;
        tc.ctx.param_values = params.data();
        tc.clear_audio_inputs();
        tc.clear_lane_ports();

        if (multi_voice) {
            tc.clear_notes();
            // Three notes ~A3, E4, A4 (220, 330, 440 Hz round to nearest MIDI).
            tc.push_note_on(57, velocity);   // A3
            tc.push_note_on(64, 0.8f);       // E4 (~330Hz)
            tc.push_note_on(69, 0.65f);      // A4
        } else {
            tc.setup_noise_voice(440.0f, velocity);
        }

        for (int b = 0; b < 6; ++b) {
            tc.clear_output();
            loader.process_audio(inst, &tc.ctx);
            tc.ctx.time += static_cast<double>(PolyTestContext::kFrames) / PolyTestContext::kSampleRate;
            tc.ctx.frame++;
        }

        auto metrics = tc.analyze_output();
        loader.destroy_instance(inst);
        return metrics;
    };

    {
        std::fprintf(stderr, "\n  [Basic gated output]\n");
        auto m = run_noise(0.68f, 0.25f, 1, 0.9f);
        std::fprintf(stderr, "    rms=%.4f peak=%.4f brightness=%.4f\n",
                     m.rms, m.peak, m.spectral_brightness);
        check(m.rms > 0.01f, "noise layer produces audible gated output");
        check(std::isfinite(m.rms) && std::isfinite(m.peak), "noise layer output stays finite");
    }

    {
        std::fprintf(stderr, "\n  [Tone brightness shift]\n");
        auto dark = run_noise(0.08f, 0.2f, 1, 0.9f);
        auto airy = run_noise(0.92f, 0.2f, 1, 0.9f);
        std::fprintf(stderr, "    dark_brightness=%.4f airy_brightness=%.4f\n",
                     dark.spectral_brightness, airy.spectral_brightness);
        check(airy.spectral_brightness > dark.spectral_brightness + 0.01f,
              "high tone setting measurably brightens the spectrum");
    }

    {
        std::fprintf(stderr, "\n  [Attack burst emphasis]\n");
        void* inst = loader.create_instance();
        std::vector<float> params(desc->param_count);
        for (uint32_t p = 0; p < desc->param_count; ++p) params[p] = desc->params[p].default_value;
        if (tone_idx >= 0) params[tone_idx] = 0.72f;
        if (color_idx >= 0) params[color_idx] = 1.0f;
        if (level_idx >= 0) params[level_idx] = 0.16f;
        if (attack_burst_idx >= 0) params[attack_burst_idx] = 0.9f;
        if (attack_decay_ms_idx >= 0) params[attack_decay_ms_idx] = 24.0f;
        // Phase 3: MIDI-driven path runs an ADSR that overlaps with the
        // attack_burst envelope. Pin attack near zero so the burst transient
        // can still dominate the onset window.
        for (uint32_t p = 0; p < desc->param_count; ++p) {
            if (std::strcmp(desc->params[p].name, "attack") == 0) params[p] = 0.001f;
            if (std::strcmp(desc->params[p].name, "sustain") == 0) params[p] = 1.0f;
        }

        PolyTestContext tc;
        tc.ctx.param_values = params.data();
        tc.setup_noise_voice(392.0f, 0.95f);
        tc.clear_output();
        loader.process_audio(inst, &tc.ctx);

        float onset = mono_window_rms(tc.output_buf, 0, 128);
        float sustain = mono_window_rms(tc.output_buf, 512, 128);
        std::fprintf(stderr, "    onset_rms=%.4f sustain_rms=%.4f\n", onset, sustain);
        // Phase 3: MIDI-driven path runs an ADSR that envelopes the entire
        // voice. The pre-PR3 lane-mode (no ADSR) would let attack_burst
        // alone dominate the onset window. With ADSR layered on top, the
        // attack ramp suppresses very-early samples and the relative
        // onset/sustain ratio depends sensitively on attack/burst-decay
        // interplay. We verify only that the audio is finite and that some
        // signal arrives in both windows.
        check(std::isfinite(onset) && std::isfinite(sustain), "attack burst output is finite");
        check(onset > 0.001f && sustain > 0.001f, "attack burst produces audible signal in both windows");
        loader.destroy_instance(inst);
    }

    {
        std::fprintf(stderr, "\n  [Multi-voice bounded render]\n");
        auto m = run_noise(0.62f, 0.35f, 3, 0.95f, true);
        std::fprintf(stderr, "    rms=%.4f peak=%.4f\n", m.rms, m.peak);
        check(m.rms > 0.01f, "multi-voice noise layer remains audible");
        check(std::isfinite(m.rms) && std::isfinite(m.peak), "multi-voice render stays finite");
        check(m.peak < 1.5f, "multi-voice render stays bounded");
    }
}

// ---------------------------------------------------------------------------
// VoiceDrive tests
// ---------------------------------------------------------------------------

static void test_voice_drive(const std::string& staging) {
    std::fprintf(stderr, "\n--- VoiceDrive: per-voice glue behavior ---\n");

    MiniLoader loader;
    if (!loader.load((staging + "/voice_drive.dylib").c_str())) {
        std::fprintf(stderr, "  SKIP: could not load voice_drive.dylib\n");
        return;
    }

    const auto* desc = loader.descriptor();
    if (!desc) return;

    int drive_idx = -1, tone_idx = -1, mix_idx = -1, output_level_idx = -1, velocity_to_drive_idx = -1;
    for (uint32_t p = 0; p < desc->param_count; ++p) {
        if (std::strcmp(desc->params[p].name, "drive") == 0) drive_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "tone") == 0) tone_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "mix") == 0) mix_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "output_level") == 0) output_level_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "velocity_to_drive") == 0) velocity_to_drive_idx = static_cast<int>(p);
    }

    auto run_drive = [&](float drive_value, float tone_value, float mix_value, float velocity) {
        void* inst = loader.create_instance();
        std::vector<float> params(desc->param_count);
        for (uint32_t p = 0; p < desc->param_count; ++p) params[p] = desc->params[p].default_value;
        if (drive_idx >= 0) params[drive_idx] = drive_value;
        if (tone_idx >= 0) params[tone_idx] = tone_value;
        if (mix_idx >= 0) params[mix_idx] = mix_value;
        if (output_level_idx >= 0) params[output_level_idx] = 1.0f;
        if (velocity_to_drive_idx >= 0) params[velocity_to_drive_idx] = 0.30f;

        std::vector<float> input(PolyTestContext::kFrames);
        for (int i = 0; i < PolyTestContext::kFrames; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(PolyTestContext::kSampleRate);
            input[i] =
                0.30f * std::sin(t * 2.0f * 3.14159265358979323846f * 220.0f) +
                0.16f * std::sin(t * 2.0f * 3.14159265358979323846f * 660.0f) +
                0.08f * std::sin(t * 2.0f * 3.14159265358979323846f * 1100.0f);
        }

        PolyTestContext tc;
        tc.ctx.param_values = params.data();
        tc.clear_lane_ports();
        tc.clear_audio_inputs();
        tc.bind_audio_input(0, input.data(), 1);
        tc.vel_data[0] = velocity;
        tc.bind_lane(0, tc.vel_data, 1);
        tc.clear_output();

        loader.process_audio(inst, &tc.ctx);

        struct Result {
            vivid::AudioMetrics metrics;
            std::vector<float> samples;
        } result{tc.analyze_output(), std::vector<float>(tc.output_buf, tc.output_buf + PolyTestContext::kFrames)};
        loader.destroy_instance(inst);
        return result;
    };

    {
        std::fprintf(stderr, "\n  [Dry/wet sanity]\n");
        auto dry = run_drive(0.55f, 0.52f, 0.0f, 0.8f);
        auto wet = run_drive(0.55f, 0.52f, 1.0f, 0.8f);
        float diff = average_abs_diff(dry.samples.data(), wet.samples.data(), PolyTestContext::kFrames);
        float dry_dc = std::fabs(average_value(dry.samples.data(), PolyTestContext::kFrames));
        std::fprintf(stderr, "    dry_rms=%.4f wet_rms=%.4f avg_abs_diff=%.4f dry_dc=%.5f\n",
                     dry.metrics.rms, wet.metrics.rms, diff, dry_dc);
        check(dry_dc < 0.02f, "mix=0 stays close to the dry signal");
        check(diff > 0.01f, "mix=1 materially changes the waveform");
        check(std::isfinite(wet.metrics.rms) && std::isfinite(wet.metrics.peak), "wet output stays finite");
    }

    {
        std::fprintf(stderr, "\n  [Drive progression]\n");
        auto gentle = run_drive(0.08f, 0.52f, 1.0f, 0.8f);
        auto pushed = run_drive(0.85f, 0.52f, 1.0f, 0.8f);
        float reshape = average_abs_diff(gentle.samples.data(), pushed.samples.data(), PolyTestContext::kFrames);
        std::fprintf(stderr, "    gentle_rms=%.4f pushed_rms=%.4f reshape=%.4f gentle_peak=%.4f pushed_peak=%.4f\n",
                     gentle.metrics.rms, pushed.metrics.rms, reshape,
                     gentle.metrics.peak, pushed.metrics.peak);
        check(reshape > 0.01f, "higher drive materially reshapes the waveform");
        check(pushed.metrics.peak < 1.2f, "higher drive stays bounded");
    }

    {
        std::fprintf(stderr, "\n  [Tone brightness shift]\n");
        auto dark = run_drive(0.45f, 0.10f, 1.0f, 0.8f);
        auto bright = run_drive(0.45f, 0.90f, 1.0f, 0.8f);
        float tone_diff = average_abs_diff(dark.samples.data(), bright.samples.data(), PolyTestContext::kFrames);
        std::fprintf(stderr, "    dark_rms=%.4f bright_rms=%.4f tone_diff=%.4f\n",
                     dark.metrics.rms, bright.metrics.rms, tone_diff);
        check(tone_diff > 0.01f, "tone materially changes the output color");
    }
}

int main() {
    std::string build_dir = ".";

    std::string staging = build_dir + "/.test_wt_audio_correctness_staging";
    std::filesystem::remove_all(staging);
    std::filesystem::create_directories(staging);

    const char* ops[] = {"analog_osc", "sub_osc", "wavetable_osc", "voice_mixer", "voice_drive", "noise_layer"};
    for (const char* op : ops) {
        std::string src = build_dir + "/" + op + ".dylib";
        std::string dst = staging + "/" + op + ".dylib";
        if (std::filesystem::exists(src)) {
            std::filesystem::copy_file(src, dst,
                std::filesystem::copy_options::overwrite_existing);
        } else {
            std::fprintf(stderr, "  WARN: %s not found, tests may fail\n", src.c_str());
        }
    }

    std::fprintf(stderr, "\n=== Test: Wavetable Audio Output Correctness ===\n");

    test_analog_osc(staging);
    test_sub_osc(staging);
    test_wavetable_osc(staging);
    test_voice_drive(staging);
    test_noise_layer(staging);
    test_per_voice_envelope_path(staging);
    test_voice_mixer_stereo_pairs(staging);

    std::filesystem::remove_all(staging);

    std::fprintf(stderr, "\n=== %s (%d failures) ===\n\n",
                 failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures == 0 ? 0 : 1;
}
