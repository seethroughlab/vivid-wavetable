// Audio output correctness tests for vivid-wavetable operators.
// Verifies that polyphonic oscillators produce expected spectral and amplitude
// properties using analyze_audio() — property-based, no golden files.

#include "operator_api/types.h"
#include "runtime/output_analyzer.h"
#include "runtime/shared_handle_registry.h"
#include "envelope.h"
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

struct LaneStateStore {
    std::unordered_map<uint64_t, std::vector<uint8_t>> slots;
};

static void* test_lane_state_fn(void* service, uint32_t lane_id, uint32_t byte_size) {
    auto* store = static_cast<LaneStateStore*>(service);
    uint64_t key = (static_cast<uint64_t>(lane_id) << 32) | static_cast<uint64_t>(byte_size);
    auto& slot = store->slots[key];
    if (slot.size() != byte_size) slot.assign(byte_size, 0);
    return slot.data();
}

// ---------------------------------------------------------------------------
// Minimal operator loader — avoids vivid-core's heavy operator_loader.cpp deps
// ---------------------------------------------------------------------------

struct MiniLoader {
    void* handle_ = nullptr;
    VividDescriptorFn  desc_fn_    = nullptr;
    VividCreateFn      create_fn_  = nullptr;
    VividDestroyFn     destroy_fn_ = nullptr;
    VividProcessAudioFn audio_fn_  = nullptr;

    bool load(const char* path) {
        handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle_) { std::fprintf(stderr, "  dlopen: %s\n", dlerror()); return false; }
        desc_fn_    = reinterpret_cast<VividDescriptorFn>(dlsym(handle_, "vivid_descriptor"));
        create_fn_  = reinterpret_cast<VividCreateFn>(dlsym(handle_, "vivid_create"));
        destroy_fn_ = reinterpret_cast<VividDestroyFn>(dlsym(handle_, "vivid_destroy"));
        audio_fn_   = reinterpret_cast<VividProcessAudioFn>(dlsym(handle_, "vivid_process_audio"));
        return desc_fn_ && create_fn_ && destroy_fn_ && audio_fn_;
    }
    ~MiniLoader() { if (handle_) dlclose(handle_); }

    const VividOperatorDescriptor* descriptor() const { return desc_fn_ ? desc_fn_() : nullptr; }
    void* create_instance() const { return create_fn_ ? create_fn_() : nullptr; }
    void  destroy_instance(void* inst) const { if (destroy_fn_ && inst) destroy_fn_(inst); }
    void  process_audio(void* inst, VividAudioContext* ctx) const { if (audio_fn_) audio_fn_(inst, ctx); }
};

// ---------------------------------------------------------------------------
// Polyphonic test context — mirrors the current vivid-core audio contract
// ---------------------------------------------------------------------------

struct PolyTestContext {
    static constexpr int kFrames = 2048;
    static constexpr uint32_t kSampleRate = 48000;
    static constexpr int kMaxVoices = 16;
    static constexpr int kMaxAudioChannels = 32;
    static constexpr int kMaxPorts = 16;

    float freq_data[kMaxVoices] = {};
    float gate_data[kMaxVoices] = {};
    float vel_data[kMaxVoices] = {};
    float pitch_mod_lane_data[kMaxVoices] = {};
    float position_mod_lane_data[kMaxVoices] = {};
    float warp_mod_lane_data[kMaxVoices] = {};
    float lane_id_data[kMaxVoices] = {};

    VividLanePort input_lanes[kMaxPorts] = {};

    float output_buf[kMaxAudioChannels * kFrames] = {};
    float* output_ptrs[kMaxPorts] = {};
    uint8_t output_ch[kMaxPorts] = {};

    float* input_ptrs[kMaxPorts] = {};
    uint8_t input_ch[kMaxPorts] = {};

    LaneStateStore lane_state;
    VividAudioContext ctx{};

    PolyTestContext() {
        output_ptrs[0] = output_buf;
        output_ch[0] = kMaxAudioChannels;

        ctx.sample_rate          = kSampleRate;
        ctx.buffer_size          = kFrames;
        ctx.input_buffers        = input_ptrs;
        ctx.output_buffers       = output_ptrs;
        ctx.input_channel_counts = input_ch;
        ctx.output_channel_counts = output_ch;
        ctx.input_lanes          = input_lanes;
        ctx.output_lanes         = nullptr;
        ctx.param_values         = nullptr;
        ctx.shared_handles       = vivid::shared_handle_service();
        ctx.lane_count           = 1;
        ctx.lane_index           = 0;
        ctx.lane_set_id          = 0;
        ctx.lane_id              = 1;
        ctx.lane_state_fn        = test_lane_state_fn;
        ctx.lane_state_service   = &lane_state;
    }

    void clear_lane_ports() {
        for (auto& lane : input_lanes) lane = {nullptr, 0, 0};
    }

    void bind_lane(uint32_t port_idx, float* data, uint32_t length) {
        input_lanes[port_idx].data = data;
        input_lanes[port_idx].length = length;
        input_lanes[port_idx].capacity = length;
    }

    void setup_analog_voice(float freq, float velocity = 1.0f) {
        clear_lane_ports();
        freq_data[0] = freq;
        gate_data[0] = 1.0f;
        vel_data[0] = velocity;
        pitch_mod_lane_data[0] = 0.0f;
        lane_id_data[0] = 1.0f;
        bind_lane(0, freq_data, 1);
        bind_lane(1, gate_data, 1);
        bind_lane(2, vel_data, 1);
        bind_lane(3, pitch_mod_lane_data, 1);
        bind_lane(4, lane_id_data, 1);
    }

    void setup_sub_voice(float freq) {
        clear_lane_ports();
        freq_data[0] = freq;
        gate_data[0] = 1.0f;
        lane_id_data[0] = 1.0f;
        bind_lane(0, freq_data, 1);
        bind_lane(1, gate_data, 1);
        bind_lane(2, lane_id_data, 1);
    }

    void setup_wavetable_voice(float freq, float velocity = 1.0f) {
        clear_lane_ports();
        freq_data[0] = freq;
        gate_data[0] = 1.0f;
        vel_data[0] = velocity;
        pitch_mod_lane_data[0] = 0.0f;
        position_mod_lane_data[0] = 0.0f;
        warp_mod_lane_data[0] = 0.0f;
        lane_id_data[0] = 1.0f;
        bind_lane(0, freq_data, 1);
        bind_lane(1, gate_data, 1);
        bind_lane(2, vel_data, 1);
        bind_lane(3, pitch_mod_lane_data, 1);
        bind_lane(4, position_mod_lane_data, 1);
        bind_lane(5, warp_mod_lane_data, 1);
        bind_lane(6, lane_id_data, 1);
    }

    void setup_noise_voice(float freq, float velocity = 1.0f) {
        clear_lane_ports();
        freq_data[0] = freq;
        gate_data[0] = 1.0f;
        vel_data[0] = velocity;
        lane_id_data[0] = 1.0f;
        bind_lane(0, freq_data, 1);
        bind_lane(1, gate_data, 1);
        bind_lane(2, vel_data, 1);
        bind_lane(3, lane_id_data, 1);
    }

    void clear_output() {
        std::memset(output_buf, 0, sizeof(output_buf));
    }

    void set_output_channels(uint8_t channels) {
        output_ch[0] = channels;
    }

    void bind_audio_input(uint32_t port_idx, float* data, uint8_t channels) {
        input_ptrs[port_idx] = data;
        input_ch[port_idx] = channels;
    }

    void clear_audio_inputs() {
        for (int i = 0; i < kMaxPorts; ++i) {
            input_ptrs[i] = nullptr;
            input_ch[i] = 0;
        }
    }

    void silence_gate() {
        gate_data[0] = 0.0f;
    }

    vivid::AudioMetrics analyze_output(uint32_t channels = 1) const {
        return vivid::analyze_audio(output_buf, kFrames, kSampleRate, channels);
    }
};

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
    for (uint32_t p = 0; p < desc->param_count; ++p) {
        if (std::strcmp(desc->params[p].name, "stereo_spread") == 0) {
            spread_idx = static_cast<int>(p);
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

    loader.destroy_instance(inst);
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

    int waveform_idx = -1, amplitude_idx = -1;
    for (uint32_t p = 0; p < desc->param_count; p++) {
        if (std::strcmp(desc->params[p].name, "waveform") == 0) waveform_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "amplitude") == 0) amplitude_idx = static_cast<int>(p);
    }

    auto make_params = [&](int waveform, float amplitude) {
        std::vector<float> params(desc->param_count);
        for (uint32_t p = 0; p < desc->param_count; p++)
            params[p] = desc->params[p].default_value;
        if (waveform_idx >= 0) params[waveform_idx] = static_cast<float>(waveform);
        if (amplitude_idx >= 0) params[amplitude_idx] = amplitude;
        return params;
    };

    auto run_osc = [&](int waveform, float amplitude, float freq) -> vivid::AudioMetrics {
        void* inst = loader.create_instance();
        auto params = make_params(waveform, amplitude);

        PolyTestContext tc;
        tc.ctx.param_values = params.data();
        tc.setup_analog_voice(freq);

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

    int octave_idx = -1, waveform_idx = -1;
    for (uint32_t p = 0; p < desc->param_count; p++) {
        if (std::strcmp(desc->params[p].name, "octave") == 0) octave_idx = static_cast<int>(p);
        if (std::strcmp(desc->params[p].name, "waveform") == 0) waveform_idx = static_cast<int>(p);
    }

    auto run_sub = [&](int octave, float freq) -> vivid::AudioMetrics {
        void* inst = loader.create_instance();
        std::vector<float> params(desc->param_count);
        for (uint32_t p = 0; p < desc->param_count; p++)
            params[p] = desc->params[p].default_value;
        if (octave_idx >= 0) params[octave_idx] = static_cast<float>(octave);
        if (waveform_idx >= 0) params[waveform_idx] = 0.0f;

        PolyTestContext tc;
        tc.ctx.param_values = params.data();
        tc.setup_sub_voice(freq);

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
        auto m = run_sub(0, 440.0f);
        std::fprintf(stderr, "    rms=%.4f centroid=%.1fHz\n", m.rms, m.spectral_centroid_hz);
        check(m.rms > 0.05f, "sub osc produces signal");
        check_float(m.spectral_centroid_hz, 220.0f, 60.0f, "octave -1: centroid near 220Hz");
    }

    {
        std::fprintf(stderr, "\n  [Octave -2, freq=440Hz]\n");
        auto m = run_sub(1, 440.0f);
        std::fprintf(stderr, "    rms=%.4f centroid=%.1fHz\n", m.rms, m.spectral_centroid_hz);
        check(m.rms > 0.05f, "sub osc produces signal");
        check_float(m.spectral_centroid_hz, 110.0f, 60.0f, "octave -2: centroid near 110Hz");
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
                      const float* pos_audio = nullptr,
                      const float* warp_audio = nullptr) -> WavetableRun {
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

        PolyTestContext tc;
        tc.ctx.param_values = params.data();
        tc.setup_wavetable_voice(440.0f);
        tc.clear_audio_inputs();
        if (pos_audio) tc.bind_audio_input(9, const_cast<float*>(pos_audio), 1);
        if (warp_audio) tc.bind_audio_input(10, const_cast<float*>(warp_audio), 1);

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
                                 pos_audio, warp_audio);
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
                                 pos_step, warp_step);
        auto smoothed = run_wt(0.18f, 1, 5, 1, 0.0f, 0, 0.0f,
                               0.25f, 18.0f, 18.0f, 0, 0.0f, 0.0f, 0.0f, 0.18f,
                               pos_step, warp_step);

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
                auto mix_metrics = mix_tc.analyze_output(2);
                std::fprintf(stderr, "    stereo_phase=%.2f raw_lr_diff=%.4f rms=%.4f peak=%.4f mix_lr_diff=%.4f\n",
                             stereo_phase_offset, raw_lr_diff, mix_metrics.rms, mix_metrics.peak, lr_diff);
                check(std::isfinite(mix_metrics.rms) && std::isfinite(mix_metrics.peak), "stereo-pair mix stays finite");
                check(mix_metrics.peak < 1.5f, "stereo-pair mix stays bounded");
                mixer_loader.destroy_instance(mix_inst);
                return raw_lr_diff;
            };

            float narrow_diff = run_stereo_mix(0.0f);
            float wide_diff = run_stereo_mix(1.0f);
            check(wide_diff > narrow_diff + 0.001f, "stereo phase offset increases raw stereo-pair divergence");
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
            tc.freq_data[0] = 220.0f;
            tc.freq_data[1] = 330.0f;
            tc.freq_data[2] = 440.0f;
            tc.gate_data[0] = 1.0f;
            tc.gate_data[1] = 1.0f;
            tc.gate_data[2] = 1.0f;
            tc.vel_data[0] = velocity;
            tc.vel_data[1] = 0.8f;
            tc.vel_data[2] = 0.65f;
            tc.lane_id_data[0] = 1.0f;
            tc.lane_id_data[1] = 2.0f;
            tc.lane_id_data[2] = 3.0f;
            tc.bind_lane(0, tc.freq_data, 3);
            tc.bind_lane(1, tc.gate_data, 3);
            tc.bind_lane(2, tc.vel_data, 3);
            tc.bind_lane(3, tc.lane_id_data, 3);
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

        PolyTestContext tc;
        tc.ctx.param_values = params.data();
        tc.setup_noise_voice(392.0f, 0.95f);
        tc.clear_output();
        loader.process_audio(inst, &tc.ctx);

        float onset = mono_window_rms(tc.output_buf, 0, 128);
        float sustain = mono_window_rms(tc.output_buf, 512, 128);
        std::fprintf(stderr, "    onset_rms=%.4f sustain_rms=%.4f\n", onset, sustain);
        check(onset > sustain * 1.12f, "attack burst makes the onset stronger than steady state");
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

int main() {
    std::string build_dir = ".";

    std::string staging = build_dir + "/.test_wt_audio_correctness_staging";
    std::filesystem::remove_all(staging);
    std::filesystem::create_directories(staging);

    const char* ops[] = {"analog_osc", "sub_osc", "wavetable_osc", "voice_mixer", "noise_layer"};
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
    test_noise_layer(staging);
    test_per_voice_envelope_path(staging);

    std::filesystem::remove_all(staging);

    std::fprintf(stderr, "\n=== %s (%d failures) ===\n\n",
                 failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures == 0 ? 0 : 1;
}
