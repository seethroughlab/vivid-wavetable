// Audio output correctness tests for vivid-wavetable operators.
// Verifies that polyphonic oscillators produce expected spectral and amplitude
// properties using analyze_audio() — property-based, no golden files.

#include "operator_api/types.h"
#include "runtime/output_analyzer.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

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
// Polyphonic test context — sets up lane inputs for voice-based operators
// ---------------------------------------------------------------------------

struct PolyTestContext {
    static constexpr int kFrames = 2048;
    static constexpr uint32_t kSampleRate = 48000;
    static constexpr int kMaxVoices = 16;

    // Lane data (frequencies, gates, velocities, pitch_mod, position_mod, warp_mod)
    float freq_data[kMaxVoices]  = {};
    float gate_data[kMaxVoices]  = {};
    float vel_data[kMaxVoices]   = {};
    float pitch_mod_data[kMaxVoices] = {};
    float position_mod_data[kMaxVoices] = {};
    float warp_mod_data[kMaxVoices] = {};
    VividLanePort lanes[6]       = {};

    // Multi-channel planar output buffer
    float output_buf[kMaxVoices * kFrames] = {};
    float* output_ptrs[16] = {output_buf};  // [0] = main output, rest null
    uint8_t output_ch[16]  = {1};  // set per-test

    // Input buffers — indexed by overall port ordinal (including lane ports).
    // Lane ports get nullptr; audio ports need valid pointers.
    // Max 16 entries covers operators with up to 16 input ports.
    float* input_ptrs[16] = {};
    uint8_t input_ch[16]  = {};

    VividAudioContext ctx{};

    PolyTestContext() {
        // Set up lane port structs
        lanes[0] = {freq_data, 0, 0};
        lanes[1] = {gate_data, 0, 0};
        lanes[2] = {vel_data,  0, 0};
        lanes[3] = {pitch_mod_data, 0, 0};
        lanes[4] = {position_mod_data, 0, 0};
        lanes[5] = {warp_mod_data, 0, 0};

        ctx.sample_rate          = kSampleRate;
        ctx.buffer_size          = kFrames;
        ctx.input_buffers        = input_ptrs;
        ctx.output_buffers       = output_ptrs;
        ctx.input_lanes          = lanes;
        ctx.output_channel_counts = output_ch;
        ctx.input_channel_counts  = input_ch;
        ctx.param_values         = nullptr;
    }

    void setup_single_voice(float freq, float velocity = 1.0f) {
        freq_data[0] = freq;
        gate_data[0] = 1.0f;
        vel_data[0]  = velocity;
        for (int i = 0; i < 6; i++) {
            lanes[i].length = 1;
            lanes[i].capacity = 1;
        }
        output_ch[0] = 1;
    }

    void clear_output() {
        std::memset(output_buf, 0, sizeof(output_buf));
    }

    void silence_gate() {
        gate_data[0] = 0.0f;
    }

    // Analyze the first voice channel output
    vivid::AudioMetrics analyze_output() const {
        return vivid::analyze_audio(output_buf, kFrames, kSampleRate, 1);
    }
};

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

    // Find param indices
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
        tc.setup_single_voice(freq);

        // Process several buffers for stabilization after gate onset
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

    // --- Sine (waveform=0) at 440Hz ---
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

    // --- Saw (waveform=1) at 440Hz ---
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

    // --- Square (waveform=2) at 440Hz ---
    {
        std::fprintf(stderr, "\n  [Square 440Hz]\n");
        auto m = run_osc(2, 0.5f, 440.0f);
        std::fprintf(stderr, "    rms=%.4f centroid=%.1fHz flatness=%.4f brightness=%.4f\n",
                     m.rms, m.spectral_centroid_hz, m.spectral_flatness, m.spectral_brightness);

        brightness_square = m.spectral_brightness;
        check(brightness_square > brightness_sine,
              "square has more high-frequency content than sine");
    }

    // --- Triangle (waveform=3) at 440Hz ---
    {
        std::fprintf(stderr, "\n  [Triangle 440Hz]\n");
        auto m = run_osc(3, 0.5f, 440.0f);
        std::fprintf(stderr, "    rms=%.4f centroid=%.1fHz flatness=%.4f brightness=%.4f\n",
                     m.rms, m.spectral_centroid_hz, m.spectral_flatness, m.spectral_brightness);

        check(m.spectral_brightness < brightness_saw,
              "triangle has less high-frequency content than saw");
    }

    // --- Gate=0 → silence ---
    {
        std::fprintf(stderr, "\n  [Gate=0 → silence]\n");
        void* inst = loader.create_instance();
        auto params = make_params(0, 0.5f);

        PolyTestContext tc;
        tc.ctx.param_values = params.data();
        tc.setup_single_voice(440.0f);
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

    // --- Amplitude scaling ---
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

    // SubOsc has 2 lane inputs: frequencies(0), gates(1), then audio pitch_mod_audio(2)
    auto run_sub = [&](int octave, float freq) -> vivid::AudioMetrics {
        void* inst = loader.create_instance();
        std::vector<float> params(desc->param_count);
        for (uint32_t p = 0; p < desc->param_count; p++)
            params[p] = desc->params[p].default_value;
        if (octave_idx >= 0) params[octave_idx] = static_cast<float>(octave);
        if (waveform_idx >= 0) params[waveform_idx] = 0.0f;  // Sine for clean spectral measurement

        // SubOsc only has 2 lane inputs and 1 audio input
        PolyTestContext tc;
        tc.ctx.param_values = params.data();

        // Set up lanes — SubOsc only uses lanes[0]=frequencies, lanes[1]=gates
        tc.freq_data[0] = freq;
        tc.gate_data[0] = 1.0f;
        tc.lanes[0].length = tc.lanes[0].capacity = 1;
        tc.lanes[1].length = tc.lanes[1].capacity = 1;
        tc.output_ch[0] = 1;

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

    // --- Octave -1 (index 0): 440Hz input → centroid near 220Hz ---
    {
        std::fprintf(stderr, "\n  [Octave -1, freq=440Hz]\n");
        auto m = run_sub(0, 440.0f);
        std::fprintf(stderr, "    rms=%.4f centroid=%.1fHz\n", m.rms, m.spectral_centroid_hz);
        check(m.rms > 0.05f, "sub osc produces signal");
        check_float(m.spectral_centroid_hz, 220.0f, 60.0f, "octave -1: centroid near 220Hz");
    }

    // --- Octave -2 (index 1): 440Hz input → centroid near 110Hz ---
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

    int position_idx = -1;
    for (uint32_t p = 0; p < desc->param_count; p++) {
        if (std::strcmp(desc->params[p].name, "position") == 0) position_idx = static_cast<int>(p);
    }
    check(position_idx >= 0, "position param found");
    if (position_idx < 0) return;

    // WavetableOsc has 6 lane inputs + 4 audio inputs
    auto run_wt = [&](float position) -> vivid::AudioMetrics {
        void* inst = loader.create_instance();
        std::vector<float> params(desc->param_count);
        for (uint32_t p = 0; p < desc->param_count; p++)
            params[p] = desc->params[p].default_value;
        params[position_idx] = position;

        PolyTestContext tc;
        tc.ctx.param_values = params.data();
        tc.setup_single_voice(440.0f);

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

    auto m_pos0 = run_wt(0.0f);
    auto m_pos5 = run_wt(0.5f);

    std::fprintf(stderr, "  pos=0.0: brightness=%.4f centroid=%.1fHz\n",
                 m_pos0.spectral_brightness, m_pos0.spectral_centroid_hz);
    std::fprintf(stderr, "  pos=0.5: brightness=%.4f centroid=%.1fHz\n",
                 m_pos5.spectral_brightness, m_pos5.spectral_centroid_hz);

    check(m_pos0.rms > 0.05f, "wavetable osc produces signal at pos=0");
    check(m_pos5.rms > 0.05f, "wavetable osc produces signal at pos=0.5");

    float brightness_diff = std::fabs(m_pos0.spectral_brightness - m_pos5.spectral_brightness);
    float centroid_diff = std::fabs(m_pos0.spectral_centroid_hz - m_pos5.spectral_centroid_hz);
    std::fprintf(stderr, "  brightness_diff=%.4f centroid_diff=%.1fHz\n",
                 brightness_diff, centroid_diff);

    check(brightness_diff > 0.01f || centroid_diff > 20.0f,
          "different positions produce different timbres");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::string build_dir = ".";

    std::string staging = build_dir + "/.test_wt_audio_correctness_staging";
    std::filesystem::remove_all(staging);
    std::filesystem::create_directories(staging);

    const char* ops[] = {"analog_osc", "sub_osc", "wavetable_osc"};
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

    std::filesystem::remove_all(staging);

    std::fprintf(stderr, "\n=== %s (%d failures) ===\n\n",
                 failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures == 0 ? 0 : 1;
}
