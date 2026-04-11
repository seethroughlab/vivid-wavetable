#pragma once

#include "operator_api/operator.h"
#include "wavetable_bank.h"
#include "wavetable_dsp.h"
#include "wavetable_layer_renderer.h"

#include <array>
#include <atomic>
#include <string>

using vivid_wavetable::bank::Wavetable;
using vivid_wavetable::bank::kBuiltinWavetableCount;

/**
 * @brief Production polyphonic wavetable layer with internal unison, stereo summing, and SIMD-ready architecture.
 *
 * Renders all active voices and unison sub-voices internally and outputs a stereo
 * mix directly, replacing the WavetableOsc + VoiceMixer chain for production
 * instruments. Accepts per-voice lane inputs from PolyVoiceAllocator and audio-rate
 * modulation for pitch, position, and warp. An external voice_gain_audio input
 * provides per-voice amplitude shaping (typically from EnvelopeAu).
 *
 * @input frequencies Per-voice frequencies from a note allocator.
 * @input gates Per-voice gates used for phase reset and note articulation.
 * @input velocities Per-voice velocities available for graph-level shaping.
 * @input lane_ids Stable per-voice identity tokens for persistent lane state.
 * @input pitch_mod Per-voice pitch modulation lane array.
 * @input position_mod Per-voice wavetable position modulation lane array.
 * @input warp_mod Per-voice warp modulation lane array.
 * @input pitch_mod_audio Audio-rate per-voice pitch modulation.
 * @input position_mod_audio Audio-rate per-voice wavetable position modulation.
 * @input warp_mod_audio Audio-rate per-voice warp modulation.
 * @input voice_gain_audio Audio-rate per-voice amplitude envelope (typically from EnvelopeAu).
 * @output output Stereo mix of all active voices.
 * @tip Use voice_gain_audio from EnvelopeAu for true per-note amplitude shaping.
 * @recipe PolyVoiceAllocator/frequencies,gates,lane_ids -> WavetableLayer/frequencies,gates,lane_ids
 * @recipe EnvelopeAu/value -> WavetableLayer/voice_gain_audio
 * @family voice_source
 * @best_used_with PolyVoiceAllocator, EnvelopeAu, DualFilter
 * @common_companions SubOsc, NoiseLayer
 */
struct WavetableLayer : vivid::OperatorBase, vivid::AudioProcessable {
    static constexpr const char* kName = "WavetableLayer";
    static constexpr bool kTimeDependent = true;
    static constexpr VividLaneBehavior kLaneBehavior = VIVID_LANE_REDUCTION;

    static constexpr int kMaxVoices = 16;
    static constexpr int kMaxUnisonVoices = 16;

    enum WavetableSourceMode {
        SOURCE_BUILTIN = 0,
        SOURCE_CUSTOM = 1,
    };

    enum PhaseResetMode {
        PHASE_FREE_RUN = 0,
        PHASE_RESET = 1,
        PHASE_RANDOMIZED = 2,
    };

    // --- Parameters ---
    vivid::Param<int> wavetable_source {"wavetable_source", 0, {"Builtin", "Custom"}};
    vivid::Param<int> wavetable_family {"wavetable_family", 0, {"AnalogWarm", "BrightDigital", "VocalFormant", "Metallic", "HarmonicSpectral", "TextureMotion"}};
    vivid::Param<int> wavetable_member {"wavetable_member", 0, {"Core", "Soft", "Rich", "Hollow", "Sweep", "Glass", "Edge", "Air"}};
    vivid::Param<vivid::FilePath> wav_file {"wav_file"};
    vivid::Param<float> position {"position", 0.0f, 0.0f, 1.0f};
    vivid::Param<float> amplitude {"amplitude", 0.3f, 0.0f, 1.0f};
    vivid::Param<int> warp_mode {"warp_mode", 0, {"None", "Sync", "BendPlus", "BendMinus", "Mirror", "Asym", "Quantize", "Flip"}};
    vivid::Param<float> warp_amount {"warp_amount", 0.0f, 0.0f, 1.0f};
    vivid::Param<float> position_smooth_ms {"position_smooth_ms", 8.0f, 0.0f, 40.0f};
    vivid::Param<float> warp_smooth_ms {"warp_smooth_ms", 8.0f, 0.0f, 40.0f};
    vivid::Param<float> drift_amount {"drift_amount", 0.0f, 0.0f, 1.0f};
    vivid::Param<float> drift_rate_hz {"drift_rate_hz", 0.18f, 0.02f, 2.0f};
    vivid::Param<int> phase_reset_mode {"phase_reset_mode", 0, {"FreeRun", "Reset", "Randomized"}};
    vivid::Param<float> start_phase {"start_phase", 0.0f, 0.0f, 1.0f};
    vivid::Param<float> phase_random {"phase_random", 0.0f, 0.0f, 1.0f};
    vivid::Param<float> stereo_phase_offset {"stereo_phase_offset", 0.25f, 0.0f, 1.0f};
    vivid::Param<int> unison_voices {"unison_voices", 1, 1, 16};
    vivid::Param<float> unison_spread {"unison_spread", 20.0f, 0.0f, 100.0f};
    vivid::Param<float> unison_stereo {"unison_stereo", 1.0f, 0.0f, 1.0f};
    vivid::Param<int> unison_spread_mode {"unison_spread_mode", 0, {"Linear", "Exponential", "Random"}};
    vivid::Param<float> detune {"detune", 0.0f, 0.0f, 50.0f};
    vivid::Param<float> portamento {"portamento", 0.0f, 0.0f, 2000.0f};

    // --- Wavetable state ---
    std::atomic<Wavetable*> custom_table_{nullptr};
    Wavetable* deferred_delete_ = nullptr;
    std::string last_wav_path_;
    const Wavetable* cached_table_ = nullptr;

    // --- Renderer working state ---
    vivid_wavetable::layer::PreparedWavetable prepared_wt_;
    vivid_wavetable::layer::RenderUnit render_units_;
    vivid_wavetable::layer::VoiceBlock voice_block_;
    vivid_wavetable::layer::RendererTelemetry renderer_telemetry_;

    // --- Per-voice persistent state (identity-keyed via vivid_lane_state) ---
    struct Voice {
        float phase[kMaxUnisonVoices] = {};
        float drift_phase[kMaxUnisonVoices] = {};
        float current_freq = 0.0f;
        float target_freq = 0.0f;
        vivid_wavetable::dsp::MotionSmoother pos_smoother;
        vivid_wavetable::dsp::MotionSmoother warp_smoother;
        bool was_gated = false;
        bool initialized = false;
        int declick_remaining = 0;
    };

    WavetableLayer();
    ~WavetableLayer();

    void collect_params(std::vector<vivid::ParamBase*>& out) override;
    void collect_ports(std::vector<VividPortDescriptor>& out) override;
    void prepare_instance_assets() override;
    void main_thread_update(double) override;
    void process_audio(const VividAudioContext* ctx) override;

    const Wavetable* resolve_table() const;
    static const std::array<Wavetable, kBuiltinWavetableCount>& builtin_tables();
};
