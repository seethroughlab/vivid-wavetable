#include "operator_api/operator.h"
#include "operator_api/audio_dsp.h"
#include "operator_api/type_id.h"
#include "wavetable_bank.h"
#include "wavetable_dsp.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using vivid_wavetable::bank::BuiltinFamily;
using vivid_wavetable::bank::BuiltinMember;
using vivid_wavetable::bank::Wavetable;
using vivid_wavetable::bank::build_builtin_wavetables;
using vivid_wavetable::bank::kBuiltinWavetableCount;
using vivid_wavetable::bank::load_wavetable_from_wav;
using vivid_wavetable::bank::resolve_builtin_wavetable;
using namespace vivid_wavetable::dsp;

// =============================================================================
// WavetableOsc — polyphonic wavetable oscillator with per-voice channel output
// =============================================================================

/**
 * @brief Polyphonic wavetable oscillator with family/member selection, warp, drift, and unison.
 *
 * Produces one audio channel per active voice and supports both lane-based pitch/gate
 * control and audio-rate modulation for wavetable position, warp, and modulation input.
 * Each voice keeps independent phase and motion state keyed by lane identity.
 *
 * @input frequencies Per-voice frequencies from a note allocator.
 * @input gates Per-voice gates used for phase reset and note articulation.
 * @input velocities Per-voice velocities available for graph-level shaping.
 * @input pitch_mod Per-voice pitch modulation lane array.
 * @input position_mod Per-voice wavetable position modulation lane array.
 * @input warp_mod Per-voice warp modulation lane array.
 * @input lane_ids Stable per-voice identity tokens for persistent lane state.
 * @input mod_input Audio-rate modulation input for FM, RM, or AM.
 * @input pitch_mod_audio Audio-rate per-voice pitch modulation.
 * @input position_mod_audio Audio-rate per-voice wavetable position modulation.
 * @input warp_mod_audio Audio-rate per-voice warp modulation.
 * @output output Per-voice audio channels, one channel per active voice or stereo pair voice path.
 * @tip Drive position_mod_audio or warp_mod_audio from per-note envelopes when you want note-shaped timbral movement instead of a global macro sweep.
 * @recipe PolyVoiceAllocator/frequencies,gates,lane_ids -> WavetableOsc/frequencies,gates,lane_ids
 * @recipe EnvelopeAu/value -> WavetableOsc/position_mod_audio
 * @pitfall position_mod and warp_mod are per-voice lane inputs; they are not interchangeable with a single shared global modulation signal when building poly patches.
 * @family voice_source
 * @best_used_with PolyVoiceAllocator, EnvelopeAu, VoiceMixer
 * @common_companions Filter, AnalogOsc, SubOsc
 */
struct WavetableOsc : vivid::OperatorBase, vivid::AudioProcessable {
    static constexpr const char* kName   = "WavetableOsc";
    static constexpr bool kTimeDependent = true;

    static constexpr int kMaxVoices = 16;
    static constexpr int kMaxUnisonVoices = 16;
    static constexpr int kMaxStereoPairVoices = kMaxVoices / 2;
    static constexpr int kDeClickSamples = 16;
    static constexpr float kMaxDriftCents = 8.0f;

    enum WavetableSourceMode {
        SOURCE_BUILTIN = 0,
        SOURCE_CUSTOM = 1,
    };

    enum PhaseResetMode {
        PHASE_FREE_RUN = 0,
        PHASE_RESET = 1,
        PHASE_RANDOMIZED = 2,
    };

    enum UnisonOutputMode {
        UNISON_OUTPUT_MONO_MIX = 0,
        UNISON_OUTPUT_STEREO_PAIRS = 1,
    };

    vivid::Param<int>   wavetable_source      {"wavetable_source", 0, {"Builtin", "Custom"}};
    vivid::Param<int>   wavetable_family      {"wavetable_family", 0, {"AnalogWarm", "BrightDigital", "VocalFormant", "Metallic", "HarmonicSpectral", "TextureMotion"}};
    vivid::Param<int>   wavetable_member      {"wavetable_member", 0, {"Core", "Soft", "Rich", "Hollow", "Sweep", "Glass", "Edge", "Air"}};
    vivid::Param<vivid::FilePath> wav_file    {"wav_file"};
    vivid::Param<float> position              {"position", 0.0f, 0.0f, 1.0f};
    vivid::Param<float> amplitude             {"amplitude", 0.3f, 0.0f, 1.0f};
    vivid::Param<int>   warp_mode             {"warp_mode", 0, {"None", "Sync", "BendPlus", "BendMinus", "Mirror", "Asym", "Quantize", "FM", "Flip"}};
    vivid::Param<float> warp_amount           {"warp_amount", 0.0f, 0.0f, 1.0f};
    vivid::Param<float> position_smooth_ms    {"position_smooth_ms", 8.0f, 0.0f, 40.0f};
    vivid::Param<float> warp_smooth_ms        {"warp_smooth_ms", 8.0f, 0.0f, 40.0f};
    vivid::Param<int>   phase_reset_mode      {"phase_reset_mode", 0, {"FreeRun", "Reset", "Randomized"}};
    vivid::Param<float> start_phase           {"start_phase", 0.0f, 0.0f, 1.0f};
    vivid::Param<float> phase_random          {"phase_random", 0.0f, 0.0f, 1.0f};
    vivid::Param<float> stereo_phase_offset   {"stereo_phase_offset", 0.25f, 0.0f, 1.0f};
    vivid::Param<float> drift_amount          {"drift_amount", 0.0f, 0.0f, 1.0f};
    vivid::Param<float> drift_rate_hz         {"drift_rate_hz", 0.18f, 0.02f, 2.0f};
    vivid::Param<int>   unison_voices         {"unison_voices", 1, 1, 16};
    vivid::Param<float> unison_spread         {"unison_spread", 20.0f, 0.0f, 100.0f};
    vivid::Param<float> unison_stereo         {"unison_stereo", 1.0f, 0.0f, 1.0f};
    vivid::Param<int>   unison_spread_mode    {"unison_spread_mode", 0, {"Linear", "Exponential", "Random"}};
    vivid::Param<int>   unison_output_mode    {"unison_output_mode", 0, {"MonoMix", "StereoPairs"}};
    vivid::Param<float> detune                {"detune", 0.0f, 0.0f, 50.0f};
    vivid::Param<float> portamento            {"portamento", 0.0f, 0.0f, 2000.0f};
    vivid::Param<int>   mod_type              {"mod_type", 0, {"Off", "FM", "RM", "AM"}};
    vivid::Param<float> mod_depth             {"mod_depth", 0.0f, 0.0f, 1.0f};

    std::atomic<Wavetable*> custom_table_{nullptr};
    Wavetable* deferred_delete_ = nullptr;
    std::string last_wav_path_;

    struct Voice {
        double phase[kMaxUnisonVoices] = {};
        double drift_phase[kMaxUnisonVoices] = {};
        float last_sample[kMaxUnisonVoices] = {};
        float current_freq = 0.0f;
        float target_freq = 0.0f;
        MotionSmoother pos_smoother;
        MotionSmoother warp_smoother;
        bool was_gated = false;
        bool initialized = false;
        int declick_remaining = 0;
    };

    WavetableOsc() {
        vivid::semantic_tag(position, "phase_01");
        vivid::semantic_intent(position, "wavetable_position");
        vivid::semantic_tag(amplitude, "amplitude_linear");
        vivid::semantic_tag(portamento, "time_milliseconds");
        vivid::semantic_unit(portamento, "ms");
        vivid::semantic_tag(position_smooth_ms, "time_milliseconds");
        vivid::semantic_unit(position_smooth_ms, "ms");
        vivid::semantic_tag(warp_smooth_ms, "time_milliseconds");
        vivid::semantic_unit(warp_smooth_ms, "ms");
        vivid::semantic_tag(start_phase, "phase_01");
        vivid::semantic_tag(stereo_phase_offset, "phase_01");
        vivid::semantic_tag(drift_rate_hz, "frequency_hz");
        vivid::semantic_unit(drift_rate_hz, "Hz");
    }

    ~WavetableOsc() {
        delete custom_table_.load(std::memory_order_relaxed);
        delete deferred_delete_;
    }

    void main_thread_update(double) override {
        if (deferred_delete_) {
            delete deferred_delete_;
            deferred_delete_ = nullptr;
        }
        if (wav_file.str_value != last_wav_path_) {
            last_wav_path_ = wav_file.str_value;
            if (last_wav_path_.empty()) {
                deferred_delete_ = custom_table_.exchange(nullptr, std::memory_order_release);
            } else {
                Wavetable* next = load_wavetable_from_wav(last_wav_path_);
                deferred_delete_ = custom_table_.exchange(next, std::memory_order_release);
            }
        }
    }

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        param_group(wavetable_source, "Core");
        param_group(wavetable_family, "Core");
        param_group(wavetable_member, "Core");
        param_group(wav_file, "Core");
        param_group(position, "Core");
        param_group(amplitude, "Core");
        param_group(warp_mode, "Warp");
        param_group(warp_amount, "Warp");
        param_group(position_smooth_ms, "Motion");
        param_group(warp_smooth_ms, "Motion");
        param_group(phase_reset_mode, "Phase");
        param_group(start_phase, "Phase");
        param_group(phase_random, "Phase");
        param_group(stereo_phase_offset, "Phase");
        param_group(drift_amount, "Motion");
        param_group(drift_rate_hz, "Motion");
        param_group(unison_voices, "Unison");
        param_group(unison_spread, "Unison");
        param_group(unison_stereo, "Unison");
        param_group(unison_spread_mode, "Unison");
        param_group(unison_output_mode, "Unison");
        param_group(detune, "Output");
        param_group(portamento, "Portamento");
        param_group(mod_type, "Modulation");
        param_group(mod_depth, "Modulation");

        out.push_back(&wavetable_source);
        out.push_back(&wavetable_family);
        out.push_back(&wavetable_member);
        out.push_back(&wav_file);
        out.push_back(&position);
        out.push_back(&amplitude);
        out.push_back(&warp_mode);
        out.push_back(&warp_amount);
        out.push_back(&position_smooth_ms);
        out.push_back(&warp_smooth_ms);
        out.push_back(&phase_reset_mode);
        out.push_back(&start_phase);
        out.push_back(&phase_random);
        out.push_back(&stereo_phase_offset);
        out.push_back(&drift_amount);
        out.push_back(&drift_rate_hz);
        out.push_back(&unison_voices);
        out.push_back(&unison_spread);
        out.push_back(&unison_stereo);
        out.push_back(&unison_spread_mode);
        out.push_back(&unison_output_mode);
        out.push_back(&detune);
        out.push_back(&portamento);
        out.push_back(&mod_type);
        out.push_back(&mod_depth);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back({"frequencies",  VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});
        out.push_back({"gates",        VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});
        out.push_back({"velocities",   VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});
        out.push_back({"pitch_mod",    VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});
        out.push_back({"position_mod", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});
        out.push_back({"warp_mod",     VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});
        out.push_back({"lane_ids",     VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});
        out.push_back({"mod_input", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                       VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});
        out.push_back({"pitch_mod_audio", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                       VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});
        out.push_back({"position_mod_audio", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                       VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});
        out.push_back({"warp_mod_audio", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                       VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});
        out.push_back({"output", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_OUTPUT,
                       VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, kMaxVoices});
    }

    static float cents_to_ratio(float cents) {
        return std::pow(2.0f, cents / 1200.0f);
    }

    static float read_lane(const VividLanePort* lane, int slot, float fallback = 0.0f) {
        if (lane && lane->data && slot >= 0 && static_cast<uint32_t>(slot) < lane->length)
            return lane->data[slot];
        return fallback;
    }

    static float* resolve_mod_channel(float* buf, uint32_t ch_count, uint32_t voice, uint32_t frames) {
        if (!buf || ch_count == 0) return nullptr;
        uint32_t ch = (voice < ch_count) ? voice : ch_count - 1;
        return buf + ch * frames;
    }

    static float wrap_phase(double phase) {
        phase -= std::floor(phase);
        return static_cast<float>(phase);
    }

    static float smoothing_coeff(float sample_rate, float smooth_ms) {
        if (smooth_ms <= 0.0f || sample_rate <= 0.0f) return 1.0f;
        float samples = smooth_ms * 0.001f * sample_rate;
        if (samples <= 1.0f) return 1.0f;
        return 1.0f - std::exp(-1.0f / samples);
    }

    static uint32_t hash_u32(uint32_t x) {
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        return x;
    }

    static float hash01(uint32_t seed) {
        return static_cast<float>(hash_u32(seed) & 0x00ffffffU) / static_cast<float>(0x01000000U);
    }

    static float normalized_unison_position(int index, int count) {
        if (count <= 1) return 0.0f;
        return (static_cast<float>(index) / static_cast<float>(count - 1)) * 2.0f - 1.0f;
    }

    static float unison_detune_offset(int index, int count, float spread_cents, int spread_mode, uint32_t lane_seed) {
        float linear = normalized_unison_position(index, count);
        switch (spread_mode) {
            case 1:
                linear = std::copysign(linear * linear, linear);
                break;
            case 2: {
                float mag = 0.35f + 0.65f * hash01(lane_seed + static_cast<uint32_t>(index * 17));
                linear *= mag;
                break;
            }
            default:
                break;
        }
        return linear * spread_cents;
    }

    static float unison_pan_position(int index, int count, float stereo_depth) {
        return normalized_unison_position(index, count) * stereo_depth;
    }

    static float base_phase_offset(int index, int count, bool stereo_pairs, float stereo_phase, uint32_t lane_seed) {
        float offset = 0.0f;
        if (stereo_pairs && count > 1 && stereo_phase > 0.0f) {
            offset += normalized_unison_position(index, count) * stereo_phase * 0.18f;
        }
        if (!stereo_pairs && count > 1 && stereo_phase > 0.0f) {
            offset += (hash01(lane_seed + static_cast<uint32_t>(index * 97)) - 0.5f) * stereo_phase * 0.12f;
        }
        return offset;
    }

    static float stereo_pair_phase_shift(int index, int count, float stereo_phase, uint32_t lane_seed) {
        if (count <= 1 || stereo_phase <= 0.0f) return 0.0f;
        float contour = 0.6f + 0.4f * std::abs(normalized_unison_position(index, count));
        float seeded = 0.04f + 0.08f * hash01(lane_seed + static_cast<uint32_t>(index * 173));
        return stereo_phase * contour * seeded;
    }

    static float gate_on_phase(PhaseResetMode mode,
                               float start_phase_value,
                               float phase_random_amount,
                               float base_offset,
                               int index,
                               uint32_t lane_seed) {
        float phase = start_phase_value + base_offset;
        if (mode == PHASE_RANDOMIZED) {
            float r = hash01(lane_seed + static_cast<uint32_t>(index * 131));
            phase += (r - 0.5f) * phase_random_amount;
        }
        phase -= std::floor(phase);
        return phase;
    }

    const Wavetable* resolve_table() const {
        const auto& builtins = builtin_tables();
        int source_mode = std::clamp(wavetable_source.int_value(),
                                     static_cast<int>(SOURCE_BUILTIN),
                                     static_cast<int>(SOURCE_CUSTOM));
        if (source_mode == SOURCE_CUSTOM) {
            const Wavetable* custom = custom_table_.load(std::memory_order_acquire);
            if (custom && custom->frame_count > 0) return custom;
        }
        const Wavetable* builtin = resolve_builtin_wavetable(builtins.data(), wavetable_family.int_value(), wavetable_member.int_value());
        return builtin ? builtin : &builtins[0];
    }

    static const std::array<Wavetable, kBuiltinWavetableCount>& builtin_tables() {
        static const std::array<Wavetable, kBuiltinWavetableCount> tables = []() {
            std::array<Wavetable, kBuiltinWavetableCount> built{};
            build_builtin_wavetables(built.data(), built.size());
            return built;
        }();
        return tables;
    }

    void process_audio(const VividAudioContext* ctx) override {
        uint32_t frames = ctx->buffer_size;
        float sr = static_cast<float>(ctx->sample_rate);

        const Wavetable& wt = *resolve_table();

        float pos = position.value;
        float amp = amplitude.value;
        int warp_m = warp_mode.int_value();
        float warp_a = warp_amount.value;
        float pos_smooth_coeff = smoothing_coeff(sr, position_smooth_ms.value);
        float warp_smooth_coeff = smoothing_coeff(sr, warp_smooth_ms.value);
        PhaseResetMode reset_mode = static_cast<PhaseResetMode>(std::clamp(phase_reset_mode.int_value(), 0, 2));
        float start_phase_value = std::clamp(start_phase.value, 0.0f, 1.0f);
        float phase_random_amount = std::clamp(phase_random.value, 0.0f, 1.0f);
        float stereo_phase = std::clamp(stereo_phase_offset.value, 0.0f, 1.0f);
        float drift_amt = std::clamp(drift_amount.value, 0.0f, 1.0f);
        float drift_rate = std::clamp(drift_rate_hz.value, 0.02f, 2.0f);
        int num_uni = std::clamp(unison_voices.int_value(), 1, kMaxUnisonVoices);
        float uni_spr = unison_spread.value;
        float uni_stereo = unison_stereo.value;
        int uni_spread_mode = unison_spread_mode.int_value();
        int uni_output_mode = std::clamp(unison_output_mode.int_value(), 0, 1);
        float det = detune.value;
        float porta_ms = portamento.value;
        int mtype = mod_type.int_value();
        float mdepth = mod_depth.value;
        bool stereo_pairs = (uni_output_mode == UNISON_OUTPUT_STEREO_PAIRS);

        const VividLanePort* freq_lane = ctx->input_lanes ? &ctx->input_lanes[0] : nullptr;
        const VividLanePort* gates_lane = ctx->input_lanes ? &ctx->input_lanes[1] : nullptr;
        const VividLanePort* pitch_lane = ctx->input_lanes ? &ctx->input_lanes[3] : nullptr;
        const VividLanePort* pos_mod_lane = ctx->input_lanes ? &ctx->input_lanes[4] : nullptr;
        const VividLanePort* warp_mod_lane = ctx->input_lanes ? &ctx->input_lanes[5] : nullptr;
        const VividLanePort* lane_id_lane = ctx->input_lanes ? &ctx->input_lanes[6] : nullptr;

        uint32_t voice_count = freq_lane ? freq_lane->length : 0;
        uint32_t max_note_voices = stereo_pairs ? static_cast<uint32_t>(kMaxStereoPairVoices)
                                                : static_cast<uint32_t>(kMaxVoices);
        if (voice_count > max_note_voices) voice_count = max_note_voices;

        float porta_rate = 1.0f;
        if (porta_ms > 0.0f) {
            float porta_samples = porta_ms * 0.001f * sr;
            porta_rate = 1.0f - std::exp(-4.0f / porta_samples);
        }

        float* mod_buf = (mtype > 0 && mdepth > 0.0f && ctx->input_buffers[7]) ? ctx->input_buffers[7] : nullptr;
        uint32_t mod_channels = mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[7] : 0;
        float* pitch_mod_buf = ctx->input_buffers[8];
        uint32_t pitch_mod_ch = pitch_mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[8] : 0;
        float* pos_mod_buf = ctx->input_buffers[9];
        uint32_t pos_mod_ch = pos_mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[9] : 0;
        float* warp_mod_buf = ctx->input_buffers[10];
        uint32_t warp_mod_ch = warp_mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[10] : 0;

        float* out_buf = ctx->output_buffers[0];
        std::memset(out_buf, 0, kMaxVoices * frames * sizeof(float));

        float unison_norm = 1.0f / std::sqrt(static_cast<float>(num_uni));

        for (uint32_t vi = 0; vi < voice_count; ++vi) {
            float gate = read_lane(gates_lane, static_cast<int>(vi));
            float freq_target = read_lane(freq_lane, static_cast<int>(vi));
            if (!std::isfinite(freq_target) || freq_target <= 0.0f) continue;

            uint32_t lid = lane_id_lane && lane_id_lane->data && vi < lane_id_lane->length
                ? static_cast<uint32_t>(lane_id_lane->data[vi]) : vi;
            Voice& v = *vivid_lane_state(ctx, lid, Voice);

            bool gate_on = gate > 0.5f;
            if (gate_on && !v.was_gated) {
                if (!v.initialized || reset_mode != PHASE_FREE_RUN) {
                    for (int ui = 0; ui < num_uni; ++ui) {
                        float phase = gate_on_phase(reset_mode,
                                                    start_phase_value,
                                                    phase_random_amount,
                                                    base_phase_offset(ui, num_uni, stereo_pairs, stereo_phase, lid),
                                                    ui,
                                                    lid);
                        if (reset_mode == PHASE_FREE_RUN && v.initialized) {
                            phase = wrap_phase(v.phase[ui]);
                        }
                        v.phase[ui] = phase;
                        v.last_sample[ui] = 0.0f;
                        v.drift_phase[ui] = hash01(lid + static_cast<uint32_t>(ui * 211)) * static_cast<float>(2.0 * M_PI);
                    }
                    float target_pos = std::clamp(pos + read_lane(pos_mod_lane, static_cast<int>(vi)), 0.0f, 1.0f);
                    float target_warp = std::clamp(warp_a + read_lane(warp_mod_lane, static_cast<int>(vi)), 0.0f, 1.0f);
                    v.pos_smoother.reset(target_pos);
                    v.warp_smoother.reset(target_warp);
                    v.declick_remaining = (reset_mode == PHASE_FREE_RUN) ? 0 : kDeClickSamples;
                }
                if (!v.initialized) {
                    v.current_freq = freq_target;
                    v.target_freq = freq_target;
                }
                v.initialized = true;
            }
            v.was_gated = gate_on;
            v.target_freq = freq_target;
            if (!std::isfinite(v.current_freq) || v.current_freq <= 0.0f)
                v.current_freq = freq_target;

            float* ch_out_l = out_buf + (stereo_pairs ? vi * 2 : vi) * frames;
            float* ch_out_r = stereo_pairs ? (out_buf + (vi * 2 + 1) * frames) : nullptr;

            float pitch_offset_lane = read_lane(pitch_lane, static_cast<int>(vi));
            float pos_mod_lane_val = read_lane(pos_mod_lane, static_cast<int>(vi));
            float warp_mod_lane_val = read_lane(warp_mod_lane, static_cast<int>(vi));

            float* mod_ch = resolve_mod_channel(mod_buf, mod_channels, vi, frames);
            float* pitch_mod_voice = resolve_mod_channel(pitch_mod_buf, pitch_mod_ch, vi, frames);
            float* pos_mod_voice = resolve_mod_channel(pos_mod_buf, pos_mod_ch, vi, frames);
            float* warp_mod_voice = resolve_mod_channel(warp_mod_buf, warp_mod_ch, vi, frames);

            for (uint32_t s = 0; s < frames; ++s) {
                if (porta_ms > 0.0f && v.current_freq != v.target_freq) {
                    v.current_freq += (v.target_freq - v.current_freq) * porta_rate;
                    if (std::abs(v.current_freq - v.target_freq) < 0.01f)
                        v.current_freq = v.target_freq;
                } else {
                    v.current_freq = v.target_freq;
                }

                float pitch_offset = pitch_mod_voice ? pitch_mod_voice[s] : pitch_offset_lane;
                float pos_mod_val = pos_mod_voice ? pos_mod_voice[s] : pos_mod_lane_val;
                float warp_mod_val = warp_mod_voice ? warp_mod_voice[s] : warp_mod_lane_val;

                float pitch_ratio = std::pow(2.0f, pitch_offset / 12.0f);
                float smooth_pos = v.pos_smoother.process(std::clamp(pos + pos_mod_val, 0.0f, 1.0f), pos_smooth_coeff);
                float smooth_warp = v.warp_smoother.process(std::clamp(warp_a + warp_mod_val, 0.0f, 1.0f), warp_smooth_coeff);

                float mono_sum = 0.0f;
                float stereo_l = 0.0f;
                float stereo_r = 0.0f;
                for (int ui = 0; ui < num_uni; ++ui) {
                    float detune_cents = det + unison_detune_offset(ui, num_uni, uni_spr, uni_spread_mode, lid);
                    float drift_seed = hash01(lid + static_cast<uint32_t>(ui * 379));
                    float drift_rate_scale = 0.7f + drift_seed * 0.8f;
                    float drift_cents = std::sin(v.drift_phase[ui]) * drift_amt * kMaxDriftCents;
                    v.drift_phase[ui] += static_cast<double>((2.0f * static_cast<float>(M_PI) * drift_rate * drift_rate_scale) / sr);
                    if (!std::isfinite(v.drift_phase[ui])) v.drift_phase[ui] = 0.0;
                    if (v.drift_phase[ui] > 2.0 * M_PI) v.drift_phase[ui] -= 2.0 * M_PI;

                    float base_freq = v.current_freq * pitch_ratio * cents_to_ratio(detune_cents + drift_cents);
                    if (!std::isfinite(base_freq) || base_freq <= 0.0f)
                        base_freq = std::max(v.current_freq, 1.0f);

                    float phase_inc = base_freq / sr;
                    if (mtype == 1 && mod_ch) {
                        phase_inc += mod_ch[s] * mdepth * 4.0f * (base_freq / sr);
                    }

                    float warped = warp_phase(wrap_phase(v.phase[ui]), warp_m, smooth_warp, v.last_sample[ui]);
                    float sig = wt.sample(warped, smooth_pos, base_freq, sr);
                    v.last_sample[ui] = sig;

                    float left_sig = sig;
                    float right_sig = sig;
                    if (stereo_pairs) {
                        float phase_shift = stereo_pair_phase_shift(ui, num_uni, stereo_phase, lid);
                        if (std::abs(phase_shift) > 0.00001f) {
                            float left_warped = warp_phase(wrap_phase(v.phase[ui] + phase_shift), warp_m, smooth_warp, v.last_sample[ui]);
                            float right_warped = warp_phase(wrap_phase(v.phase[ui] - phase_shift), warp_m, smooth_warp, v.last_sample[ui]);
                            left_sig = wt.sample(left_warped, smooth_pos, base_freq, sr);
                            right_sig = wt.sample(right_warped, smooth_pos, base_freq, sr);
                        }
                    }

                    if (mod_ch) {
                        if (mtype == 2) {
                            sig *= mod_ch[s];
                            left_sig *= mod_ch[s];
                            right_sig *= mod_ch[s];
                        } else if (mtype == 3) {
                            sig *= 1.0f + mdepth * mod_ch[s];
                            left_sig *= 1.0f + mdepth * mod_ch[s];
                            right_sig *= 1.0f + mdepth * mod_ch[s];
                        }
                    }

                    sig *= amp * unison_norm;
                    left_sig *= amp * unison_norm;
                    right_sig *= amp * unison_norm;
                    if (stereo_pairs) {
                        float pan = std::clamp(unison_pan_position(ui, num_uni, uni_stereo), -1.0f, 1.0f);
                        float theta = (pan + 1.0f) * static_cast<float>(M_PI) * 0.25f;
                        stereo_l += left_sig * std::cos(theta);
                        stereo_r += right_sig * std::sin(theta);
                    } else {
                        mono_sum += sig;
                    }

                    v.phase[ui] += static_cast<double>(phase_inc);
                    if (!std::isfinite(v.phase[ui])) {
                        v.phase[ui] = 0.0;
                    } else {
                        v.phase[ui] -= std::floor(v.phase[ui]);
                    }
                }

                float gate_gain = 1.0f;
                if (v.declick_remaining > 0) {
                    gate_gain = static_cast<float>(kDeClickSamples - v.declick_remaining + 1) / static_cast<float>(kDeClickSamples);
                    v.declick_remaining--;
                }

                if (stereo_pairs) {
                    ch_out_l[s] = stereo_l * gate_gain;
                    ch_out_r[s] = stereo_r * gate_gain;
                } else {
                    ch_out_l[s] = mono_sum * gate_gain;
                }
            }
        }
    }
};

VIVID_REGISTER(WavetableOsc)
