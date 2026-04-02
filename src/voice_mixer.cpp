#include "operator_api/operator.h"
#include "operator_api/type_id.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr float PI_F = static_cast<float>(M_PI);

// =============================================================================
// VoiceMixer — sums N-channel per-voice audio to stereo with panning & envelope
// =============================================================================

/**
 * @brief Reduces per-voice audio channels to stereo with envelope, pan, and velocity shaping.
 *
 * Accepts either mono-per-voice or stereo-pair voice layouts and combines lane-based
 * and audio-rate modulation inputs for amplitude and pan shaping before summing to
 * the final stereo output.
 *
 * @input input Per-voice audio channels from oscillators or layered voice chains.
 * @input amp_env_audio Audio-rate per-voice envelope input, typically from EnvelopeAu/value.
 * @input pan_mod_audio Audio-rate per-voice pan modulation.
 * @input amp_env Lane-array fallback envelope values when amp_env_audio is not connected.
 * @input velocities Per-voice velocities used for velocity-to-volume scaling.
 * @input pan_mod Lane-array fallback pan modulation values.
 * @output output Final stereo mix.
 * @tip Drive amp_env_audio from EnvelopeAu/value when you want true per-note amplitude shaping.
 * @recipe WavetableOsc/output -> VoiceMixer/input
 * @recipe EnvelopeAu/value -> VoiceMixer/amp_env_audio
 * @pitfall VoiceMixer is the reduction stage in the poly chain; once audio is summed here, downstream operators no longer see separate note lanes.
 * @family voice_mixer
 * @best_used_with EnvelopeAu, PolyVoiceAllocator, Filter
 * @common_companions WavetableOsc, AnalogOsc, SubOsc
 */
struct VoiceMixer : vivid::OperatorBase, vivid::AudioProcessable {
    static constexpr const char* kName   = "VoiceMixer";
    static constexpr bool kTimeDependent = true;
    static constexpr VividLaneBehavior kLaneBehavior = VIVID_LANE_REDUCTION;

    static constexpr int kMaxChannels = 32;

    enum InputLayout {
        INPUT_LAYOUT_MONO_VOICES = 0,
        INPUT_LAYOUT_STEREO_PAIRS = 1,
    };

    // --- Parameters ---
    vivid::Param<int> input_layout  {"input_layout", 0, {"MonoVoices", "StereoPairs"}};
    vivid::Param<float> stereo_spread {"stereo_spread", 0.5f,  0.0f, 1.0f};
    vivid::Param<float> vel_to_volume {"vel_to_volume", 1.0f,  0.0f, 1.0f};

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        param_group(input_layout, "Routing");
        param_group(stereo_spread, "Output");
        param_group(vel_to_volume, "Velocity");

        out.push_back(&input_layout);
        out.push_back(&stereo_spread);
        out.push_back(&vel_to_volume);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        // N-channel audio input (port 0)
        out.push_back({"input", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0}); // auto channels
        // Audio-rate envelope input (N-channel, one per voice) (port 1)
        out.push_back({"amp_env_audio", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});
        // Audio-rate pan modulation input (N-channel, one per voice) (port 2)
        out.push_back({"pan_mod_audio", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});
        // Lane inputs for per-voice control (ports 3-5)
        out.push_back({"amp_env",    VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 3
        out.push_back({"velocities", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 4
        out.push_back({"pan_mod",    VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 5
        // Stereo output
        out.push_back({"output", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_OUTPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 2}); // stereo
    }

    static float read_lane(const VividLanePort* sp, int slot, float fallback = 0.0f) {
        if (sp && sp->data && slot >= 0 && static_cast<uint32_t>(slot) < sp->length)
            return sp->data[slot];
        return fallback;
    }

    static float* resolve_mod_channel(float* buf, uint32_t ch_count, uint32_t voice, uint32_t frames) {
        if (!buf || ch_count == 0) return nullptr;
        uint32_t ch = (voice < ch_count) ? voice : ch_count - 1;
        return buf + ch * frames;
    }

    // Check if an audio buffer has any non-zero samples (detects unconnected ports
    // whose buffers are allocated but zero-filled by the runtime).
    static bool buffer_has_signal(const float* buf, uint32_t len) {
        if (!buf) return false;
        uint32_t check = std::min(len, 16u);
        for (uint32_t i = 0; i < check; ++i)
            if (buf[i] != 0.0f) return true;
        return false;
    }

    void process_audio(const VividAudioContext* ctx) override {
        uint32_t frames = ctx->buffer_size;

        // Port layout: [0] input, [1] amp_env_audio, [2] pan_mod_audio, [3-5] lanes
        uint32_t num_ch = ctx->input_channel_counts ? ctx->input_channel_counts[0] : 2;
        if (num_ch > kMaxChannels) num_ch = kMaxChannels;

        int layout = input_layout.int_value();
        float spread = stereo_spread.value;
        float v2vol  = vel_to_volume.value;

        const VividLanePort* env_lane = ctx->input_lanes ? &ctx->input_lanes[3] : nullptr;
        const VividLanePort* vel_lane = ctx->input_lanes ? &ctx->input_lanes[4] : nullptr;
        const VividLanePort* pan_lane = ctx->input_lanes ? &ctx->input_lanes[5] : nullptr;

        // Audio-rate modulation buffers (ports 1, 2)
        float* env_audio_buf = ctx->input_buffers[1];
        uint32_t env_audio_ch = (env_audio_buf && ctx->input_channel_counts
                                 && buffer_has_signal(env_audio_buf, frames))
                                ? ctx->input_channel_counts[1] : 0;
        float* pan_audio_buf = ctx->input_buffers[2];
        uint32_t pan_audio_ch = (pan_audio_buf && ctx->input_channel_counts
                                 && buffer_has_signal(pan_audio_buf, frames))
                                ? ctx->input_channel_counts[2] : 0;

        float* in_buf = ctx->input_buffers[0];
        float* out_l  = ctx->output_buffers[0];
        float* out_r  = ctx->output_buffers[0] + frames;

        std::memset(out_l, 0, frames * sizeof(float));
        std::memset(out_r, 0, frames * sizeof(float));

        if (layout == INPUT_LAYOUT_STEREO_PAIRS) {
            uint32_t pair_count = num_ch / 2;
            if (env_lane && env_lane->length > 0)
                pair_count = std::min(pair_count, env_lane->length);
            float norm = (pair_count > 0) ? 1.0f / std::sqrt(static_cast<float>(pair_count)) : 1.0f;

            for (uint32_t pair = 0; pair < pair_count; ++pair) {
                float* in_l = in_buf + (pair * 2) * frames;
                float* in_r = in_buf + (pair * 2 + 1) * frames;

                float vel = read_lane(vel_lane, pair, 1.0f);
                float vel_vol = 1.0f - v2vol * (1.0f - vel);

                float* env_voice = resolve_mod_channel(env_audio_buf, env_audio_ch, pair, frames);
                float* pan_voice = resolve_mod_channel(pan_audio_buf, pan_audio_ch, pair, frames);

                float base_pan = 0.0f;
                if (pair_count > 1) {
                    base_pan = (static_cast<float>(pair) / static_cast<float>(pair_count - 1)
                                * 2.0f - 1.0f) * spread;
                }

                if (env_voice || pan_voice) {
                    float lane_env = read_lane(env_lane, pair, 1.0f);
                    float lane_pan_mod = read_lane(pan_lane, pair);

                    for (uint32_t s = 0; s < frames; ++s) {
                        float env_val = env_voice ? env_voice[s] : lane_env;
                        float gain = env_val * vel_vol * norm;

                        float pan_mod = pan_voice ? pan_voice[s] : lane_pan_mod;
                        float pan = std::clamp(base_pan + pan_mod, -1.0f, 1.0f);
                        float theta = (pan + 1.0f) * PI_F * 0.25f;
                        float gl = std::cos(theta) * gain;
                        float gr = std::sin(theta) * gain;

                        out_l[s] += in_l[s] * gl;
                        out_r[s] += in_r[s] * gr;
                    }
                } else {
                    float env_val = read_lane(env_lane, pair, 1.0f);
                    float gain = env_val * vel_vol * norm;
                    if (gain < 0.0001f) continue;

                    float pan_mod = read_lane(pan_lane, pair);
                    float pan = std::clamp(base_pan + pan_mod, -1.0f, 1.0f);
                    float theta = (pan + 1.0f) * PI_F * 0.25f;
                    float gl = std::cos(theta) * gain;
                    float gr = std::sin(theta) * gain;

                    for (uint32_t s = 0; s < frames; ++s) {
                        out_l[s] += in_l[s] * gl;
                        out_r[s] += in_r[s] * gr;
                    }
                }
            }
            return;
        }

        uint32_t active_channels = num_ch;
        if (env_lane && env_lane->length > 0)
            active_channels = std::min(active_channels, env_lane->length);
        float norm = (active_channels > 0) ? 1.0f / std::sqrt(static_cast<float>(active_channels)) : 1.0f;

        for (uint32_t ch = 0; ch < active_channels; ++ch) {
            float* ch_in = in_buf + ch * frames;

            float vel = read_lane(vel_lane, ch, 1.0f);
            float vel_vol = 1.0f - v2vol * (1.0f - vel);

            float* env_voice = resolve_mod_channel(env_audio_buf, env_audio_ch, ch, frames);
            float* pan_voice = resolve_mod_channel(pan_audio_buf, pan_audio_ch, ch, frames);

            float base_pan = 0.0f;
            if (active_channels > 1) {
                base_pan = (static_cast<float>(ch) / static_cast<float>(active_channels - 1)
                            * 2.0f - 1.0f) * spread;
            }

            if (env_voice || pan_voice) {
                float lane_env = read_lane(env_lane, ch, 1.0f);
                float lane_pan_mod = read_lane(pan_lane, ch);

                for (uint32_t s = 0; s < frames; ++s) {
                    float env_val = env_voice ? env_voice[s] : lane_env;
                    float gain = env_val * vel_vol * norm;

                    float pan_mod = pan_voice ? pan_voice[s] : lane_pan_mod;
                    float pan = std::clamp(base_pan + pan_mod, -1.0f, 1.0f);
                    float theta = (pan + 1.0f) * PI_F * 0.25f;

                    float sig = ch_in[s];
                    out_l[s] += sig * std::cos(theta) * gain;
                    out_r[s] += sig * std::sin(theta) * gain;
                }
            } else {
                float env_val = read_lane(env_lane, ch, 1.0f);
                float gain = env_val * vel_vol * norm;
                if (gain < 0.0001f) continue;

                float pan_mod = read_lane(pan_lane, ch);
                float pan = std::clamp(base_pan + pan_mod, -1.0f, 1.0f);
                float theta = (pan + 1.0f) * PI_F * 0.25f;
                float gl = std::cos(theta) * gain;
                float gr = std::sin(theta) * gain;

                for (uint32_t s = 0; s < frames; ++s) {
                    float sig = ch_in[s];
                    out_l[s] += sig * gl;
                    out_r[s] += sig * gr;
                }
            }
        }
    }
};

VIVID_REGISTER(VoiceMixer)
