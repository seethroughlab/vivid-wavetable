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

struct VoiceMixer : vivid::OperatorBase, vivid::AudioProcessable {
    static constexpr const char* kName   = "VoiceMixer";
    static constexpr bool kTimeDependent = true;

    static constexpr int kMaxChannels = 16;

    // --- Parameters ---
    vivid::Param<float> stereo_spread {"stereo_spread", 0.5f,  0.0f, 1.0f};
    vivid::Param<float> vel_to_volume {"vel_to_volume", 1.0f,  0.0f, 1.0f};

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        param_group(stereo_spread, "Output");
        param_group(vel_to_volume, "Velocity");

        out.push_back(&stereo_spread);
        out.push_back(&vel_to_volume);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        // N-channel audio input (port 0)
        out.push_back({"input", VIVID_PORT_AUDIO, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0}); // auto channels
        // Audio-rate envelope input (N-channel, one per voice) (port 1)
        out.push_back({"amp_env_audio", VIVID_PORT_AUDIO, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});
        // Audio-rate pan modulation input (N-channel, one per voice) (port 2)
        out.push_back({"pan_mod_audio", VIVID_PORT_AUDIO, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});
        // Spread inputs for per-voice control (ports 3-5)
        out.push_back({"amp_env",    VIVID_PORT_SPREAD, VIVID_PORT_INPUT});  // 3
        out.push_back({"velocities", VIVID_PORT_SPREAD, VIVID_PORT_INPUT});  // 4
        out.push_back({"pan_mod",    VIVID_PORT_SPREAD, VIVID_PORT_INPUT});  // 5
        // Stereo output
        out.push_back({"output", VIVID_PORT_AUDIO, VIVID_PORT_OUTPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 2}); // stereo
    }

    static float read_spread(const VividSpreadPort* sp, int slot, float fallback = 0.0f) {
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

        // Port layout: [0] input, [1] amp_env_audio, [2] pan_mod_audio, [3-5] spreads
        uint32_t num_ch = ctx->input_channel_counts ? ctx->input_channel_counts[0] : 2;
        if (num_ch > kMaxChannels) num_ch = kMaxChannels;

        float spread = stereo_spread.value;
        float v2vol  = vel_to_volume.value;

        const VividSpreadPort* env_sp = ctx->input_spreads ? &ctx->input_spreads[3] : nullptr;
        const VividSpreadPort* vel_sp = ctx->input_spreads ? &ctx->input_spreads[4] : nullptr;
        const VividSpreadPort* pan_sp = ctx->input_spreads ? &ctx->input_spreads[5] : nullptr;

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

        // Normalize by active voice count (from envelope spread), not total channel count
        uint32_t active_voices = (env_sp && env_sp->length > 0) ? env_sp->length : num_ch;
        float norm = (active_voices > 0) ? 1.0f / std::sqrt(static_cast<float>(active_voices)) : 1.0f;

        for (uint32_t ch = 0; ch < num_ch; ++ch) {
            float* ch_in = in_buf + ch * frames;

            // Velocity scaling (always from spread — velocity doesn't change per-sample)
            float vel = read_spread(vel_sp, ch, 1.0f);
            float vel_vol = 1.0f - v2vol * (1.0f - vel);

            // Audio-rate modulation channels (1-ch broadcasts to all voices)
            float* env_voice = resolve_mod_channel(env_audio_buf, env_audio_ch, ch, frames);
            float* pan_voice = resolve_mod_channel(pan_audio_buf, pan_audio_ch, ch, frames);

            // Base pan position from stereo spread
            float base_pan = 0.0f;
            if (num_ch > 1) {
                base_pan = (static_cast<float>(ch) / static_cast<float>(num_ch - 1)
                            * 2.0f - 1.0f) * spread;
            }

            if (env_voice || pan_voice) {
                // Per-sample path: audio-rate envelope and/or pan modulation
                float sp_env = read_spread(env_sp, ch, 1.0f);
                float sp_pan_mod = read_spread(pan_sp, ch);

                for (uint32_t s = 0; s < frames; ++s) {
                    float env_val = env_voice ? env_voice[s] : sp_env;
                    float gain = env_val * vel_vol * norm;

                    float pan_mod = pan_voice ? pan_voice[s] : sp_pan_mod;
                    float pan = std::clamp(base_pan + pan_mod, -1.0f, 1.0f);
                    float theta = (pan + 1.0f) * PI_F * 0.25f;

                    float sig = ch_in[s];
                    out_l[s] += sig * std::cos(theta) * gain;
                    out_r[s] += sig * std::sin(theta) * gain;
                }
            } else {
                // Block-rate path: spread-only (original behavior, more efficient)
                float env_val = read_spread(env_sp, ch, 1.0f);
                float gain = env_val * vel_vol * norm;
                if (gain < 0.0001f) continue;

                float pan_mod = read_spread(pan_sp, ch);
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
