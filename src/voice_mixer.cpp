#include "operator_api/operator.h"
#include "operator_api/audio_operator.h"
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

struct VoiceMixer : vivid::AudioOperatorBase {
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
        // N-channel audio input
        out.push_back({"input", VIVID_PORT_AUDIO, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0}); // auto channels
        // Spread inputs for per-voice control
        out.push_back({"amp_env",    VIVID_PORT_SPREAD, VIVID_PORT_INPUT});  // 0
        out.push_back({"velocities", VIVID_PORT_SPREAD, VIVID_PORT_INPUT});  // 1
        out.push_back({"pan_mod",    VIVID_PORT_SPREAD, VIVID_PORT_INPUT});  // 2
        // Stereo output
        out.push_back({"output", VIVID_PORT_AUDIO, VIVID_PORT_OUTPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 2}); // stereo
    }

    static float read_spread(const VividSpreadPort* sp, int slot, float fallback = 0.0f) {
        if (sp && sp->data && slot >= 0 && static_cast<uint32_t>(slot) < sp->length)
            return sp->data[slot];
        return fallback;
    }

    void process_audio(const VividAudioContext* ctx) override {
        uint32_t frames = ctx->buffer_size;

        uint32_t num_ch = ctx->input_channel_counts ? ctx->input_channel_counts[0] : 2;
        if (num_ch > kMaxChannels) num_ch = kMaxChannels;

        float spread = stereo_spread.value;
        float v2vol  = vel_to_volume.value;

        const VividSpreadPort* env_sp = ctx->input_spreads ? &ctx->input_spreads[0] : nullptr;
        const VividSpreadPort* vel_sp = ctx->input_spreads ? &ctx->input_spreads[1] : nullptr;
        const VividSpreadPort* pan_sp = ctx->input_spreads ? &ctx->input_spreads[2] : nullptr;

        float* in_buf = ctx->input_buffers[0];
        float* out_l  = ctx->output_buffers[0];
        float* out_r  = ctx->output_buffers[0] + frames;

        std::memset(out_l, 0, frames * sizeof(float));
        std::memset(out_r, 0, frames * sizeof(float));

        float norm = 1.0f / std::sqrt(static_cast<float>(kMaxChannels));

        for (uint32_t ch = 0; ch < num_ch; ++ch) {
            float* ch_in = in_buf + ch * frames;

            // Per-voice amplitude from envelope
            float env_val = read_spread(env_sp, ch, 1.0f);

            // Velocity scaling
            float vel = read_spread(vel_sp, ch, 1.0f);
            float vel_vol = 1.0f - v2vol * (1.0f - vel);

            float gain = env_val * vel_vol * norm;
            if (gain < 0.0001f) continue;

            // Panning: distribute voices across stereo field
            float pan = 0.0f;
            if (num_ch > 1) {
                pan = (static_cast<float>(ch) / static_cast<float>(num_ch - 1)
                       * 2.0f - 1.0f) * spread;
            }
            // Add per-voice pan modulation
            float pan_mod = read_spread(pan_sp, ch);
            pan = std::clamp(pan + pan_mod, -1.0f, 1.0f);

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
};

VIVID_REGISTER(VoiceMixer)
