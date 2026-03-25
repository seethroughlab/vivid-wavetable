#include "operator_api/operator.h"
#include "operator_api/audio_operator.h"
#include "operator_api/audio_dsp.h"
#include "operator_api/type_id.h"
#include <cmath>
#include <cstring>
#include <algorithm>

// =============================================================================
// NoiseOsc — per-voice noise generator, outputs N-channel audio
// =============================================================================

struct NoiseOsc : vivid::AudioOperatorBase {
    static constexpr const char* kName   = "NoiseOsc";
    static constexpr bool kTimeDependent = true;

    static constexpr int kMaxVoices = 16;

    vivid::Param<float> level      {"level",      0.5f, 0.0f, 1.0f};
    vivid::Param<int>   noise_type {"noise_type", 0,    {"White", "Pink"}};

    struct Voice {
        audio_dsp::WhiteNoise white_noise;
        audio_dsp::PinkNoise  pink_noise;
        bool was_gated = false;
    };
    Voice voices_[kMaxVoices] = {};

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        param_group(level,      "Noise");
        param_group(noise_type, "Noise");

        out.push_back(&level);
        out.push_back(&noise_type);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back({"gates", VIVID_PORT_SPREAD, VIVID_PORT_INPUT});
        out.push_back({"output", VIVID_PORT_AUDIO, VIVID_PORT_OUTPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, kMaxVoices});
    }

    static float read_spread(const VividSpreadPort* sp, int slot, float fallback = 0.0f) {
        if (sp && sp->data && slot >= 0 && static_cast<uint32_t>(slot) < sp->length)
            return sp->data[slot];
        return fallback;
    }

    void process_audio(const VividAudioContext* ctx) override {
        uint32_t frames = ctx->buffer_size;

        float lvl  = level.value;
        int   ntype = noise_type.int_value();

        const VividSpreadPort* gates_sp = ctx->input_spreads ? &ctx->input_spreads[0] : nullptr;

        uint32_t voice_count = gates_sp ? gates_sp->length : 0;
        if (voice_count > kMaxVoices) voice_count = kMaxVoices;

        float* out_buf = ctx->output_buffers[0];
        std::memset(out_buf, 0, kMaxVoices * frames * sizeof(float));

        for (uint32_t vi = 0; vi < voice_count; ++vi) {
            float gate = read_spread(gates_sp, vi);
            if (gate <= 0.5f) continue;

            Voice& v = voices_[vi];
            if (!v.was_gated) {
                v.white_noise.state = 67890u + static_cast<uint32_t>(vi) * 1664525u;
                v.pink_noise = {};
                v.pink_noise.white.state = 12345u + static_cast<uint32_t>(vi) * 1664525u;
            }
            v.was_gated = (gate > 0.5f);

            float* ch_out = out_buf + vi * frames;
            for (uint32_t s = 0; s < frames; ++s) {
                float n = (ntype == 0) ? v.white_noise.next() : v.pink_noise.next();
                ch_out[s] = n * lvl;
            }
        }
    }
};

VIVID_REGISTER(NoiseOsc)
