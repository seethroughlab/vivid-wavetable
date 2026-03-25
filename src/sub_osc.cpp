#include "operator_api/operator.h"
#include "operator_api/audio_operator.h"
#include "operator_api/audio_dsp.h"
#include "operator_api/type_id.h"
#include <cmath>
#include <cstring>
#include <algorithm>

// =============================================================================
// SubOsc — polyphonic sub oscillator, outputs N-channel per-voice audio
// =============================================================================

struct SubOsc : vivid::AudioOperatorBase {
    static constexpr const char* kName   = "SubOsc";
    static constexpr bool kTimeDependent = true;

    static constexpr int kMaxVoices = 16;

    vivid::Param<float> level    {"level",    0.5f, 0.0f, 1.0f};
    vivid::Param<int>   octave   {"octave",   0,    {"-1", "-2"}};
    vivid::Param<int>   waveform {"waveform", 0,    {"Sine", "Triangle", "Saw", "Square", "Noise"}};

    struct Voice {
        double phase     = 0;
        bool   was_gated = false;
        audio_dsp::WhiteNoise white_noise;
    };
    Voice voices_[kMaxVoices] = {};

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        param_group(level,    "Sub");
        param_group(octave,   "Sub");
        param_group(waveform, "Sub");

        out.push_back(&level);
        out.push_back(&octave);
        out.push_back(&waveform);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back({"frequencies", VIVID_PORT_SPREAD, VIVID_PORT_INPUT});
        out.push_back({"gates",       VIVID_PORT_SPREAD, VIVID_PORT_INPUT});
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
        float sr = static_cast<float>(ctx->sample_rate);

        float lvl     = level.value;
        float sub_div = (octave.int_value() == 1) ? 4.0f : 2.0f;
        int   wave    = waveform.int_value();

        const VividSpreadPort* freq_sp  = ctx->input_spreads ? &ctx->input_spreads[0] : nullptr;
        const VividSpreadPort* gates_sp = ctx->input_spreads ? &ctx->input_spreads[1] : nullptr;

        uint32_t voice_count = freq_sp ? freq_sp->length : 0;
        if (voice_count > kMaxVoices) voice_count = kMaxVoices;

        float* out_buf = ctx->output_buffers[0];
        std::memset(out_buf, 0, kMaxVoices * frames * sizeof(float));

        // Waveform mapping: param order (Sine=0, Tri=1, Saw=2, Sq=3)
        // to audio_dsp::waveform order (sine=0, saw=1, sq=2, tri=3)
        static constexpr int wf_map[] = {0, 3, 1, 2};

        for (uint32_t vi = 0; vi < voice_count; ++vi) {
            float gate = read_spread(gates_sp, vi);
            float freq = read_spread(freq_sp, vi);
            if (freq <= 0.0f || gate <= 0.5f) continue;

            Voice& v = voices_[vi];

            bool gate_on = (gate > 0.5f);
            if (gate_on && !v.was_gated) {
                v.phase = 0.0;
                v.white_noise.state = 12345u + static_cast<uint32_t>(vi) * 1664525u;
            }
            v.was_gated = gate_on;

            float sub_freq = freq / sub_div;
            float sub_inc  = sub_freq / sr;
            float* ch_out  = out_buf + vi * frames;

            for (uint32_t s = 0; s < frames; ++s) {
                float sig;
                if (wave == 4) {
                    sig = v.white_noise.next();
                } else {
                    sig = static_cast<float>(audio_dsp::waveform(v.phase, wf_map[wave]));
                }
                ch_out[s] = sig * lvl;

                v.phase += static_cast<double>(sub_inc);
                if (v.phase >= 1.0) v.phase -= 1.0;
                if (!std::isfinite(v.phase)) v.phase = 0.0;
            }
        }
    }
};

VIVID_REGISTER(SubOsc)
