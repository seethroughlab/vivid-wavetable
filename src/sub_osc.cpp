#include "operator_api/operator.h"
#include "operator_api/audio_dsp.h"
#include "operator_api/type_id.h"
#include <cmath>
#include <cstring>
#include <algorithm>

// =============================================================================
// SubOsc — polyphonic sub oscillator, outputs N-channel per-voice audio
// =============================================================================

struct SubOsc : vivid::OperatorBase, vivid::AudioProcessable {
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

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        param_group(level,    "Sub");
        param_group(octave,   "Sub");
        param_group(waveform, "Sub");

        out.push_back(&level);
        out.push_back(&octave);
        out.push_back(&waveform);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back({"frequencies", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});    // 0
        out.push_back({"gates",       VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});    // 1
        out.push_back({"lane_ids",    VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});    // 2 (identity tokens)
        // Audio-rate pitch modulation (N-channel, one per voice, semitones)
        out.push_back({"pitch_mod_audio", VIVID_PORT_AUDIO, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});     // 3
        out.push_back({"output", VIVID_PORT_AUDIO, VIVID_PORT_OUTPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, kMaxVoices});
    }

    static float* resolve_mod_channel(float* buf, uint32_t ch_count, uint32_t voice, uint32_t frames) {
        if (!buf || ch_count == 0) return nullptr;
        uint32_t ch = (voice < ch_count) ? voice : ch_count - 1;
        return buf + ch * frames;
    }

    static float read_lane(const VividLanePort* sp, int slot, float fallback = 0.0f) {
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

        const VividLanePort* freq_lane    = ctx->input_lanes ? &ctx->input_lanes[0] : nullptr;
        const VividLanePort* gates_lane   = ctx->input_lanes ? &ctx->input_lanes[1] : nullptr;
        const VividLanePort* lane_id_lane = ctx->input_lanes ? &ctx->input_lanes[2] : nullptr;

        uint32_t voice_count = freq_lane ? freq_lane->length : 0;
        if (voice_count > static_cast<uint32_t>(kMaxVoices)) voice_count = kMaxVoices;

        // Audio-rate pitch modulation (port 3: after 3 lane ports)
        float* pitch_mod_buf = ctx->input_buffers[3];
        uint32_t pitch_mod_ch = pitch_mod_buf && ctx->input_channel_counts
                                ? ctx->input_channel_counts[3] : 0;

        float* out_buf = ctx->output_buffers[0];
        std::memset(out_buf, 0, kMaxVoices * frames * sizeof(float));

        // Waveform mapping: param order (Sine=0, Tri=1, Saw=2, Sq=3)
        // to audio_dsp::waveform order (sine=0, saw=1, sq=2, tri=3)
        static constexpr int wf_map[] = {0, 3, 1, 2};

        for (uint32_t vi = 0; vi < voice_count; ++vi) {
            float gate = read_lane(gates_lane, vi);
            float freq = read_lane(freq_lane, vi);
            if (freq <= 0.0f) continue;
            // Don't skip gate=0 voices — releasing voices need audio for
            // downstream envelope release tails.

            uint32_t lid = lane_id_lane && lane_id_lane->data && vi < lane_id_lane->length
                ? static_cast<uint32_t>(lane_id_lane->data[vi]) : vi;
            Voice& v = *vivid_lane_state(ctx, lid, Voice);

            bool gate_on = (gate > 0.5f);
            if (gate_on && !v.was_gated) {
                v.phase = 0.0;
                v.white_noise.state = 12345u + static_cast<uint32_t>(vi) * 1664525u;
            }
            v.was_gated = gate_on;

            float base_sub_freq = freq / sub_div;
            float base_sub_inc  = base_sub_freq / sr;
            float* ch_out  = out_buf + vi * frames;
            float* pitch_mod_voice = resolve_mod_channel(pitch_mod_buf, pitch_mod_ch, vi, frames);

            for (uint32_t s = 0; s < frames; ++s) {
                float sig;
                if (wave == 4) {
                    sig = v.white_noise.next();
                } else {
                    sig = static_cast<float>(audio_dsp::waveform(v.phase, wf_map[wave]));
                }
                ch_out[s] = sig * lvl;

                float sub_inc = base_sub_inc;
                if (pitch_mod_voice) {
                    sub_inc *= std::pow(2.0f, pitch_mod_voice[s] / 12.0f);
                }
                v.phase += static_cast<double>(sub_inc);
                if (v.phase >= 1.0) v.phase -= 1.0;
                if (!std::isfinite(v.phase)) v.phase = 0.0;
            }
        }
    }
};

VIVID_REGISTER(SubOsc)
