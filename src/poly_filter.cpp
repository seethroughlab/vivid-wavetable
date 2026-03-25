#include "operator_api/operator.h"
#include "operator_api/audio_operator.h"
#include "operator_api/type_id.h"
#include "wavetable_dsp.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr float TWO_PI_F = 2.0f * static_cast<float>(M_PI);

using namespace vivid_wavetable::dsp;

// =============================================================================
// PolyFilter — per-channel (per-voice) filter on N-channel audio
// =============================================================================

struct PolyFilter : vivid::AudioOperatorBase {
    static constexpr const char* kName   = "PolyFilter";
    static constexpr bool kTimeDependent = true;

    static constexpr int kMaxChannels = 16;

    // --- Parameters ---
    vivid::Param<int>   filter_type      {"filter_type",      1,        {"LP12", "LP24", "HP12", "BP", "Notch", "Comb", "Ladder", "Formant"}};
    vivid::Param<float> filter_cutoff    {"filter_cutoff",    20000.0f, 20.0f,  20000.0f};
    vivid::Param<float> filter_resonance {"filter_resonance", 0.0f,     0.0f,   1.0f};
    vivid::Param<float> filter_keytrack  {"filter_keytrack",  0.0f,     0.0f,   1.0f};
    vivid::Param<float> filter_drive     {"filter_drive",     0.0f,     0.0f,   1.0f};

    // --- Per-channel filter state ---
    struct ChannelState {
        float fz1[2] = {};
        float fz2[2] = {};
        CombFilterState    comb;
        LadderFilterState  ladder;
        FormantFilterState formant;
        bool  prev_gated = false;

        void reset() {
            fz1[0] = fz1[1] = 0.0f;
            fz2[0] = fz2[1] = 0.0f;
            comb.reset();
            ladder.reset();
            formant.reset();
        }
    };
    ChannelState channels_[kMaxChannels] = {};

    PolyFilter() {
        vivid::semantic_tag(filter_cutoff, "frequency_hz");
        vivid::semantic_unit(filter_cutoff, "Hz");
        vivid::semantic_tag(filter_resonance, "resonance");
    }

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        param_group(filter_type,      "Filter");
        param_group(filter_cutoff,    "Filter");
        param_group(filter_resonance, "Filter");
        param_group(filter_keytrack,  "Filter");
        param_group(filter_drive,     "Filter");

        display_hint(filter_cutoff,    VIVID_DISPLAY_KNOB);
        display_hint(filter_resonance, VIVID_DISPLAY_KNOB);
        display_hint(filter_keytrack,  VIVID_DISPLAY_KNOB);
        display_hint(filter_drive,     VIVID_DISPLAY_KNOB);

        layout_row(filter_cutoff,    4, 0);
        layout_row(filter_resonance, 4, 1);
        layout_row(filter_keytrack,  4, 2);
        layout_row(filter_drive,     4, 3);

        out.push_back(&filter_type);
        out.push_back(&filter_cutoff);
        out.push_back(&filter_resonance);
        out.push_back(&filter_keytrack);
        out.push_back(&filter_drive);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        // Audio in/out: channel count auto-propagated from upstream
        out.push_back({"input",  VIVID_PORT_AUDIO, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0}); // 0=auto channels
        out.push_back({"frequencies", VIVID_PORT_SPREAD, VIVID_PORT_INPUT});  // for keytracking
        out.push_back({"gates",       VIVID_PORT_SPREAD, VIVID_PORT_INPUT});  // reset on gate-on
        out.push_back({"cutoff_mod",  VIVID_PORT_SPREAD, VIVID_PORT_INPUT});  // per-voice cutoff mod
        out.push_back({"output", VIVID_PORT_AUDIO, VIVID_PORT_OUTPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0}); // 0=auto channels
    }

    static float read_spread(const VividSpreadPort* sp, int slot, float fallback = 0.0f) {
        if (sp && sp->data && slot >= 0 && static_cast<uint32_t>(slot) < sp->length)
            return sp->data[slot];
        return fallback;
    }

    // --- Biquad filter ---

    float apply_biquad(ChannelState& ch, float input, float cutoff_hz, float reso,
                       int ftype, float sr) {
        cutoff_hz = std::clamp(cutoff_hz, 20.0f, sr * 0.45f);
        reso = std::clamp(reso, 0.0f, 1.0f);

        float omega = TWO_PI_F * cutoff_hz / sr;
        float sin_w = std::sin(omega);
        float cos_w = std::cos(omega);
        float Q     = 0.5f + reso * 19.5f;
        float alpha = sin_w / (2.0f * Q);

        float b0, b1, b2, a0, a1, a2;

        switch (ftype) {
            case FILTER_LP12: case FILTER_LP24:
                b0 = (1.0f - cos_w) * 0.5f;
                b1 =  1.0f - cos_w;
                b2 = (1.0f - cos_w) * 0.5f;
                a0 =  1.0f + alpha;
                a1 = -2.0f * cos_w;
                a2 =  1.0f - alpha;
                break;
            case FILTER_HP12:
                b0 = (1.0f + cos_w) * 0.5f;
                b1 = -(1.0f + cos_w);
                b2 = (1.0f + cos_w) * 0.5f;
                a0 =  1.0f + alpha;
                a1 = -2.0f * cos_w;
                a2 =  1.0f - alpha;
                break;
            case FILTER_BP:
                b0 =  sin_w * 0.5f;
                b1 =  0.0f;
                b2 = -sin_w * 0.5f;
                a0 =  1.0f + alpha;
                a1 = -2.0f * cos_w;
                a2 =  1.0f - alpha;
                break;
            case FILTER_NOTCH:
                b0 =  1.0f;
                b1 = -2.0f * cos_w;
                b2 =  1.0f;
                a0 =  1.0f + alpha;
                a1 = -2.0f * cos_w;
                a2 =  1.0f - alpha;
                break;
            default:
                return input;
        }

        float inv_a0 = 1.0f / a0;
        b0 *= inv_a0; b1 *= inv_a0; b2 *= inv_a0;
        a1 *= inv_a0; a2 *= inv_a0;

        float out = b0 * input + ch.fz1[0];
        ch.fz1[0] = b1 * input - a1 * out + ch.fz2[0];
        ch.fz2[0] = b2 * input - a2 * out;

        if (ftype == FILTER_LP24) {
            float in2 = out;
            out = b0 * in2 + ch.fz1[1];
            ch.fz1[1] = b1 * in2 - a1 * out + ch.fz2[1];
            ch.fz2[1] = b2 * in2 - a2 * out;
        }

        return out;
    }

    float apply_filter(ChannelState& ch, float input, float cutoff_hz, float reso,
                       int ftype, float sr) {
        switch (ftype) {
            case FILTER_LP12: case FILTER_LP24: case FILTER_HP12:
            case FILTER_BP:   case FILTER_NOTCH:
                return apply_biquad(ch, input, cutoff_hz, reso, ftype, sr);
            case FILTER_COMB: {
                float delay_samples = sr / std::max(cutoff_hz, 20.0f);
                float feedback = reso * 0.98f;
                return ch.comb.process(input, delay_samples, feedback);
            }
            case FILTER_LADDER:
                return ch.ladder.process(input, cutoff_hz, reso, sr);
            case FILTER_FORMANT: {
                float morph = std::log2(cutoff_hz / 20.0f)
                            / std::log2(20000.0f / 20.0f);
                morph = std::clamp(morph, 0.0f, 1.0f);
                return ch.formant.process(input, morph, reso, sr);
            }
            default:
                return input;
        }
    }

    // --- Main process ---

    void process_audio(const VividAudioContext* ctx) override {
        uint32_t frames = ctx->buffer_size;
        float sr = static_cast<float>(ctx->sample_rate);

        int   ftype    = filter_type.int_value();
        float f_cutoff = filter_cutoff.value;
        float f_reso   = filter_resonance.value;
        float f_kt     = filter_keytrack.value;
        float f_drive  = filter_drive.value;

        // Determine channel count from input buffer
        // The framework sets up input/output with matched channel counts
        uint32_t num_ch = ctx->input_channel_counts ? ctx->input_channel_counts[0] : 2;
        if (num_ch > kMaxChannels) num_ch = kMaxChannels;

        const VividSpreadPort* freq_sp = ctx->input_spreads ? &ctx->input_spreads[1] : nullptr;
        const VividSpreadPort* gates_sp = ctx->input_spreads ? &ctx->input_spreads[2] : nullptr;
        const VividSpreadPort* cutoff_sp = ctx->input_spreads ? &ctx->input_spreads[3] : nullptr;

        float* in_buf  = ctx->input_buffers[0];
        float* out_buf = ctx->output_buffers[0];

        for (uint32_t ch = 0; ch < num_ch; ++ch) {
            ChannelState& cs = channels_[ch];

            // Gate-on detection: reset filter state
            float gate = read_spread(gates_sp, ch);
            bool gated = gate > 0.5f;
            if (gated && !cs.prev_gated) {
                cs.reset();
            }
            cs.prev_gated = gated;

            // Per-voice cutoff modulation
            float cutoff = f_cutoff;
            float cutoff_mod = read_spread(cutoff_sp, ch);
            if (cutoff_mod != 0.0f)
                cutoff *= std::pow(2.0f, cutoff_mod * 4.0f);

            // Keytracking
            float freq = read_spread(freq_sp, ch);
            if (f_kt > 0.0f && freq > 0.0f) {
                float oct_from_c4 = std::log2(freq / 261.63f);
                cutoff *= std::pow(2.0f, oct_from_c4 * f_kt);
            }

            float* ch_in  = in_buf  + ch * frames;
            float* ch_out = out_buf + ch * frames;

            for (uint32_t s = 0; s < frames; ++s) {
                float sig = ch_in[s];

                // Drive
                if (f_drive > 0.001f) {
                    float d = 1.0f + f_drive * 7.0f;
                    sig = std::tanh(sig * d) / std::tanh(d);
                }

                ch_out[s] = apply_filter(cs, sig, cutoff, f_reso, ftype, sr);
            }
        }

        // Zero unused channels
        for (uint32_t ch = num_ch; ch < static_cast<uint32_t>(kMaxChannels); ++ch) {
            float* ch_out = out_buf + ch * frames;
            std::memset(ch_out, 0, frames * sizeof(float));
        }
    }
};

VIVID_REGISTER(PolyFilter)
