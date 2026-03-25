#include "operator_api/operator.h"
#include "operator_api/audio_operator.h"
#include "operator_api/type_id.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr float TWO_PI_F = 2.0f * static_cast<float>(M_PI);

// =============================================================================
// AnalogOsc — polyphonic virtual analog oscillator with PolyBLEP anti-aliasing
// =============================================================================

struct AnalogOsc : vivid::AudioOperatorBase {
    static constexpr const char* kName   = "AnalogOsc";
    static constexpr bool kTimeDependent = true;

    static constexpr int kMaxVoices = 16;

    enum Waveform { WAVE_SINE, WAVE_SAW, WAVE_SQUARE, WAVE_TRIANGLE, WAVE_PULSE };
    enum ModType  { MOD_OFF, MOD_FM, MOD_RM, MOD_AM };

    // --- Parameters ---
    vivid::Param<int>   waveform    {"waveform",    1,    {"Sine", "Saw", "Square", "Triangle", "Pulse"}};
    vivid::Param<float> pulse_width {"pulse_width", 0.5f, 0.01f, 0.99f};
    vivid::Param<float> amplitude   {"amplitude",   0.3f, 0.0f,  1.0f};
    vivid::Param<float> detune      {"detune",      0.0f, 0.0f,  50.0f};
    vivid::Param<float> portamento  {"portamento",  0.0f, 0.0f,  2000.0f};
    vivid::Param<int>   mod_type    {"mod_type",    0,    {"Off", "FM", "RM", "AM"}};
    vivid::Param<float> mod_depth   {"mod_depth",   0.0f, 0.0f,  1.0f};

    // --- Per-voice state ---
    struct Voice {
        double phase        = 0;
        float  current_freq = 0;
        float  target_freq  = 0;
        bool   was_gated    = false;
    };
    Voice voices_[kMaxVoices] = {};

    AnalogOsc() {
        vivid::semantic_tag(amplitude, "amplitude_linear");
        vivid::semantic_tag(portamento, "time_milliseconds");
        vivid::semantic_unit(portamento, "ms");
    }

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        param_group(waveform,    "Core");
        param_group(pulse_width, "Core");
        param_group(amplitude,   "Core");
        param_group(detune,      "Tuning");
        param_group(portamento,  "Tuning");
        param_group(mod_type,    "Modulation");
        param_group(mod_depth,   "Modulation");

        out.push_back(&waveform);
        out.push_back(&pulse_width);
        out.push_back(&amplitude);
        out.push_back(&detune);
        out.push_back(&portamento);
        out.push_back(&mod_type);
        out.push_back(&mod_depth);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back({"frequencies", VIVID_PORT_SPREAD, VIVID_PORT_INPUT});    // 0
        out.push_back({"gates",       VIVID_PORT_SPREAD, VIVID_PORT_INPUT});    // 1
        out.push_back({"velocities",  VIVID_PORT_SPREAD, VIVID_PORT_INPUT});    // 2
        out.push_back({"pitch_mod",   VIVID_PORT_SPREAD, VIVID_PORT_INPUT});    // 3
        // N-channel audio modulation input (for FM/RM/AM from another osc)
        out.push_back({"mod_input", VIVID_PORT_AUDIO, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});     // auto channels
        // N-channel audio output
        out.push_back({"output", VIVID_PORT_AUDIO, VIVID_PORT_OUTPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, kMaxVoices});
    }

    // --- Helpers ---

    static float cents_to_ratio(float cents) {
        return std::pow(2.0f, cents / 1200.0f);
    }

    static float read_spread(const VividSpreadPort* sp, int slot, float fallback = 0.0f) {
        if (sp && sp->data && slot >= 0 && static_cast<uint32_t>(slot) < sp->length)
            return sp->data[slot];
        return fallback;
    }

    // --- PolyBLEP residual ---
    // Subtracts polynomial correction near discontinuities to reduce aliasing.
    // `t` is distance from the discontinuity, normalized by phase_inc.
    static float polyblep(double t, double phase_inc) {
        double dt = phase_inc;
        if (t < dt) {
            t /= dt;
            return static_cast<float>(t + t - t * t - 1.0);
        } else if (t > 1.0 - dt) {
            t = (t - 1.0) / dt;
            return static_cast<float>(t * t + t + t + 1.0);
        }
        return 0.0f;
    }

    // --- Waveform generation with PolyBLEP ---

    static float generate_saw(double phase, double phase_inc) {
        float saw = static_cast<float>(2.0 * phase - 1.0);
        saw -= polyblep(phase, phase_inc);  // Correct discontinuity at phase=0/1
        return saw;
    }

    static float generate_square(double phase, double phase_inc) {
        float sq = (phase < 0.5) ? 1.0f : -1.0f;
        sq += polyblep(phase, phase_inc);                    // Rising edge at 0
        sq -= polyblep(std::fmod(phase + 0.5, 1.0), phase_inc);  // Falling edge at 0.5
        return sq;
    }

    static float generate_pulse(double phase, double phase_inc, float pw) {
        float sq = (phase < static_cast<double>(pw)) ? 1.0f : -1.0f;
        sq += polyblep(phase, phase_inc);                              // Rising edge at 0
        sq -= polyblep(std::fmod(phase + (1.0 - pw), 1.0), phase_inc); // Falling edge at pw
        return sq;
    }

    static float generate_triangle(double phase, double phase_inc) {
        // Integrated square wave (leaky integrator of PolyBLEP square)
        // For simplicity, use naive triangle with no BLEP (triangle has weak harmonics,
        // aliasing is much less audible than saw/square)
        float t = static_cast<float>(phase);
        return 4.0f * (t < 0.5f ? t : (1.0f - t)) - 1.0f;
    }

    // --- Main process ---

    void process_audio(const VividAudioContext* ctx) override {
        uint32_t frames = ctx->buffer_size;
        float sr = static_cast<float>(ctx->sample_rate);

        int   wave     = waveform.int_value();
        float pw       = pulse_width.value;
        float amp      = amplitude.value;
        float det      = detune.value;
        float porta_ms = portamento.value;
        int   mtype    = mod_type.int_value();
        float mdepth   = mod_depth.value;

        const VividSpreadPort* freq_sp  = ctx->input_spreads ? &ctx->input_spreads[0] : nullptr;
        const VividSpreadPort* gates_sp = ctx->input_spreads ? &ctx->input_spreads[1] : nullptr;
        const VividSpreadPort* pitch_sp = ctx->input_spreads ? &ctx->input_spreads[3] : nullptr;

        uint32_t voice_count = freq_sp ? freq_sp->length : 0;
        if (voice_count > kMaxVoices) voice_count = kMaxVoices;

        // Portamento rate
        float porta_rate = 1.0f;
        if (porta_ms > 0.0f) {
            float porta_samples = porta_ms * 0.001f * sr;
            porta_rate = 1.0f - std::exp(-4.0f / porta_samples);
        }

        // Modulation input
        float* mod_buf = (mtype > 0 && mdepth > 0.0f && ctx->input_buffers[0])
                         ? ctx->input_buffers[0] : nullptr;
        uint32_t mod_channels = mod_buf && ctx->input_channel_counts
                                ? ctx->input_channel_counts[0] : 0;

        // Zero all output channels
        float* out_buf = ctx->output_buffers[0];
        std::memset(out_buf, 0, kMaxVoices * frames * sizeof(float));

        for (uint32_t vi = 0; vi < voice_count; ++vi) {
            float gate = read_spread(gates_sp, vi);
            float freq_target = read_spread(freq_sp, vi);
            if (freq_target <= 0.0f) continue;

            Voice& v = voices_[vi];

            bool gate_on = (gate > 0.5f);
            if (gate_on && !v.was_gated) {
                v.phase = 0.0;
                v.current_freq = freq_target;
                v.target_freq = freq_target;
            }
            v.was_gated = gate_on;
            v.target_freq = freq_target;

            if (!gate_on) continue;

            float* ch_out = out_buf + vi * frames;
            float pitch_offset = read_spread(pitch_sp, vi);

            // Modulation input for this voice
            float* mod_ch = (mod_buf && vi < mod_channels)
                            ? mod_buf + vi * frames : nullptr;

            for (uint32_t s = 0; s < frames; ++s) {
                // Portamento
                if (porta_ms > 0.0f && v.current_freq != v.target_freq) {
                    v.current_freq += (v.target_freq - v.current_freq) * porta_rate;
                    if (std::abs(v.current_freq - v.target_freq) < 0.01f)
                        v.current_freq = v.target_freq;
                }

                float freq = v.current_freq *
                    cents_to_ratio(det) *
                    std::pow(2.0f, pitch_offset / 12.0f);
                if (!std::isfinite(freq) || freq <= 0.0f) freq = v.current_freq;

                double phase_inc = static_cast<double>(freq) / sr;

                // FM modulation
                if (mtype == MOD_FM && mod_ch) {
                    phase_inc += mod_ch[s] * mdepth * 4.0 * (freq / sr);
                }

                // Generate waveform
                float sig;
                switch (wave) {
                    case WAVE_SINE:
                        sig = std::sin(static_cast<float>(v.phase) * TWO_PI_F);
                        break;
                    case WAVE_SAW:
                        sig = generate_saw(v.phase, phase_inc);
                        break;
                    case WAVE_SQUARE:
                        sig = generate_square(v.phase, phase_inc);
                        break;
                    case WAVE_TRIANGLE:
                        sig = generate_triangle(v.phase, phase_inc);
                        break;
                    case WAVE_PULSE:
                        sig = generate_pulse(v.phase, phase_inc, pw);
                        break;
                    default:
                        sig = 0.0f;
                }

                // RM/AM
                if (mod_ch) {
                    if (mtype == MOD_RM) {
                        sig *= mod_ch[s];
                    } else if (mtype == MOD_AM) {
                        sig *= 1.0f + mdepth * mod_ch[s];
                    }
                }

                ch_out[s] = sig * amp;

                // Advance phase
                v.phase += phase_inc;
                if (v.phase >= 1.0) v.phase -= 1.0;
                if (v.phase < 0.0) v.phase += 1.0;
                if (!std::isfinite(v.phase)) v.phase = 0.0;
            }
        }
    }
};

VIVID_REGISTER(AnalogOsc)
