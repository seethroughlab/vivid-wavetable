#include "operator_api/operator.h"
#include "operator_api/thumbnail.h"
#include "operator_api/draw_plot_helpers.h"
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

/**
 * @brief Polyphonic virtual analog oscillator with anti-aliased classic waveforms.
 *
 * Outputs one audio channel per active voice and supports both lane-based and audio-rate
 * pitch modulation plus optional FM, RM, or AM from an incoming audio signal.
 * Uses stable lane identities so each voice keeps its own phase and glide state.
 *
 * @input frequencies Per-voice frequencies from a note allocator.
 * @input gates Per-voice gates for reset and articulation.
 * @input velocities Per-voice velocities available for graph-level shaping.
 * @input pitch_mod Per-voice pitch modulation lane array.
 * @input lane_ids Stable per-voice identity tokens for persistent lane state.
 * @input mod_input Audio-rate modulation input for FM, RM, or AM.
 * @input pitch_mod_audio Audio-rate per-voice pitch modulation.
 * @output output Per-voice audio channels.
 * @recipe PolyVoiceAllocator/frequencies,gates,lane_ids -> AnalogOsc/frequencies,gates,lane_ids
 * @recipe AnalogOsc/output -> VoiceMixer/input
 * @pitfall AnalogOsc stays per-voice until VoiceMixer; do not treat its output as already summed stereo.
 * @family voice_source
 * @best_used_with PolyVoiceAllocator, VoiceMixer, Filter
 * @common_companions EnvelopeAu, WavetableOsc, SubOsc
 */
struct AnalogOsc : vivid::OperatorBase, vivid::AudioProcessable {
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
    // voices_ array removed — state is now identity-keyed via vivid_lane_state

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
        out.push_back({"frequencies", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});    // 0
        out.push_back({"gates",       VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});    // 1
        out.push_back({"velocities",  VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});    // 2
        out.push_back({"pitch_mod",   VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});    // 3
        out.push_back({"lane_ids",    VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});    // 4 (identity tokens)
        // N-channel audio modulation input (for FM/RM/AM from another osc)
        out.push_back({"mod_input", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});     // 5 (auto channels)
        // Audio-rate pitch modulation (N-channel, one per voice)
        out.push_back({"pitch_mod_audio", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});     // 6
        // N-channel audio output
        out.push_back({"output", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_OUTPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, kMaxVoices});
    }

    // --- Helpers ---

    static float cents_to_ratio(float cents) {
        return std::pow(2.0f, cents / 1200.0f);
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

    // --- Thumbnail ---

    void draw_thumbnail(const VividThumbnailContext* ctx) override {
        if (!ctx || !ctx->draw.opaque) return;
        auto& d = const_cast<VividDrawAPI&>(ctx->draw);
        void* o = d.opaque;

        float w = static_cast<float>(ctx->thumbnail_logical_width ? ctx->thumbnail_logical_width : ctx->thumbnail_width);
        float h = static_cast<float>(ctx->thumbnail_logical_height ? ctx->thumbnail_logical_height : ctx->thumbnail_height);

        int wave = (ctx->param_count > 0) ? static_cast<int>(ctx->param_values[0]) : 1;
        float pw = (ctx->param_count > 1) ? ctx->param_values[1] : 0.5f;
        float amp = (ctx->param_count > 2) ? std::clamp(ctx->param_values[2], 0.0f, 1.0f) : 0.3f;

        vivid::draw_plot::draw_thumb_background(d, o, w, h);

        const char* wave_name = "SAW";
        switch (wave) {
            case 0: wave_name = "SIN"; break;
            case 1: wave_name = "SAW"; break;
            case 2: wave_name = "SQR"; break;
            case 3: wave_name = "TRI"; break;
            case 4: wave_name = "PLS"; break;
        }
        vivid::draw_plot::draw_thumb_label(d, o, 6.0f, 4.0f, wave_name, {0.45f, 0.55f, 0.65f, 0.9f}, 0.8f);

        auto sample_fn = [wave, amp, pw](float phase) {
            float p = phase - std::floor(phase);
            float raw = 0.0f;
            switch (wave) {
                case 0: raw = std::sin(p * 2.0f * static_cast<float>(M_PI)); break;
                case 1: raw = 2.0f * p - 1.0f; break;
                case 2: raw = (p < 0.5f) ? 1.0f : -1.0f; break;
                case 3: raw = 4.0f * ((p < 0.5f) ? p : 1.0f - p) - 1.0f; break;
                case 4: raw = (p < pw) ? 1.0f : -1.0f; break;
                default: raw = std::sin(p * 2.0f * static_cast<float>(M_PI)); break;
            }
            return raw * amp;
        };

        vivid::draw_plot::draw_waveform_plot(d, o,
            8.0f, 20.0f, w - 16.0f, h - 26.0f,
            sample_fn,
            {0.38f, 0.58f, 0.42f, 0.35f},   // fill: warm green
            {0.55f, 0.82f, 0.58f, 0.95f},   // line: bright green
            {0.24f, 0.25f, 0.29f, 0.7f},
            true, 2.0f, 2.0f);
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

        const VividLanePort* freq_lane    = ctx->input_lanes ? &ctx->input_lanes[0] : nullptr;
        const VividLanePort* gates_lane   = ctx->input_lanes ? &ctx->input_lanes[1] : nullptr;
        const VividLanePort* pitch_lane   = ctx->input_lanes ? &ctx->input_lanes[3] : nullptr;
        const VividLanePort* lane_id_lane = ctx->input_lanes ? &ctx->input_lanes[4] : nullptr;

        uint32_t voice_count = freq_lane ? freq_lane->length : 0;
        if (voice_count > static_cast<uint32_t>(kMaxVoices)) voice_count = kMaxVoices;

        // Portamento rate
        float porta_rate = 1.0f;
        if (porta_ms > 0.0f) {
            float porta_samples = porta_ms * 0.001f * sr;
            porta_rate = 1.0f - std::exp(-4.0f / porta_samples);
        }

        // Modulation input — port layout: [0-4] lane (incl lane_ids), [5] mod_input, [6] pitch_mod_audio
        float* mod_buf = (mtype > 0 && mdepth > 0.0f && ctx->input_buffers[5])
                         ? ctx->input_buffers[5] : nullptr;
        uint32_t mod_channels = mod_buf && ctx->input_channel_counts
                                ? ctx->input_channel_counts[5] : 0;

        float* pitch_mod_buf = ctx->input_buffers[6];
        uint32_t pitch_mod_ch = pitch_mod_buf && ctx->input_channel_counts
                                ? ctx->input_channel_counts[6] : 0;

        // Zero all output channels
        float* out_buf = ctx->output_buffers[0];
        std::memset(out_buf, 0, kMaxVoices * frames * sizeof(float));

        for (uint32_t vi = 0; vi < voice_count; ++vi) {
            float gate = read_lane(gates_lane, vi);
            float freq_target = read_lane(freq_lane, vi);
            if (freq_target <= 0.0f) continue;

            uint32_t lid = lane_id_lane && lane_id_lane->data && vi < lane_id_lane->length
                ? static_cast<uint32_t>(lane_id_lane->data[vi]) : vi;
            Voice& v = *vivid_lane_state(ctx, lid, Voice);

            bool gate_on = (gate > 0.5f);
            if (gate_on && !v.was_gated) {
                v.phase = 0.0;
                v.current_freq = freq_target;
                v.target_freq = freq_target;
            }
            v.was_gated = gate_on;
            v.target_freq = freq_target;

            // Don't skip gate=0 voices — releasing voices need audio for
            // downstream envelope release tails.

            float* ch_out = out_buf + vi * frames;
            float pitch_offset_lane = read_lane(pitch_lane, vi);

            float* mod_ch          = resolve_mod_channel(mod_buf, mod_channels, vi, frames);
            float* pitch_mod_voice = resolve_mod_channel(pitch_mod_buf, pitch_mod_ch, vi, frames);

            for (uint32_t s = 0; s < frames; ++s) {
                // Portamento
                if (porta_ms > 0.0f && v.current_freq != v.target_freq) {
                    v.current_freq += (v.target_freq - v.current_freq) * porta_rate;
                    if (std::abs(v.current_freq - v.target_freq) < 0.01f)
                        v.current_freq = v.target_freq;
                }

                float pitch_offset = pitch_mod_voice ? pitch_mod_voice[s] : pitch_offset_lane;
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
VIVID_THUMBNAIL(AnalogOsc)
