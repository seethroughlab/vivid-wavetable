#include "operator_api/operator.h"
#include "operator_api/audio_dsp.h"
#include "operator_api/thumbnail.h"
#include "operator_api/draw_plot_helpers.h"
#include "operator_api/type_id.h"
#include "lane_audio_utils.h"
#include <cmath>
#include <cstring>
#include <algorithm>

// =============================================================================
// SubOsc — polyphonic sub oscillator, outputs N-channel per-voice audio
// =============================================================================

/**
 * @brief Polyphonic sub oscillator for reinforcing each active voice below the main pitch.
 *
 * Generates one output channel per active lane and supports audio-rate pitch modulation
 * so the sub layer can follow the same modulation topology as the main oscillators.
 *
 * @input frequencies Per-voice frequencies from a note allocator.
 * @input gates Per-voice gates for reset and note articulation.
 * @input velocities Per-voice velocities for optional dynamic sub-level scaling.
 * @input lane_ids Stable per-voice identity tokens for persistent lane state.
 * @input pitch_mod_audio Audio-rate per-voice pitch modulation.
 * @output output Per-voice sub-layer audio channels.
 * @recipe VoiceAllocator/frequencies,gates,velocities,lane_ids -> SubOsc/frequencies,gates,velocities,lane_ids
 * @recipe SubOsc/output -> VoiceMixer/input
 * @pitfall SubOsc still emits one channel per voice; route it through VoiceMixer instead of treating it as a ready-made mono bass bus.
 * @family voice_source
 * @best_used_with VoiceAllocator, VoiceMixer, WavetableLayer
 * @common_companions AnalogOsc, EnvelopeAu, Filter
 */
struct SubOsc : vivid::OperatorBase, vivid::AudioProcessable {
    static constexpr const char* kName   = "SubOsc";
    static constexpr bool kTimeDependent = true;

    static constexpr int kMaxVoices = 16;

    vivid::Param<float> level    {"level",    0.35f, 0.0f, 1.0f};
    vivid::Param<float> velocity_to_level {"velocity_to_level", 0.35f, 0.0f, 1.0f};
    vivid::Param<int>   octave   {"octave",   0,    {"-1", "-2"}};
    vivid::Param<int>   waveform {"waveform", 0,    {"Sine", "Triangle", "Saw", "Square", "Noise"}};

    struct Voice {
        double phase     = 0;
        bool   was_gated = false;
        audio_dsp::WhiteNoise white_noise;
    };

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        param_group(level,    "Sub");
        param_group(velocity_to_level, "Dynamics");
        param_group(octave,   "Sub");
        param_group(waveform, "Sub");

        out.push_back(&level);
        out.push_back(&velocity_to_level);
        out.push_back(&octave);
        out.push_back(&waveform);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back({"frequencies", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});    // 0
        out.push_back({"gates",       VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});    // 1
        out.push_back({"velocities",  VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});    // 2
        out.push_back({"lane_ids",    VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});    // 3 (identity tokens)
        // Audio-rate pitch modulation (N-channel, one per voice, semitones)
        out.push_back({"pitch_mod_audio", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});     // 4
        out.push_back({"output", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_OUTPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, kMaxVoices});
    }

    void draw_thumbnail(const VividThumbnailContext* ctx) override {
        if (!ctx || !ctx->draw.opaque) return;
        auto& d = const_cast<VividDrawAPI&>(ctx->draw);
        void* o = d.opaque;

        float w = static_cast<float>(ctx->thumbnail_logical_width ? ctx->thumbnail_logical_width : ctx->thumbnail_width);
        float h = static_cast<float>(ctx->thumbnail_logical_height ? ctx->thumbnail_logical_height : ctx->thumbnail_height);

        float lvl  = (ctx->param_count > 0) ? std::clamp(ctx->param_values[0], 0.0f, 1.0f) : 0.5f;
        int   oct  = (ctx->param_count > 1) ? static_cast<int>(ctx->param_values[1]) : 0;
        int   wave = (ctx->param_count > 2) ? static_cast<int>(ctx->param_values[2]) : 0;

        vivid::draw_plot::draw_thumb_background(d, o, w, h);

        // Octave label
        vivid::draw_plot::draw_thumb_label(d, o, 6.0f, 4.0f,
            oct == 0 ? "-1 OCT" : "-2 OCT", {0.55f, 0.50f, 0.65f, 0.9f}, 0.8f);

        // Waveform name
        const char* wn = "SIN";
        switch (wave) {
            case 0: wn = "SIN"; break;
            case 1: wn = "TRI"; break;
            case 2: wn = "SAW"; break;
            case 3: wn = "SQR"; break;
            case 4: wn = "NSE"; break;
        }
        vivid::draw_plot::draw_thumb_value(d, o, w - 34.0f, 4.0f, 28.0f, wn,
            {0.55f, 0.50f, 0.65f, 0.9f}, 0.75f);

        // Level meter
        float bar_w = w * 0.25f;
        float bar_left = w * 0.1f;
        float bar_top = 22.0f;
        float bar_h = h - bar_top - 6.0f;
        vivid::draw_plot::draw_scalar_meter(d, o,
            bar_left, bar_top, bar_w, bar_h, lvl,
            {0.16f, 0.16f, 0.19f, 0.8f},
            {0.45f, 0.38f, 0.70f, 0.86f},  // low: purple
            {0.65f, 0.50f, 0.85f, 0.86f},  // high: bright purple
            2.0f, -1.0f);

        // Waveform shape (right side)
        auto sample_fn = [wave](float phase) {
            float p = phase - std::floor(phase);
            switch (wave) {
                case 0: return std::sin(p * 2.0f * 3.14159265f);
                case 1: return 4.0f * ((p < 0.5f) ? p : 1.0f - p) - 1.0f;
                case 2: return 2.0f * p - 1.0f;
                case 3: return (p < 0.5f) ? 1.0f : -1.0f;
                default: return 0.0f;
            }
        };
        if (wave < 4) {
            vivid::draw_plot::draw_waveform_plot(d, o,
                bar_left + bar_w + 10.0f, 22.0f, w - bar_left - bar_w - 18.0f, bar_h,
                sample_fn,
                {0.45f, 0.38f, 0.70f, 0.25f},
                {0.65f, 0.50f, 0.85f, 0.85f},
                {0.24f, 0.25f, 0.29f, 0.5f},
                true, 1.0f, 2.0f);
        } else {
            // Noise — draw random dots
            vivid::draw_plot::draw_thumb_label(d, o, bar_left + bar_w + 14.0f,
                bar_top + bar_h * 0.4f, "NOISE", {0.65f, 0.50f, 0.85f, 0.7f}, 0.8f);
        }
    }

    void process_audio(const VividAudioContext* ctx) override {
        uint32_t frames = ctx->buffer_size;
        float sr = static_cast<float>(ctx->sample_rate);

        float lvl     = level.value;
        float v2l     = velocity_to_level.value;
        float sub_div = (octave.int_value() == 1) ? 4.0f : 2.0f;
        int   wave    = waveform.int_value();

        const VividLaneView* freq_lane    = ctx->input_lanes ? &ctx->input_lanes[0] : nullptr;
        const VividLaneView* gates_lane   = ctx->input_lanes ? &ctx->input_lanes[1] : nullptr;
        const VividLaneView* vel_lane     = ctx->input_lanes ? &ctx->input_lanes[2] : nullptr;
        const VividLaneView* lane_id_lane = ctx->input_lanes ? &ctx->input_lanes[3] : nullptr;

        uint32_t voice_count = freq_lane ? freq_lane->length : 0;
        if (voice_count > static_cast<uint32_t>(kMaxVoices)) voice_count = kMaxVoices;

        // Audio-rate pitch modulation (port 4: after 4 lane ports)
        float* pitch_mod_buf = ctx->input_buffers[4];
        uint32_t pitch_mod_ch = pitch_mod_buf && ctx->input_channel_counts
                                ? ctx->input_channel_counts[4] : 0;

        float* out_buf = ctx->output_buffers[0];
        std::memset(out_buf, 0, kMaxVoices * frames * sizeof(float));

        // Waveform mapping: param order (Sine=0, Tri=1, Saw=2, Sq=3)
        // to audio_dsp::waveform order (sine=0, saw=1, sq=2, tri=3)
        static constexpr int wf_map[] = {0, 3, 1, 2};

        for (uint32_t vi = 0; vi < voice_count; ++vi) {
            float gate = vivid_wavetable::lane_audio::read_lane(gates_lane, vi, 0.0f);
            float freq = vivid_wavetable::lane_audio::read_lane(freq_lane, vi, 0.0f);
            float velocity = vivid_wavetable::lane_audio::clamp01(
                vivid_wavetable::lane_audio::read_lane(vel_lane, vi, 1.0f));
            if (freq <= 0.0f) continue;
            // Don't skip gate=0 voices — releasing voices need audio for
            // downstream envelope release tails.

            uint32_t lid = vivid_wavetable::lane_audio::resolve_lane_id(lane_id_lane, vi);
            Voice& v = *vivid_lane_state(ctx, lid, Voice);

            bool gate_on = (gate > 0.5f);
            if (gate_on && !v.was_gated) {
                v.phase = 0.0;
                v.white_noise.state = 12345u + static_cast<uint32_t>(vi) * 1664525u;
            }
            v.was_gated = gate_on;

            float base_sub_freq = freq / sub_div;
            float base_sub_inc  = base_sub_freq / sr;
            float velocity_gain = (1.0f - v2l) + v2l * velocity;
            float* ch_out  = out_buf + vi * frames;
            float* pitch_mod_voice = vivid_wavetable::lane_audio::resolve_mod_channel(
                pitch_mod_buf, pitch_mod_ch, vi, frames);

            for (uint32_t s = 0; s < frames; ++s) {
                float sig;
                if (wave == 4) {
                    sig = v.white_noise.next();
                } else {
                    sig = static_cast<float>(audio_dsp::waveform(v.phase, wf_map[wave]));
                }
                ch_out[s] = sig * lvl * velocity_gain;

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
VIVID_THUMBNAIL(SubOsc)
