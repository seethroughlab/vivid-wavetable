#include "operator_api/operator.h"
#include "operator_api/audio_dsp.h"
#include "operator_api/type_id.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// =============================================================================
// NoiseLayer — polyphonic per-voice noise/air source
// =============================================================================

/**
 * @brief Polyphonic per-voice noise source for air, breath, and transient detail.
 *
 * Generates one audio channel per active voice lane and keeps independent noise and
 * shaping state keyed by `lane_ids`, so it can sit beside the main oscillator path
 * while still following per-note articulation and release tails.
 *
 * @input frequencies Per-voice frequencies used for tone-tracking brightness.
 * @input gates Per-voice gates used to trigger the short attack emphasis.
 * @input velocities Per-voice velocities used for level scaling.
 * @input lane_ids Stable per-voice identity tokens for persistent lane state.
 * @output output Per-voice audio channels for downstream VoiceMixer reduction.
 * @recipe PolyVoiceAllocator/frequencies,gates,velocities,lane_ids -> NoiseLayer/frequencies,gates,velocities,lane_ids
 * @recipe NoiseLayer/output -> VoiceMixer/input with EnvelopeAu/value -> VoiceMixer/amp_env_audio
 * @pitfall NoiseLayer is still a per-voice source. Route it through VoiceMixer instead of treating it like a ready-made global hiss bed.
 * @family voice_source
 * @best_used_with PolyVoiceAllocator, VoiceMixer, EnvelopeAu
 * @common_companions WavetableOsc, AnalogOsc, Filter
 */
struct NoiseLayer : vivid::OperatorBase, vivid::AudioProcessable {
    static constexpr const char* kName = "NoiseLayer";
    static constexpr bool kTimeDependent = true;

    static constexpr int kMaxVoices = 16;
    static constexpr float kC4Hz = 261.625565f;

    enum Color {
        COLOR_WHITE = 0,
        COLOR_PINK = 1,
        COLOR_BROWN = 2,
        COLOR_BLUE = 3,
        COLOR_VIOLET = 4,
    };

    vivid::Param<int> color {"color", 1, {"White", "Pink", "Brown", "Blue", "Violet"}};
    vivid::Param<float> level {"level", 0.12f, 0.0f, 1.0f};
    vivid::Param<float> tone {"tone", 0.68f, 0.0f, 1.0f};
    vivid::Param<float> tone_tracking {"tone_tracking", 0.20f, 0.0f, 1.0f};
    vivid::Param<float> attack_burst {"attack_burst", 0.25f, 0.0f, 1.0f};
    vivid::Param<float> attack_decay_ms {"attack_decay_ms", 40.0f, 5.0f, 250.0f};
    vivid::Param<float> velocity_to_level {"velocity_to_level", 0.60f, 0.0f, 1.0f};

    struct Voice {
        audio_dsp::WhiteNoise white;
        audio_dsp::PinkNoise pink;
        audio_dsp::BrownNoise brown;
        audio_dsp::BlueNoise blue;
        audio_dsp::VioletNoise violet;
        float lp_state = 0.0f;
        float attack_env = 0.0f;
        bool was_gated = false;
        bool initialized = false;
    };

    NoiseLayer() {
        vivid::semantic_tag(level, "amplitude_linear");
        vivid::semantic_tag(attack_decay_ms, "time_milliseconds");
        vivid::semantic_unit(attack_decay_ms, "ms");
    }

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        param_group(color, "Core");
        param_group(level, "Core");
        param_group(tone, "Tone");
        param_group(tone_tracking, "Tone");
        param_group(attack_burst, "Attack");
        param_group(attack_decay_ms, "Attack");
        param_group(velocity_to_level, "Dynamics");

        out.push_back(&color);
        out.push_back(&level);
        out.push_back(&tone);
        out.push_back(&tone_tracking);
        out.push_back(&attack_burst);
        out.push_back(&attack_decay_ms);
        out.push_back(&velocity_to_level);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back({"frequencies", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});
        out.push_back({"gates", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});
        out.push_back({"velocities", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});
        out.push_back({"lane_ids", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});
        out.push_back({"output", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_OUTPUT,
                       VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, kMaxVoices});
    }

    static float read_lane(const VividLanePort* lane, int slot, float fallback = 0.0f) {
        if (lane && lane->data && slot >= 0 && static_cast<uint32_t>(slot) < lane->length) {
            return lane->data[slot];
        }
        return fallback;
    }

    static uint32_t lane_seed(uint32_t lane_id) {
        uint32_t seed = lane_id ? lane_id : 1u;
        seed ^= seed >> 16;
        seed *= 0x7feb352dU;
        seed ^= seed >> 15;
        seed *= 0x846ca68bU;
        seed ^= seed >> 16;
        return seed | 1u;
    }

    static void seed_voice(Voice& voice, uint32_t seed) {
        voice.white.state = 0x12345u ^ seed;
        voice.pink = {};
        voice.pink.white.state = 0x23456u ^ (seed * 1664525u);
        voice.brown = {};
        voice.brown.white.state = 0x34567u ^ (seed * 22695477u);
        voice.blue = {};
        voice.blue.white.state = 0x45678u ^ (seed * 1103515245u);
        voice.violet = {};
        voice.violet.white.state = 0x56789u ^ (seed * 214013u);
        voice.lp_state = 0.0f;
        voice.attack_env = 0.0f;
    }

    static float clamp01(float x) {
        return std::clamp(x, 0.0f, 1.0f);
    }

    static float one_pole_coeff(float sr, float cutoff_hz) {
        cutoff_hz = std::clamp(cutoff_hz, 10.0f, sr * 0.45f);
        return 1.0f - std::exp(-2.0f * 3.14159265358979323846f * cutoff_hz / sr);
    }

    static float sample_color(Voice& voice, int color_index) {
        switch (color_index) {
            case COLOR_WHITE: return voice.white.next();
            case COLOR_PINK: return voice.pink.next();
            case COLOR_BROWN: return voice.brown.next();
            case COLOR_BLUE: return voice.blue.next();
            case COLOR_VIOLET: return voice.violet.next();
            default: return voice.pink.next();
        }
    }

    void process_audio(const VividAudioContext* ctx) override {
        uint32_t frames = ctx->buffer_size;
        float sample_rate = static_cast<float>(ctx->sample_rate);

        const VividLanePort* freq_lane = ctx->input_lanes ? &ctx->input_lanes[0] : nullptr;
        const VividLanePort* gates_lane = ctx->input_lanes ? &ctx->input_lanes[1] : nullptr;
        const VividLanePort* vel_lane = ctx->input_lanes ? &ctx->input_lanes[2] : nullptr;
        const VividLanePort* lane_id_lane = ctx->input_lanes ? &ctx->input_lanes[3] : nullptr;

        uint32_t voice_count = freq_lane ? freq_lane->length : 0;
        if (voice_count > static_cast<uint32_t>(kMaxVoices)) voice_count = kMaxVoices;

        float* out_buf = ctx->output_buffers[0];
        std::memset(out_buf, 0, kMaxVoices * frames * sizeof(float));

        int color_index = color.int_value();
        float base_level = level.value;
        float tone_base = tone.value;
        float tracking = tone_tracking.value;
        float burst = attack_burst.value;
        float velocity_mix = velocity_to_level.value;
        float attack_decay = attack_decay_ms.value;
        float attack_samples = std::max(1.0f, attack_decay * 0.001f * sample_rate);
        float attack_decay_coeff = std::exp(-1.0f / attack_samples);

        for (uint32_t vi = 0; vi < voice_count; ++vi) {
            float freq = std::max(read_lane(freq_lane, static_cast<int>(vi), kC4Hz), 1.0f);
            float gate = read_lane(gates_lane, static_cast<int>(vi), 0.0f);
            float velocity = clamp01(read_lane(vel_lane, static_cast<int>(vi), 1.0f));

            uint32_t lane_id = lane_id_lane && lane_id_lane->data && vi < lane_id_lane->length
                ? static_cast<uint32_t>(lane_id_lane->data[vi]) : vi;
            Voice& voice = *vivid_lane_state(ctx, lane_id, Voice);
            if (!voice.initialized) {
                seed_voice(voice, lane_seed(lane_id));
                voice.initialized = true;
            }

            bool gate_on = gate > 0.5f;
            if (gate_on && !voice.was_gated) {
                voice.attack_env = 1.0f;
            }
            voice.was_gated = gate_on;

            float note_octaves = std::log2(std::max(freq, 1.0f) / kC4Hz);
            float tracked_tone = clamp01(tone_base + note_octaves * 0.18f * tracking);
            float cutoff = 180.0f + std::pow(tracked_tone, 1.35f) * 9500.0f;
            float lp_coeff = one_pole_coeff(sample_rate, cutoff);
            float velocity_gain = (1.0f - velocity_mix) + velocity_mix * velocity;

            float* ch_out = out_buf + vi * frames;
            for (uint32_t s = 0; s < frames; ++s) {
                float raw = sample_color(voice, color_index);
                voice.lp_state += lp_coeff * (raw - voice.lp_state);
                float hp = raw - voice.lp_state;
                float warm = voice.lp_state * 0.88f + raw * 0.12f;
                float airy = raw * 0.45f + hp * 1.25f;
                float shaped = warm * (1.0f - tracked_tone) + airy * tracked_tone;
                float onset_gain = 1.0f + burst * voice.attack_env * voice.attack_env;
                ch_out[s] = std::clamp(shaped * base_level * velocity_gain * onset_gain, -1.5f, 1.5f);
                voice.attack_env *= attack_decay_coeff;
            }
        }
    }
};

VIVID_REGISTER(NoiseLayer)
