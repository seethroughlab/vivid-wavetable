#include "operator_api/operator.h"
#include "operator_api/adsr.h"
#include "operator_api/audio_dsp.h"
#include "operator_api/note_types.h"
#include "operator_api/thumbnail.h"
#include "operator_api/draw_plot_helpers.h"
#include "operator_api/type_id.h"
#include "operator_api/voice_table.h"
#include "voice_breakouts.h"
#include "lane_audio_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// =============================================================================
// NoiseLayer — polyphonic per-voice noise/air source
// =============================================================================

/**
 * @brief Polyphonic per-voice noise source for air, breath, and transient detail.
 *
 * Drive with `notes_in` from any note source — voices are allocated internally
 * with a built-in ADSR and summed into a stereo output. The advanced
 * `voices_out` (per-voice multichannel) and `voice_*` breakouts expose
 * per-voice state for downstream VoiceMixer / EnvelopeAu / Filter routing.
 *
 * @input notes_in Native note stream — canonical input for note sources.
 * @output output Stereo summed audio.
 * @output voices_out Advanced: per-voice audio channels (note_id sorted).
 * @output voice_ids Advanced: per-voice note_id, sorted ascending.
 * @output voice_gates Advanced: per-voice gate (1.0 while held).
 * @output voice_velocities Advanced: per-voice velocity 0..1.
 * @output voice_freqs Advanced: per-voice frequency in Hz.
 * @recipe Tracker/notes_out -> NoiseLayer/notes_in
 * @recipe NoiseLayer/output -> audio_out/input
 * @recipe NoiseLayer/voices_out -> VoiceMixer/input
 * @pitfall NoiseLayer voices_out is still a per-voice source. Route it through VoiceMixer instead of treating it like a ready-made global hiss bed.
 * @family voice_source
 * @best_used_with Tracker, VoiceMixer, EnvelopeAu
 * @common_companions WavetableLayer, WavetableOsc, AnalogOsc, Filter
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
    // ADSR for the MIDI-driven path. Lane-array driven graphs typically run
    // their envelope upstream (e.g., via VoiceMixer's amp_env_audio).
    vivid::Param<float> attack  {"attack",  0.005f, 0.001f, 5.0f};
    vivid::Param<float> decay   {"decay",   0.05f,  0.001f, 5.0f};
    vivid::Param<float> sustain {"sustain", 1.0f,   0.0f,   1.0f};
    vivid::Param<float> release {"release", 0.1f,   0.001f, 5.0f};

    // MIDI-driven path state: noise generators, lp filter, attack burst,
    // and primary ADSR that gates the MIDI path's stereo summed output.
    struct MidiVoice {
        audio_dsp::WhiteNoise white;
        audio_dsp::PinkNoise pink;
        audio_dsp::BrownNoise brown;
        audio_dsp::BlueNoise blue;
        audio_dsp::VioletNoise violet;
        float lp_state = 0.0f;
        float attack_env = 0.0f;
        vivid::adsr::State env;
        bool initialized = false;
    };
    MidiVoice midi_voices_[kMaxVoices] = {};
    vivid::VoiceTable<kMaxVoices> midi_allocator_;
    uint64_t midi_frame_counter_ = 0;

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
        param_group(attack,  "Envelope");
        param_group(decay,   "Envelope");
        param_group(sustain, "Envelope");
        param_group(release, "Envelope");

        out.push_back(&color);
        out.push_back(&level);
        out.push_back(&tone);
        out.push_back(&tone_tracking);
        out.push_back(&attack_burst);
        out.push_back(&attack_decay_ms);
        out.push_back(&velocity_to_level);
        out.push_back(&attack);
        out.push_back(&decay);
        out.push_back(&sustain);
        out.push_back(&release);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        // Canonical native note input — drive directly from Tracker/NotePattern/etc.
        out.push_back(VIVID_CUSTOM_REF_PORT("notes_in", VIVID_PORT_INPUT, VividNoteBuffer)); // 0

        // Primary stereo output — sum of all active voices.
        out.push_back({"output", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_OUTPUT,
                       VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 2});

        // Advanced per-voice breakouts. voices_out is the multichannel buffer
        // (one channel per voice) that previously lived on `output`.
        out.push_back({"voices_out", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_OUTPUT,
                       VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, kMaxVoices});
        vivid::advanced_breakout(out.back());
        out.push_back({"voice_ids",        VIVID_PORT_LANE_ARRAY, VIVID_PORT_OUTPUT});
        vivid::advanced_breakout(out.back());
        out.push_back({"voice_gates",      VIVID_PORT_LANE_ARRAY, VIVID_PORT_OUTPUT});
        vivid::advanced_breakout(out.back());
        out.push_back({"voice_velocities", VIVID_PORT_LANE_ARRAY, VIVID_PORT_OUTPUT});
        vivid::advanced_breakout(out.back());
        out.push_back({"voice_freqs",      VIVID_PORT_LANE_ARRAY, VIVID_PORT_OUTPUT});
        vivid::advanced_breakout(out.back());
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

    template <typename V>
    static float sample_color(V& voice, int color_index) {
        switch (color_index) {
            case COLOR_WHITE: return voice.white.next();
            case COLOR_PINK: return voice.pink.next();
            case COLOR_BROWN: return voice.brown.next();
            case COLOR_BLUE: return voice.blue.next();
            case COLOR_VIOLET: return voice.violet.next();
            default: return voice.pink.next();
        }
    }

    template <typename V>
    static void seed_noise_voice(V& voice, uint32_t seed) {
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

    void draw_thumbnail(const VividThumbnailContext* ctx) override {
        if (!ctx || !ctx->draw.opaque) return;
        auto& d = const_cast<VividDrawAPI&>(ctx->draw);
        void* o = d.opaque;

        float w = static_cast<float>(ctx->thumbnail_logical_width ? ctx->thumbnail_logical_width : ctx->thumbnail_width);
        float h = static_cast<float>(ctx->thumbnail_logical_height ? ctx->thumbnail_logical_height : ctx->thumbnail_height);

        int   clr  = (ctx->param_count > 0) ? static_cast<int>(ctx->param_values[0]) : 1;
        float lvl  = (ctx->param_count > 1) ? std::clamp(ctx->param_values[1], 0.0f, 1.0f) : 0.12f;
        float tn   = (ctx->param_count > 2) ? std::clamp(ctx->param_values[2], 0.0f, 1.0f) : 0.68f;

        vivid::draw_plot::draw_thumb_background(d, o, w, h);

        // Color name
        const char* cn = "PINK";
        VividColor accent = {0.75f, 0.55f, 0.65f, 0.9f};
        switch (clr) {
            case 0: cn = "WHITE";  accent = {0.75f, 0.75f, 0.75f, 0.9f}; break;
            case 1: cn = "PINK";   accent = {0.80f, 0.50f, 0.60f, 0.9f}; break;
            case 2: cn = "BROWN";  accent = {0.65f, 0.45f, 0.30f, 0.9f}; break;
            case 3: cn = "BLUE";   accent = {0.40f, 0.55f, 0.80f, 0.9f}; break;
            case 4: cn = "VIOLET"; accent = {0.60f, 0.40f, 0.80f, 0.9f}; break;
        }
        vivid::draw_plot::draw_thumb_label(d, o, 6.0f, 4.0f, cn, accent, 0.8f);

        // Level meter (left)
        float bar_w = w * 0.2f;
        float bar_left = w * 0.1f;
        float bar_top = 22.0f;
        float bar_h = h - bar_top - 6.0f;
        VividColor lo = {accent.r * 0.6f, accent.g * 0.6f, accent.b * 0.6f, 0.86f};
        vivid::draw_plot::draw_scalar_meter(d, o,
            bar_left, bar_top, bar_w, bar_h, lvl,
            {0.16f, 0.16f, 0.19f, 0.8f}, lo, accent, 2.0f, -1.0f);

        // Tone meter (right)
        float tone_left = w * 0.55f;
        vivid::draw_plot::draw_scalar_meter(d, o,
            tone_left, bar_top, bar_w, bar_h, tn,
            {0.16f, 0.16f, 0.19f, 0.8f},
            {0.35f, 0.35f, 0.40f, 0.86f},
            {0.75f, 0.78f, 0.82f, 0.86f},
            2.0f, -1.0f);

        // Labels under meters
        float label_y = bar_top + bar_h + 1.0f;
        if (d.draw_text) {
            d.draw_text(o, bar_left, label_y, "LVL", {0.45f, 0.50f, 0.55f, 0.7f}, 0.55f);
            d.draw_text(o, tone_left, label_y, "TONE", {0.45f, 0.50f, 0.55f, 0.7f}, 0.55f);
        }
    }

    // Generate one shaped noise sample. Updates lp_state and decays attack_env.
    template <typename V>
    static float render_noise_sample(V& voice, int color_index, float lp_coeff,
                                     float tracked_tone, float base_level,
                                     float velocity_gain, float burst,
                                     float attack_decay_coeff) {
        float raw = sample_color(voice, color_index);
        voice.lp_state += lp_coeff * (raw - voice.lp_state);
        float hp = raw - voice.lp_state;
        float warm = voice.lp_state * 0.88f + raw * 0.12f;
        float airy = raw * 0.45f + hp * 1.25f;
        float shaped = warm * (1.0f - tracked_tone) + airy * tracked_tone;
        float onset_gain = 1.0f + burst * voice.attack_env * voice.attack_env;
        float out = std::clamp(shaped * base_level * velocity_gain * onset_gain, -1.5f, 1.5f);
        voice.attack_env *= attack_decay_coeff;
        return out;
    }

    void process_audio_midi(const VividAudioContext* ctx, uint32_t frames, float sample_rate,
                            int color_index, float base_level, float tone_base,
                            float tracking, float burst, float velocity_mix,
                            float attack_decay_coeff) {
        const auto* notes = static_cast<const VividNoteBuffer*>(ctx->custom_inputs[0]);

        // Process note events: trigger envelope + onset burst on note-on,
        // release envelope on note-off. Slot stays alive until env hits IDLE.
        midi_allocator_.process_note_buffer(notes, midi_frame_counter_,
            [this](int slot, int /*note*/, float /*vel*/, uint32_t /*offset*/, uint64_t note_id) {
                auto& vs = midi_voices_[slot];
                if (!vs.initialized) {
                    seed_noise_voice(vs, lane_seed(static_cast<uint32_t>(
                        note_id ? note_id : (slot + 1))));
                    vs.initialized = true;
                }
                vs.attack_env = 1.0f;
                vivid::adsr::gate_on(vs.env);
            },
            [this](int slot, int /*note*/, uint64_t /*note_id*/) {
                vivid::adsr::gate_off(midi_voices_[slot].env);
            },
            [](int /*slot*/, VividNoteEventType /*kind*/, float /*value*/) {});

        // Sort active slots by note_id so voices_out channels and the four
        // voice_* breakout lanes line up across operators (cross-cutting #4).
        int sorted_idx[kMaxVoices];
        int slot_to_pos[kMaxVoices];
        int n_active = vivid_sequencers::collect_sorted_voice_indices(
            midi_allocator_.slots, kMaxVoices, sorted_idx, kMaxVoices);
        for (int v = 0; v < kMaxVoices; ++v) slot_to_pos[v] = -1;
        for (int i = 0; i < n_active; ++i) slot_to_pos[sorted_idx[i]] = i;

        float* stereo_out = ctx->output_buffers && ctx->output_buffers[0]
                            ? ctx->output_buffers[0] : nullptr;
        float* voices_buf = ctx->output_buffers && ctx->output_buffers[1]
                            ? ctx->output_buffers[1] : nullptr;
        if (stereo_out) std::memset(stereo_out, 0, 2 * frames * sizeof(float));
        if (voices_buf) std::memset(voices_buf, 0, kMaxVoices * frames * sizeof(float));

        const float dt = 1.0f / sample_rate;

        for (uint32_t s = 0; s < frames; ++s) {
            float sample = 0.0f;
            for (int v = 0; v < kMaxVoices; ++v) {
                auto& slot = midi_allocator_.slots[v];
                if (!slot.active) continue;
                auto& vs = midi_voices_[v];

                vivid::adsr::advance(vs.env, dt, attack.value, decay.value,
                                     sustain.value, release.value);

                const float voice_freq = 440.0f *
                    std::pow(2.0f, (static_cast<float>(slot.note) - 69.0f
                                    + slot.pitch_bend_semis) / 12.0f);
                float note_octaves = std::log2(std::max(voice_freq, 1.0f) / kC4Hz);
                float tracked_tone = vivid_wavetable::lane_audio::clamp01(
                    tone_base + note_octaves * 0.18f * tracking);
                float cutoff = 180.0f + std::pow(tracked_tone, 1.35f) * 9500.0f;
                float lp_coeff = vivid_wavetable::lane_audio::one_pole_coeff(sample_rate, cutoff);
                float velocity_gain = (1.0f - velocity_mix) + velocity_mix * slot.velocity;

                float shaped = render_noise_sample(vs, color_index, lp_coeff, tracked_tone,
                                                   base_level, velocity_gain, burst,
                                                   attack_decay_coeff);
                const float voice_sample = shaped * vs.env.env_value;
                sample += voice_sample;

                if (voices_buf && slot_to_pos[v] >= 0) {
                    voices_buf[slot_to_pos[v] * frames + s] = voice_sample;
                }

                if (vs.env.stage == vivid::adsr::IDLE) slot.active = false;
            }
            if (stereo_out) {
                stereo_out[s]            = sample;
                stereo_out[frames + s]   = sample;
            }
            ++midi_frame_counter_;
        }

        // Emit voice_* breakouts in note_id-sorted order.
        if (ctx->output_lanes) {
            VividLaneOutput lanes[vivid_sequencers::kVoiceBreakoutLaneCount] = {
                ctx->output_lanes[0], ctx->output_lanes[1],
                ctx->output_lanes[2], ctx->output_lanes[3],
            };
            vivid_sequencers::emit_voice_breakouts(midi_allocator_, lanes);
        }
    }

    void process_audio(const VividAudioContext* ctx) override {
        uint32_t frames = ctx->buffer_size;
        float sample_rate = static_cast<float>(ctx->sample_rate);

        int color_index = color.int_value();
        float base_level = level.value;
        float tone_base = tone.value;
        float tracking = tone_tracking.value;
        float burst = attack_burst.value;
        float velocity_mix = velocity_to_level.value;
        float attack_decay = attack_decay_ms.value;
        float attack_samples = std::max(1.0f, attack_decay * 0.001f * sample_rate);
        float attack_decay_coeff = std::exp(-1.0f / attack_samples);

        process_audio_midi(ctx, frames, sample_rate, color_index, base_level,
                           tone_base, tracking, burst, velocity_mix,
                           attack_decay_coeff);
    }
};

VIVID_REGISTER(NoiseLayer)
VIVID_THUMBNAIL(NoiseLayer)
