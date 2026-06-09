#include "operator_api/operator.h"
#include "operator_api/adsr.h"
#include "operator_api/audio_dsp.h"
#include "operator_api/note_types.h"
#include "operator_api/thumbnail.h"
#include "operator_api/draw_plot_helpers.h"
#include "operator_api/type_id.h"
#include "operator_api/voice_table.h"
#include "voice_breakouts.h"
#include <algorithm>
#include <cmath>
#include <cstring>

// =============================================================================
// SubOsc — polyphonic sub oscillator, outputs N-channel per-voice audio
// =============================================================================

/**
 * @brief Polyphonic sub oscillator for reinforcing each active voice below the main pitch.
 *
 * Drive with `notes_in` from any note source — voices are allocated internally
 * with a built-in ADSR and summed into a stereo output. The advanced
 * `voices_out` (per-voice multichannel) and `voice_*` breakouts expose
 * per-voice state for downstream VoiceMixer/EnvelopeAu/Filter routing.
 *
 * @input notes_in Native note stream — canonical input for note sources.
 * @input pitch_mod_audio Audio-rate per-voice pitch modulation.
 * @output output Stereo summed audio.
 * @output voices_out Advanced: per-voice audio channels (note_id sorted).
 * @output voice_ids Advanced: per-voice note_id, sorted ascending.
 * @output voice_gates Advanced: per-voice gate (1.0 while held).
 * @output voice_velocities Advanced: per-voice velocity 0..1.
 * @output voice_freqs Advanced: per-voice frequency in Hz.
 * @recipe Tracker/notes_out -> SubOsc/notes_in
 * @recipe SubOsc/output -> audio_out/input
 * @recipe SubOsc/voices_out -> VoiceMixer/input
 * @pitfall SubOsc voices_out still emits one channel per voice; route it through VoiceMixer instead of treating it as a ready-made mono bass bus.
 * @family voice_source
 * @best_used_with Tracker, VoiceMixer, EnvelopeAu, WavetableLayer
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
    // ADSR for the MIDI-driven path. Lane-array driven graphs typically run
    // their envelope upstream (e.g., via VoiceMixer's amp_env_audio).
    vivid::Param<float> attack  {"attack",  0.005f, 0.001f, 5.0f};
    vivid::Param<float> decay   {"decay",   0.05f,  0.001f, 5.0f};
    vivid::Param<float> sustain {"sustain", 1.0f,   0.0f,   1.0f};
    vivid::Param<float> release {"release", 0.1f,   0.001f, 5.0f};

    // Per-note expression depth (Phase 5). Pressure scales per-voice
    // amplitude; timbre offsets the sub level so X-axis MPE movement
    // can dim or push the sub against the rest of the patch.
    vivid::Param<float> pressure_to_amp   {"pressure_to_amp",   0.5f,  0.0f, 1.0f};
    vivid::Param<float> timbre_to_level   {"timbre_to_level",   0.3f, -1.0f, 1.0f};

    // MIDI-driven path state: phase + ADSR + per-voice noise generator.
    struct MidiVoice {
        double phase = 0.0;
        vivid::adsr::State env;
        audio_dsp::WhiteNoise white_noise;
    };
    MidiVoice midi_voices_[kMaxVoices] = {};
    vivid::VoiceTable<kMaxVoices> midi_allocator_;
    uint64_t midi_frame_counter_ = 0;

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        param_group(level,    "Sub");
        param_group(velocity_to_level, "Dynamics");
        param_group(octave,   "Sub");
        param_group(waveform, "Sub");
        param_group(attack,  "Envelope");
        param_group(decay,   "Envelope");
        param_group(sustain, "Envelope");
        param_group(release, "Envelope");

        out.push_back(&level);
        out.push_back(&velocity_to_level);
        out.push_back(&octave);
        out.push_back(&waveform);
        out.push_back(&attack);
        out.push_back(&decay);
        out.push_back(&sustain);
        out.push_back(&release);
        param_group(pressure_to_amp, "Expression");
        param_group(timbre_to_level, "Expression");
        out.push_back(&pressure_to_amp);
        out.push_back(&timbre_to_level);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        // Canonical native note input — drive directly from Tracker/NotePattern/etc.
        out.push_back(VIVID_CUSTOM_REF_PORT("notes_in", VIVID_PORT_INPUT, VividNoteBuffer)); // 0
        // Audio-rate pitch modulation (N-channel, one per voice, semitones)
        out.push_back({"pitch_mod_audio", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});         // 1

        // Primary stereo output — sum of all active voices.
        out.push_back({"output", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_OUTPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 2});

        // Advanced per-voice breakouts. voices_out is the multichannel buffer
        // (one channel per voice) that previously lived on `output`.
        out.push_back({"voices_out", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_OUTPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, kMaxVoices});
        vivid::advanced_breakout(out.back());
        out.push_back({.name="voice_ids", .type=VIVID_PORT_SCALAR, .direction=VIVID_PORT_OUTPUT, .multiplicity=VIVID_MULTIPLICITY_MANY});
        vivid::advanced_breakout(out.back());
        out.push_back({.name="voice_gates", .type=VIVID_PORT_SCALAR, .direction=VIVID_PORT_OUTPUT, .multiplicity=VIVID_MULTIPLICITY_MANY});
        vivid::advanced_breakout(out.back());
        out.push_back({.name="voice_velocities", .type=VIVID_PORT_SCALAR, .direction=VIVID_PORT_OUTPUT, .multiplicity=VIVID_MULTIPLICITY_MANY});
        vivid::advanced_breakout(out.back());
        out.push_back({.name="voice_freqs", .type=VIVID_PORT_SCALAR, .direction=VIVID_PORT_OUTPUT, .multiplicity=VIVID_MULTIPLICITY_MANY});
        vivid::advanced_breakout(out.back());
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

    // Waveform mapping: param order (Sine=0, Tri=1, Saw=2, Sq=3) to
    // audio_dsp::waveform order (sine=0, saw=1, sq=2, tri=3). Index 4 is
    // noise (handled separately).
    static constexpr int wf_map_[] = {0, 3, 1, 2};

    static float render_voice_sample(int wave, double phase,
                                     audio_dsp::WhiteNoise& white_noise) {
        if (wave == 4) return white_noise.next();
        return static_cast<float>(audio_dsp::waveform(phase, wf_map_[wave]));
    }

    void process_audio_midi(const VividAudioContext* ctx, uint32_t frames, float sr,
                            float lvl, float v2l, float sub_div, int wave) {
        const auto* notes = static_cast<const VividNoteBuffer*>(ctx->custom_inputs[0]);

        // Port indices: notes_in=0, pitch_mod_audio=1.
        float* pitch_mod_buf = ctx->input_buffers ? ctx->input_buffers[1] : nullptr;
        uint32_t pitch_mod_ch = pitch_mod_buf && ctx->input_channel_counts
                                ? ctx->input_channel_counts[1] : 0;

        // Process note events: trigger envelope + reset phase on note-on,
        // release envelope on note-off.
        midi_allocator_.process_note_buffer(notes, midi_frame_counter_,
            [this](int slot, int /*note*/, float /*vel*/, uint32_t /*offset*/, uint64_t note_id) {
                midi_voices_[slot].phase = 0.0;
                midi_voices_[slot].white_noise.state =
                    12345u + static_cast<uint32_t>(note_id ? note_id : (slot + 1)) * 1664525u;
                vivid::adsr::gate_on(midi_voices_[slot].env);
            },
            [this](int slot, int /*note*/, uint64_t /*note_id*/) {
                vivid::adsr::gate_off(midi_voices_[slot].env);
            },
            [](int /*slot*/, VividNoteEventType /*kind*/, float /*value*/) {});

        // Sort active slots by note_id — voices_out channels and the four
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

        const float dt = 1.0f / sr;
        const float p_amp_depth   = pressure_to_amp.value;
        const float t_level_depth = timbre_to_level.value;

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
                float pitch_off = (pitch_mod_buf && pitch_mod_ch > 0) ? pitch_mod_buf[s] : 0.0f;
                float sub_freq = (voice_freq / sub_div) *
                                 std::pow(2.0f, pitch_off / 12.0f);
                if (!std::isfinite(sub_freq) || sub_freq <= 0.0f)
                    sub_freq = voice_freq / sub_div;

                float sig = render_voice_sample(wave, vs.phase, vs.white_noise);
                float velocity_gain = (1.0f - v2l) + v2l * slot.velocity;
                // Per-note expression: pressure scales amplitude; timbre
                // shifts the sub's level (signed depth × slot.timbre).
                const float pressure_scale = 1.0f + p_amp_depth * slot.pressure;
                const float lvl_voice = std::clamp(
                    lvl + t_level_depth * slot.timbre, 0.0f, 2.0f);
                const float voice_sample =
                    sig * lvl_voice * velocity_gain * vs.env.env_value * pressure_scale;
                sample += voice_sample;

                if (voices_buf && slot_to_pos[v] >= 0) {
                    voices_buf[slot_to_pos[v] * frames + s] = voice_sample;
                }

                vs.phase += static_cast<double>(sub_freq) / sr;
                if (vs.phase >= 1.0) vs.phase -= 1.0;
                if (!std::isfinite(vs.phase)) vs.phase = 0.0;

                if (vs.env.stage == vivid::adsr::IDLE) slot.active = false;
            }
            if (stereo_out) {
                stereo_out[s]            = sample;
                stereo_out[frames + s]   = sample;
            }
            ++midi_frame_counter_;
        }

        // Emit voice_* breakouts in note_id-sorted order.
        // ctx->value_outputs[] is indexed by overall OUTPUT port position.
        // Output port order: output(0), voices_out(1), voice_ids(2),
        // voice_gates(3), voice_velocities(4), voice_freqs(5).
        if (ctx->value_outputs) {
            VividValueOutput lanes[vivid_sequencers::kVoiceBreakoutLaneCount] = {
                ctx->value_outputs[2], ctx->value_outputs[3],
                ctx->value_outputs[4], ctx->value_outputs[5],
            };
            vivid_sequencers::emit_voice_breakouts(midi_allocator_, lanes);
        }
    }

    void process_audio(const VividAudioContext* ctx) override {
        uint32_t frames = ctx->buffer_size;
        float sr = static_cast<float>(ctx->sample_rate);

        float lvl     = level.value;
        float v2l     = velocity_to_level.value;
        float sub_div = (octave.int_value() == 1) ? 4.0f : 2.0f;
        int   wave    = waveform.int_value();

        process_audio_midi(ctx, frames, sr, lvl, v2l, sub_div, wave);
    }
};

VIVID_REGISTER(SubOsc)
VIVID_THUMBNAIL(SubOsc)
