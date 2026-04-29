#include "operator_api/operator.h"
#include "operator_api/adsr.h"
#include "operator_api/note_types.h"
#include "operator_api/thumbnail.h"
#include "operator_api/draw_plot_helpers.h"
#include "operator_api/type_id.h"
#include "operator_api/voice_table.h"
#include "voice_breakouts.h"
#include "wavetable_dsp.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr float TWO_PI_F = 2.0f * static_cast<float>(M_PI);

using namespace vivid_wavetable::dsp;

// =============================================================================
// AnalogOsc — polyphonic virtual analog oscillator with PolyBLEP anti-aliasing
// =============================================================================

/**
 * @brief Polyphonic virtual analog oscillator with anti-aliased classic waveforms.
 *
 * Drive with `notes_in` from any note source (Tracker, NotePattern,
 * Sequencer, ChordProgression, ...) — voices are allocated internally with
 * built-in ADSR and summed into a stereo output. Advanced `voices_out`
 * (per-voice multichannel) plus the four `voice_*` control lanes expose
 * per-voice state for downstream VoiceMixer/VoiceDrive/Filter routing.
 *
 * @input notes_in Native note stream — canonical input for note sources.
 * @input mod_input Audio-rate modulation input for oscillator interaction.
 * @input pitch_mod_audio Audio-rate per-voice pitch modulation.
 * @output output Stereo summed audio.
 * @recipe Tracker/notes_out -> AnalogOsc/notes_in
 * @recipe ChordProgression/notes_out -> AnalogOsc/notes_in
 * @recipe AnalogOsc/output -> audio_out/input
 * @pitfall Interaction belongs on the carrier oscillator; keep the modulator in the graph and feed its output into mod_input instead of expecting VoiceMixer-stage interaction.
 * @family voice_source
 * @best_used_with Tracker, ChordProgression, NotePattern, Filter
 * @common_companions FmSynth, Sampler, WavetableOsc
 */
struct AnalogOsc : vivid::OperatorBase, vivid::AudioProcessable {
    static constexpr const char* kName   = "AnalogOsc";
    static constexpr bool kTimeDependent = true;

    static constexpr int kMaxVoices = 16;

    enum Waveform { WAVE_SINE, WAVE_SAW, WAVE_SQUARE, WAVE_TRIANGLE, WAVE_PULSE };
    enum InteractionMode  { INTERACTION_OFF, INTERACTION_FM, INTERACTION_PM, INTERACTION_RM, INTERACTION_AM };

    // --- Parameters ---
    vivid::Param<int>   waveform    {"waveform",    1,    {"Sine", "Saw", "Square", "Triangle", "Pulse"}};
    vivid::Param<float> pulse_width {"pulse_width", 0.5f, 0.01f, 0.99f};
    vivid::Param<float> amplitude   {"amplitude",   0.3f, 0.0f,  1.0f};
    vivid::Param<float> detune      {"detune",      0.0f, 0.0f,  50.0f};
    vivid::Param<float> portamento  {"portamento",  0.0f, 0.0f,  2000.0f};
    vivid::Param<int>   interaction_mode       {"interaction_mode", 0, {"Off", "FM", "PM", "RM", "AM"}};
    vivid::Param<float> interaction_depth      {"interaction_depth", 0.0f, 0.0f, 1.0f};
    vivid::Param<float> interaction_input_gain {"interaction_input_gain", 1.0f, 0.0f, 4.0f};
    vivid::Param<float> interaction_tracking   {"interaction_tracking", 1.0f, 0.0f, 1.0f};
    // ADSR for the MIDI-driven path. Lane-array driven graphs typically
    // run their envelope upstream (e.g., via VoiceMixer's amp_env_audio).
    vivid::Param<float> attack  {"attack",  0.005f, 0.001f, 5.0f};
    vivid::Param<float> decay   {"decay",   0.1f,   0.001f, 5.0f};
    vivid::Param<float> sustain {"sustain", 0.8f,   0.0f,   1.0f};
    vivid::Param<float> release {"release", 0.2f,   0.001f, 5.0f};

    // Per-note expression depth (Phase 5). Pressure scales per-voice
    // amplitude; timbre offsets pulse_width (signed) so X-axis MPE
    // movement opens or closes the duty cycle.
    vivid::Param<float> pressure_to_amp {"pressure_to_amp", 0.5f,  0.0f, 1.0f};
    vivid::Param<float> timbre_to_pwm   {"timbre_to_pwm",   0.3f, -1.0f, 1.0f};

    // --- MIDI-driven path state ---
    struct MidiVoice {
        double phase        = 0.0;
        double current_freq = 0.0;  // for portamento glide
        double target_freq  = 0.0;
        vivid::adsr::State env;
        DCBlocker interaction_dc;
    };
    MidiVoice midi_voices_[kMaxVoices] = {};
    vivid::VoiceTable<kMaxVoices> midi_allocator_;
    uint64_t midi_frame_counter_ = 0;

    AnalogOsc() {
        vivid::semantic_tag(amplitude, "amplitude_linear");
        vivid::semantic_tag(portamento, "time_milliseconds");
        vivid::semantic_unit(portamento, "ms");
        vivid::semantic_tag(interaction_input_gain, "gain_linear");
    }

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        param_group(waveform,    "Core");
        param_group(pulse_width, "Core");
        param_group(amplitude,   "Core");
        param_group(detune,      "Tuning");
        param_group(portamento,  "Tuning");
        param_group(attack,  "Envelope");
        param_group(decay,   "Envelope");
        param_group(sustain, "Envelope");
        param_group(release, "Envelope");
        param_group(interaction_mode, "Interaction");
        param_group(interaction_depth, "Interaction");
        param_group(interaction_input_gain, "Interaction");
        param_group(interaction_tracking, "Interaction");

        out.push_back(&waveform);
        out.push_back(&pulse_width);
        out.push_back(&amplitude);
        out.push_back(&detune);
        out.push_back(&portamento);
        out.push_back(&attack);
        out.push_back(&decay);
        out.push_back(&sustain);
        out.push_back(&release);
        out.push_back(&interaction_mode);
        out.push_back(&interaction_depth);
        out.push_back(&interaction_input_gain);
        out.push_back(&interaction_tracking);
        param_group(pressure_to_amp, "Expression");
        param_group(timbre_to_pwm,   "Expression");
        out.push_back(&pressure_to_amp);
        out.push_back(&timbre_to_pwm);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        // Canonical native note input — drive directly from Tracker/NotePattern/etc.
        out.push_back(VIVID_CUSTOM_REF_PORT("notes_in", VIVID_PORT_INPUT, VividNoteBuffer)); // 0
        // N-channel audio modulation input (for FM/RM/AM from another osc)
        out.push_back({"mod_input", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});     // 1
        // Audio-rate pitch modulation (N-channel, one per voice)
        out.push_back({"pitch_mod_audio", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});     // 2

        // Primary stereo output — sum of all active voices.
        out.push_back({"output", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_OUTPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 2});

        // Advanced per-voice breakouts. voices_out is the multichannel
        // buffer (one channel per voice) that previously lived on `output`.
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

    // --- Helpers ---

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

    // Generate one sample of the configured waveform. Pure helper, used by
    // both the lane-driven and MIDI-driven paths.
    static float render_waveform(int wave, double phase_sample, double phase_inc, float pw) {
        switch (wave) {
            case WAVE_SINE:
                return std::sin(static_cast<float>(phase_sample) * TWO_PI_F);
            case WAVE_SAW:
                return generate_saw(phase_sample, phase_inc);
            case WAVE_SQUARE:
                return generate_square(phase_sample, phase_inc);
            case WAVE_TRIANGLE:
                return generate_triangle(phase_sample, phase_inc);
            case WAVE_PULSE:
                return generate_pulse(phase_sample, phase_inc, pw);
            default:
                return 0.0f;
        }
    }

    // MIDI-driven rendering: internal allocator + ADSR + summed stereo output.
    // Channels 2..kMaxVoices-1 are zeroed. Lane-array inputs are ignored on
    // this path — by definition voice_count == 0 means no lane-array source.
    void process_audio_midi(const VividAudioContext* ctx, uint32_t frames, float sr,
                            int wave, float pw, float amp, float det, float porta_ms,
                            int interaction, float interaction_depth_value,
                            float interaction_input_gain_value,
                            float interaction_tracking_value) {
        const auto* notes = static_cast<const VividNoteBuffer*>(ctx->custom_inputs[0]);

        // Audio-rate modulation buffers (mono-summed across all voices).
        // Port indices: notes_in=0, mod_input=1, pitch_mod_audio=2.
        float* mod_buf = (interaction > INTERACTION_OFF && interaction_depth_value > 0.0f
                          && ctx->input_buffers && ctx->input_buffers[1])
                         ? ctx->input_buffers[1] : nullptr;
        float* pitch_mod_buf = ctx->input_buffers ? ctx->input_buffers[2] : nullptr;

        // Process note events into the allocator. Trigger envelopes + reset
        // phase on note-on; release envelopes on note-off. Per-note
        // expression is recorded on the matching slot for later phases.
        midi_allocator_.process_note_buffer(notes, midi_frame_counter_,
            [this](int slot, int /*note*/, float /*vel*/, uint32_t /*offset*/, uint64_t /*note_id*/) {
                midi_voices_[slot].phase = 0.0;
                midi_voices_[slot].interaction_dc.reset();
                vivid::adsr::gate_on(midi_voices_[slot].env);
            },
            [this](int slot, int /*note*/, uint64_t /*note_id*/) {
                vivid::adsr::gate_off(midi_voices_[slot].env);
            },
            [](int /*slot*/, VividNoteEventType /*kind*/, float /*value*/) {});

        // Portamento rate
        float porta_rate = 1.0f;
        if (porta_ms > 0.0f) {
            float porta_samples = porta_ms * 0.001f * sr;
            porta_rate = 1.0f - std::exp(-4.0f / porta_samples);
        }

        // Update target frequencies for active slots from their current note,
        // including per-note pitch_bend (consumed by the allocator into slot
        // state on PITCH_BEND events).
        for (int v = 0; v < kMaxVoices; ++v) {
            auto& slot = midi_allocator_.slots[v];
            if (!slot.active) continue;
            auto& vs = midi_voices_[v];
            const float voice_freq = 440.0f *
                std::pow(2.0f, (static_cast<float>(slot.note) - 69.0f
                                + slot.pitch_bend_semis) / 12.0f);
            vs.target_freq = voice_freq * cents_to_ratio(det);
            if (vs.current_freq <= 0.0) vs.current_freq = vs.target_freq;
        }

        // Sort active slots by note_id so voices_out channels and the four
        // voice_* breakout lanes line up across operators (cross-cutting #4).
        int sorted_idx[kMaxVoices];
        int slot_to_pos[kMaxVoices];
        int n_active = 0;
        for (int v = 0; v < kMaxVoices; ++v) {
            slot_to_pos[v] = -1;
            if (midi_allocator_.slots[v].active) sorted_idx[n_active++] = v;
        }
        std::sort(sorted_idx, sorted_idx + n_active,
                  [this](int a, int b) {
                      return midi_allocator_.slots[a].note_id <
                             midi_allocator_.slots[b].note_id;
                  });
        for (int i = 0; i < n_active; ++i) slot_to_pos[sorted_idx[i]] = i;

        // Stereo output (port 0) and multichannel voices_out (port 1).
        float* stereo_out  = ctx->output_buffers && ctx->output_buffers[0]
                             ? ctx->output_buffers[0] : nullptr;
        float* voices_buf  = ctx->output_buffers && ctx->output_buffers[1]
                             ? ctx->output_buffers[1] : nullptr;
        if (stereo_out) std::memset(stereo_out, 0, 2 * frames * sizeof(float));
        if (voices_buf) std::memset(voices_buf, 0, kMaxVoices * frames * sizeof(float));

        const float dt = 1.0f / sr;
        const float p_amp_depth = pressure_to_amp.value;
        const float t_pwm_depth = timbre_to_pwm.value;

        for (uint32_t s = 0; s < frames; ++s) {
            float sample = 0.0f;
            for (int v = 0; v < kMaxVoices; ++v) {
                auto& slot = midi_allocator_.slots[v];
                if (!slot.active) continue;
                auto& vs = midi_voices_[v];

                vivid::adsr::advance(vs.env, dt, attack.value, decay.value,
                                     sustain.value, release.value);

                if (porta_ms > 0.0f && vs.current_freq != vs.target_freq) {
                    vs.current_freq += (vs.target_freq - vs.current_freq) * porta_rate;
                    if (std::abs(vs.current_freq - vs.target_freq) < 0.01)
                        vs.current_freq = vs.target_freq;
                }

                float pitch_off = pitch_mod_buf ? pitch_mod_buf[s] : 0.0f;
                float freq = static_cast<float>(vs.current_freq) *
                    std::pow(2.0f, pitch_off / 12.0f);
                if (!std::isfinite(freq) || freq <= 0.0f)
                    freq = static_cast<float>(vs.current_freq);

                double phase_inc = static_cast<double>(freq) / sr;
                InteractionSignal interaction_signal = prepare_interaction_signal(
                    interaction, freq, interaction_depth_value, interaction_input_gain_value,
                    interaction_tracking_value, mod_buf ? mod_buf[s] : 0.0f, mod_buf != nullptr,
                    vs.interaction_dc);
                if (interaction == INTERACTION_FM && mod_buf) {
                    phase_inc += static_cast<double>(interaction_fm_phase_delta(interaction_signal, sr));
                }
                double phase_sample = vs.phase;
                if (interaction == INTERACTION_PM && mod_buf) {
                    phase_sample += static_cast<double>(interaction_pm_offset(interaction_signal));
                    phase_sample -= std::floor(phase_sample);
                }

                // Per-note timbre (X-axis MPE) offsets pulse width — bright
                // duty when timbre rises, narrow when it falls. Clamped into
                // the param's safe range.
                const float pw_voice = std::clamp(
                    pw + t_pwm_depth * slot.timbre, 0.01f, 0.99f);
                float sig = render_waveform(wave, phase_sample, phase_inc, pw_voice);
                if (mod_buf && interaction > INTERACTION_OFF) {
                    if (interaction == INTERACTION_RM)
                        sig = interaction_rm_sample(sig, interaction_signal);
                    else if (interaction == INTERACTION_AM)
                        sig *= interaction_am_gain(interaction_signal);
                    sig *= interaction_output_compensation(interaction, interaction_signal.amount);
                }

                // Per-note pressure scales voice amplitude.
                const float pressure_scale = 1.0f + p_amp_depth * slot.pressure;
                const float voice_sample = sig * vs.env.env_value * slot.velocity * pressure_scale;
                sample += voice_sample;

                // Mirror to voices_out at this voice's note_id-sorted rank,
                // applying the master amplitude so breakout audio matches
                // what the stereo `output` carries.
                if (voices_buf && slot_to_pos[v] >= 0) {
                    voices_buf[slot_to_pos[v] * frames + s] = voice_sample * amp;
                }

                vs.phase += phase_inc;
                if (vs.phase >= 1.0) vs.phase -= 1.0;
                if (vs.phase < 0.0) vs.phase += 1.0;
                if (!std::isfinite(vs.phase)) vs.phase = 0.0;

                if (vs.env.stage == vivid::adsr::IDLE) slot.active = false;
            }
            sample *= amp;
            if (stereo_out) {
                stereo_out[s]            = sample;
                stereo_out[frames + s]   = sample;
            }
            ++midi_frame_counter_;
        }

        // Emit the four voice_* control breakouts in note_id-sorted order.
        // ctx->output_lanes[] is indexed by overall OUTPUT port position.
        // Output port order: output(0), voices_out(1), voice_ids(2),
        // voice_gates(3), voice_velocities(4), voice_freqs(5).
        if (ctx->output_lanes) {
            VividLaneOutput lanes[vivid_sequencers::kVoiceBreakoutLaneCount] = {
                ctx->output_lanes[2], ctx->output_lanes[3],
                ctx->output_lanes[4], ctx->output_lanes[5],
            };
            vivid_sequencers::emit_voice_breakouts(midi_allocator_, lanes);
        }
    }

    void process_audio(const VividAudioContext* ctx) override {
        uint32_t frames = ctx->buffer_size;
        float sr = static_cast<float>(ctx->sample_rate);

        int   wave = waveform.int_value();
        float pw = pulse_width.value;
        float amp = amplitude.value;
        float det = detune.value;
        float porta_ms = portamento.value;
        int interaction = interaction_mode.int_value();
        float interaction_depth_value = interaction_depth.value;
        float interaction_input_gain_value = interaction_input_gain.value;
        float interaction_tracking_value = interaction_tracking.value;

        process_audio_midi(ctx, frames, sr, wave, pw, amp, det, porta_ms,
                           interaction, interaction_depth_value,
                           interaction_input_gain_value,
                           interaction_tracking_value);
    }
};

VIVID_REGISTER(AnalogOsc)
VIVID_THUMBNAIL(AnalogOsc)
