#include "operator_api/operator.h"
#include "operator_api/midi_types.h"
#include "operator_api/thumbnail.h"
#include "operator_api/draw_plot_helpers.h"
#include "operator_api/type_id.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>

// =============================================================================
// PolyVoiceAllocator — converts MIDI / lane inputs into polyphonic lane arrays
// =============================================================================

/**
 * @brief Converts note/gate inputs into stable polyphonic voice lanes.
 *
 * Accepts either explicit note/velocity/gate lane arrays or MIDI input and turns
 * them into a bounded set of stable voice lanes for downstream oscillators,
 * filters, envelopes, and mixers. Emits note pitch, velocity, gate, frequency,
 * and stable lane identifiers so per-note state can persist across the voice path.
 *
 * @input notes_in Note numbers as a lane array.
 * @input velocities_in Velocity values aligned with notes_in.
 * @input gates_in Gate values aligned with notes_in.
 * @input midi_in Optional MIDI note stream input.
 * @output notes Active note numbers per allocated voice lane.
 * @output velocities Active velocities per allocated voice lane.
 * @output gates Active gates per allocated voice lane.
 * @output frequencies Active frequencies per allocated voice lane.
 * @output lane_ids Stable per-voice identity tokens for downstream lane state.
 * @tip Use lane_ids with oscillators so each voice keeps stable state across buffers and retriggers.
 * @recipe ChordProgressionAu/notes,velocities,gates -> PolyVoiceAllocator/notes_in,velocities_in,gates_in
 * @recipe PolyVoiceAllocator/frequencies,gates,lane_ids -> WavetableOsc/frequencies,gates,lane_ids
 * @pitfall Downstream per-note operators should stay lane-aware until the final reduction stage; do not collapse lanes before envelopes and oscillators have consumed them.
 * @family note_source
 * @best_used_with ChordProgressionAu, EnvelopeAu, VoiceMixer
 * @common_companions WavetableOsc, AnalogOsc, SubOsc, Filter
 */
struct PolyVoiceAllocator : vivid::OperatorBase, vivid::AudioProcessable {
    static constexpr const char* kName   = "PolyVoiceAllocator";
    static constexpr bool kTimeDependent = true;
    static constexpr VividLaneBehavior kLaneBehavior = VIVID_LANE_STRUCTURAL;

    static constexpr int kMaxVoices = 16;

    // --- Parameters ---
    vivid::Param<int>   max_voices {"max_voices", 8, 1, 16};
    vivid::Param<float> portamento {"portamento", 0.0f, 0.0f, 2000.0f};

    // --- Voice state ---
    struct Voice {
        float    note       = 0;
        float    velocity   = 0;
        float    freq       = 0;
        int      gate_slot  = -1;
        uint64_t note_id    = 0;
        uint32_t lane_id    = 0;   // stable identity for this voice
        bool     active     = false;
        bool     releasing  = false;
        uint32_t release_buffers = 0;  // buffers since gate-off (for release-tail retention)
        uint32_t release_hold_buffers = 0;
    };

    Voice    voices_[kMaxVoices] = {};
    uint64_t note_counter_       = 0;
    const VividAudioContext* cur_ctx_ = nullptr;  // set during process_audio

    // Gate edge detection for lane passthrough
    float    prev_gates_[kMaxVoices] = {};
    float    prev_notes_[kMaxVoices] = {};
    uint32_t prev_lane_len_          = 0;

    // MIDI voice allocation
    static constexpr int kMidiSlotBase = 128;
    struct MidiVoiceEntry {
        uint8_t note    = 0;
        bool    active  = false;
        int     slot    = -1;
    };
    MidiVoiceEntry midi_voices_[kMaxVoices] = {};

    static constexpr float kLaneReleaseHoldMs = 2000.0f;
    static constexpr uint32_t kMidiReleaseHoldBuffers = 375;

    static uint32_t release_hold_buffers_for_ms(const VividAudioContext* ctx, float ms) {
        if (!ctx || ctx->sample_rate == 0 || ctx->buffer_size == 0) return 1;
        double release_frames = static_cast<double>(ms) * 0.001 * static_cast<double>(ctx->sample_rate);
        return std::max<uint32_t>(1, static_cast<uint32_t>(
            std::ceil(release_frames / static_cast<double>(ctx->buffer_size))));
    }

    PolyVoiceAllocator() {
        vivid::semantic_tag(portamento, "time_milliseconds");
        vivid::semantic_unit(portamento, "ms");
    }

    // --- Param / port registration ---

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        out.push_back(&max_voices);
        out.push_back(&portamento);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        // Inputs
        out.push_back({"notes_in",      VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});   // 0
        out.push_back({"velocities_in", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});   // 1
        out.push_back({"gates_in",      VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});   // 2
        out.push_back(VIVID_CUSTOM_REF_PORT("midi_in", VIVID_PORT_INPUT, VividMidiBuffer)); // 3
        // Outputs
        out.push_back({"notes",       VIVID_PORT_LANE_ARRAY, VIVID_PORT_OUTPUT});    // out 0
        out.push_back({"velocities",  VIVID_PORT_LANE_ARRAY, VIVID_PORT_OUTPUT});    // out 1
        out.push_back({"gates",       VIVID_PORT_LANE_ARRAY, VIVID_PORT_OUTPUT});    // out 2
        out.push_back({"frequencies", VIVID_PORT_LANE_ARRAY, VIVID_PORT_OUTPUT});    // out 3
        out.push_back({"lane_ids",    VIVID_PORT_LANE_ARRAY, VIVID_PORT_OUTPUT});    // out 4
    }

    // --- Helpers ---

    static float midi_to_freq(float note) {
        return 440.0f * std::pow(2.0f, (note - 69.0f) / 12.0f);
    }

    static float read_lane_slot(const VividLaneView* sp, int slot, float fallback = 0.0f) {
        if (sp && sp->data && slot >= 0 && static_cast<uint32_t>(slot) < sp->length)
            return sp->data[slot];
        return fallback;
    }

    uint32_t voice_limit() const {
        return static_cast<uint32_t>(std::clamp(max_voices.int_value(), 1, kMaxVoices));
    }

    void deactivate_voice(int idx, const VividAudioContext* ctx) {
        Voice& v = voices_[idx];
        if (v.active && v.lane_id != 0 && ctx && ctx->retire_lane_id_fn)
            ctx->retire_lane_id_fn(ctx->lane_state_service, v.lane_id);
        v = {};
    }

    void trim_to_voice_limit(const VividAudioContext* ctx, uint32_t limit) {
        for (uint32_t i = limit; i < static_cast<uint32_t>(kMaxVoices); ++i)
            deactivate_voice(static_cast<int>(i), ctx);
        for (uint32_t i = limit; i < static_cast<uint32_t>(kMaxVoices); ++i)
            midi_voices_[i] = {};
    }

    // --- Voice management ---

    int find_free_voice(uint32_t limit) const {
        for (uint32_t i = 0; i < limit; ++i)
            if (!voices_[i].active) return i;
        return -1;
    }

    int find_voice_to_steal(uint32_t limit) const {
        int releasing_idx = -1;
        uint64_t releasing_oldest = UINT64_MAX;
        for (uint32_t i = 0; i < limit; ++i) {
            if (voices_[i].active && voices_[i].releasing && voices_[i].note_id < releasing_oldest) {
                releasing_oldest = voices_[i].note_id;
                releasing_idx = static_cast<int>(i);
            }
        }
        if (releasing_idx >= 0) return releasing_idx;

        int idx = -1;
        uint64_t oldest = UINT64_MAX;
        for (uint32_t i = 0; i < limit; ++i) {
            if (voices_[i].active && voices_[i].note_id < oldest) {
                oldest = voices_[i].note_id;
                idx = i;
            }
        }
        return idx;
    }

    void trigger_note_on(float note, float vel, int slot, float porta_ms,
                         uint32_t release_hold_buffers, uint32_t limit) {
        // With portamento, reuse existing voice for this slot
        if (porta_ms > 0.0f) {
            for (uint32_t i = 0; i < limit; ++i) {
                if (voices_[i].active && !voices_[i].releasing &&
                    voices_[i].gate_slot == slot) {
                    // Glide: update note/freq, don't retrigger
                    voices_[i].note = note;
                    voices_[i].velocity = vel;
                    voices_[i].freq = midi_to_freq(note);
                    voices_[i].note_id = ++note_counter_;
                    voices_[i].release_hold_buffers = release_hold_buffers;
                    return;
                }
            }
        }

        int vi = find_free_voice(limit);
        if (vi < 0) vi = find_voice_to_steal(limit);
        if (vi < 0) return;

        Voice& v = voices_[vi];
        // Retire old lane_id if stealing an active voice
        if (v.active && v.lane_id != 0 && cur_ctx_ && cur_ctx_->retire_lane_id_fn)
            cur_ctx_->retire_lane_id_fn(cur_ctx_->lane_state_service, v.lane_id);

        v.note = note;
        v.velocity = vel;
        v.freq = midi_to_freq(note);
        v.gate_slot = slot;
        v.note_id = ++note_counter_;
        v.active = true;
        v.releasing = false;
        v.release_buffers = 0;
        v.release_hold_buffers = release_hold_buffers;
        // Allocate fresh lane_id for new voice
        if (cur_ctx_ && cur_ctx_->allocate_lane_id_fn)
            v.lane_id = cur_ctx_->allocate_lane_id_fn(cur_ctx_->lane_state_service);
    }

    void trigger_note_off_slot(int slot, uint32_t limit) {
        for (uint32_t i = 0; i < limit; ++i) {
            if (voices_[i].active && !voices_[i].releasing &&
                voices_[i].gate_slot == slot) {
                voices_[i].releasing = true;
                voices_[i].release_buffers = 0;
            }
        }
    }

    void trim_releasing_voices(const VividAudioContext* ctx, uint32_t limit) {
        uint32_t sustaining_count = 0;
        uint32_t releasing_count = 0;
        for (uint32_t i = 0; i < limit; ++i) {
            if (!voices_[i].active) continue;
            if (voices_[i].releasing) {
                releasing_count++;
            } else {
                sustaining_count++;
            }
        }

        if (releasing_count == 0 || sustaining_count == 0) return;

        uint32_t releasing_budget = std::min(limit - sustaining_count, sustaining_count);
        while (releasing_count > releasing_budget) {
            int retire_idx = -1;
            uint32_t max_release_buffers = 0;
            uint64_t oldest_note_id = 0;
            for (uint32_t i = 0; i < limit; ++i) {
                const Voice& v = voices_[i];
                if (!v.active || !v.releasing) continue;
                if (retire_idx < 0 ||
                    v.release_buffers > max_release_buffers ||
                    (v.release_buffers == max_release_buffers && v.note_id < oldest_note_id)) {
                    retire_idx = static_cast<int>(i);
                    max_release_buffers = v.release_buffers;
                    oldest_note_id = v.note_id;
                }
            }
            if (retire_idx < 0) break;
            deactivate_voice(retire_idx, ctx);
            releasing_count--;
        }
    }

    // --- MIDI processing ---

    void process_midi(const VividAudioContext* ctx) {
        if (!ctx->custom_inputs || ctx->custom_input_count == 0 || !ctx->custom_inputs[0])
            return;

        auto* midi = static_cast<const VividMidiBuffer*>(ctx->custom_inputs[0]);
        float porta_ms = portamento.value;
        uint32_t limit = voice_limit();

        for (uint32_t m = 0; m < midi->count; ++m) {
            const auto& msg = midi->messages[m];
            uint8_t status = msg.status & 0xF0;

            if (status == 0x90 && msg.data2 > 0) {
                float note = static_cast<float>(msg.data1);
                float vel  = msg.data2 / 127.0f;

                int entry = -1;
                for (uint32_t i = 0; i < limit; ++i) {
                    if (midi_voices_[i].active && midi_voices_[i].note == msg.data1) {
                        entry = i; break;
                    }
                }
                if (entry < 0) {
                    for (uint32_t i = 0; i < limit; ++i) {
                        if (!midi_voices_[i].active) { entry = i; break; }
                    }
                }
                if (entry < 0) {
                    entry = 0;
                    trigger_note_off_slot(midi_voices_[0].slot, limit);
                    midi_voices_[0].active = false;
                }

                int slot = kMidiSlotBase + entry;
                if (midi_voices_[entry].active && midi_voices_[entry].note == msg.data1)
                    trigger_note_off_slot(slot, limit);

                midi_voices_[entry].note   = msg.data1;
                midi_voices_[entry].active = true;
                midi_voices_[entry].slot   = slot;
                trigger_note_on(note, vel, slot, porta_ms, kMidiReleaseHoldBuffers, limit);

            } else if (status == 0x80 || (status == 0x90 && msg.data2 == 0)) {
                for (uint32_t i = 0; i < limit; ++i) {
                    if (midi_voices_[i].active && midi_voices_[i].note == msg.data1) {
                        trigger_note_off_slot(midi_voices_[i].slot, limit);
                        midi_voices_[i].active = false;
                        break;
                    }
                }
            }
        }
    }

    // --- Gate passthrough processing ---

    void update_gates(const VividAudioContext* ctx) {
        if (!ctx->input_lanes) return;

        const auto& notes_lane = ctx->input_lanes[0];
        const auto& vel_lane   = ctx->input_lanes[1];
        const auto& gates_lane = ctx->input_lanes[2];

        uint32_t limit = voice_limit();
        uint32_t len = std::min(gates_lane.length, limit);

        float porta_ms = portamento.value;
        uint32_t lane_release_hold_buffers = release_hold_buffers_for_ms(ctx, kLaneReleaseHoldMs);

        for (uint32_t i = 0; i < len; ++i) {
            float cur_gate = read_lane_slot(&gates_lane, static_cast<int>(i));
            float cur_note = read_lane_slot(&notes_lane, static_cast<int>(i));
            float cur_vel  = read_lane_slot(&vel_lane,   static_cast<int>(i), 0.8f);

            float prev_gate = (i < prev_lane_len_) ? prev_gates_[i] : 0.0f;
            float prev_note = (i < prev_lane_len_) ? prev_notes_[i] : 0.0f;

            bool on        = (cur_gate > 0.5f) && (prev_gate <= 0.5f);
            bool off       = (cur_gate <= 0.5f) && (prev_gate > 0.5f);
            bool retrigger = (cur_gate > 0.5f) && (prev_gate > 0.5f) &&
                             (std::abs(cur_note - prev_note) > 0.5f);

            if (on || retrigger) {
                if (retrigger && porta_ms <= 0.0f)
                    trigger_note_off_slot(static_cast<int>(i), limit);
                trigger_note_on(cur_note, cur_vel, static_cast<int>(i),
                                retrigger ? porta_ms : 0.0f,
                                lane_release_hold_buffers, limit);
            } else if (off) {
                trigger_note_off_slot(static_cast<int>(i), limit);
            }

            prev_gates_[i] = cur_gate;
            prev_notes_[i] = cur_note;
        }

        for (uint32_t i = len; i < prev_lane_len_; ++i) {
            if (prev_gates_[i] > 0.5f)
                trigger_note_off_slot(static_cast<int>(i), limit);
            prev_gates_[i] = 0.0f;
            prev_notes_[i] = 0.0f;
        }

        prev_lane_len_ = len;
    }

    // --- Thumbnail ---

    void draw_thumbnail(const VividThumbnailContext* ctx) override {
        if (!ctx || !ctx->draw.opaque) return;
        auto& d = const_cast<VividDrawAPI&>(ctx->draw);
        void* o = d.opaque;

        float w = static_cast<float>(ctx->thumbnail_logical_width ? ctx->thumbnail_logical_width : ctx->thumbnail_width);
        float h = static_cast<float>(ctx->thumbnail_logical_height ? ctx->thumbnail_logical_height : ctx->thumbnail_height);

        int max_v = (ctx->param_count > 0) ? std::clamp(static_cast<int>(ctx->param_values[0]), 1, 16) : 8;

        vivid::draw_plot::draw_thumb_background(d, o, w, h);

        // Voice count label
        char label[16];
        std::snprintf(label, sizeof(label), "%d VOICES", max_v);
        vivid::draw_plot::draw_thumb_label(d, o, 6.0f, 4.0f, label, {0.55f, 0.65f, 0.55f, 0.9f}, 0.8f);

        // Voice slots grid
        float grid_top = 22.0f;
        float grid_h = h - grid_top - 6.0f;
        float pad = 6.0f;
        float gap = 3.0f;
        int cols = max_v <= 8 ? max_v : 8;
        int rows = (max_v + cols - 1) / cols;
        float cell_w = (w - 2.0f * pad - (cols - 1) * gap) / static_cast<float>(cols);
        float cell_h = (grid_h - (rows - 1) * gap) / static_cast<float>(rows);
        cell_w = std::min(cell_w, cell_h);
        cell_h = cell_w;

        float total_w = cols * cell_w + (cols - 1) * gap;
        float total_h = rows * cell_h + (rows - 1) * gap;
        float start_x = (w - total_w) * 0.5f;
        float start_y = grid_top + (grid_h - total_h) * 0.5f;

        for (int i = 0; i < max_v; ++i) {
            int col = i % cols;
            int row = i / cols;
            float cx = start_x + col * (cell_w + gap);
            float cy = start_y + row * (cell_h + gap);
            VividColor slot_col = {0.35f, 0.50f, 0.40f, 0.7f};
            if (d.draw_rounded_rect)
                d.draw_rounded_rect(o, cx, cy, cell_w, cell_h, 2.0f, slot_col);
            else if (d.draw_rect)
                d.draw_rect(o, cx, cy, cell_w, cell_h, slot_col);
        }
    }

    // --- Main process ---

    void process_audio(const VividAudioContext* ctx) override {
        cur_ctx_ = ctx;
        uint32_t limit = voice_limit();
        trim_to_voice_limit(ctx, limit);
        process_midi(ctx);
        update_gates(ctx);
        trim_releasing_voices(ctx, limit);

        // Write output lanes
        if (!ctx->output_lanes) {
            cur_ctx_ = nullptr;
            return;
        }

        auto& notes_out   = ctx->output_lanes[0];
        auto& vel_out     = ctx->output_lanes[1];
        auto& gates_out   = ctx->output_lanes[2];
        auto& freq_out    = ctx->output_lanes[3];
        auto& lane_id_out = ctx->output_lanes[4];

        // Pre-count active voices
        uint32_t count = 0;
        for (uint32_t vi = 0; vi < limit; ++vi) {
            if (voices_[vi].active) count++;
        }

        // Resize all lane outputs and fill
        float* notes_buf   = notes_out.resize(notes_out.handle, count);
        float* vel_buf     = vel_out.resize(vel_out.handle, count);
        float* gates_buf   = gates_out.resize(gates_out.handle, count);
        float* freq_buf    = freq_out.resize(freq_out.handle, count);
        float* lane_id_buf = lane_id_out.resize(lane_id_out.handle, count);

        uint32_t idx = 0;
        for (uint32_t vi = 0; vi < limit && idx < count; ++vi) {
            if (!voices_[vi].active) continue;
            if (notes_buf)   notes_buf[idx]   = voices_[vi].note;
            if (vel_buf)     vel_buf[idx]     = voices_[vi].velocity;
            if (gates_buf)   gates_buf[idx]   = voices_[vi].releasing ? 0.0f : 1.0f;
            if (freq_buf)    freq_buf[idx]    = voices_[vi].freq;
            if (lane_id_buf) lane_id_buf[idx] = static_cast<float>(voices_[vi].lane_id);
            idx++;
        }

        notes_out.commit(notes_out.handle, count);
        vel_out.commit(vel_out.handle, count);
        gates_out.commit(gates_out.handle, count);
        freq_out.commit(freq_out.handle, count);
        lane_id_out.commit(lane_id_out.handle, count);

        // Keep a short per-voice gate-off handoff for lane-driven graphs,
        // while preserving longer MIDI tails for interactive note input.
        for (uint32_t i = 0; i < limit; ++i) {
            if (voices_[i].releasing) {
                voices_[i].release_buffers++;
                if (voices_[i].release_buffers >= voices_[i].release_hold_buffers) {
                    // Release tail complete — retire lane_id and deactivate.
                    deactivate_voice(static_cast<int>(i), ctx);
                }
            }
        }
        cur_ctx_ = nullptr;
    }
};

VIVID_REGISTER(PolyVoiceAllocator)
VIVID_THUMBNAIL(PolyVoiceAllocator)
