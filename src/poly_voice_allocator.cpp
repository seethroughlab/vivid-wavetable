#include "operator_api/operator.h"
#include "operator_api/midi_types.h"
#include "operator_api/type_id.h"
#include <cmath>
#include <cstring>
#include <algorithm>

// =============================================================================
// PolyVoiceAllocator — converts MIDI / spread inputs into polyphonic spreads
// =============================================================================

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
    };

    Voice    voices_[kMaxVoices] = {};
    uint64_t note_counter_       = 0;
    const VividAudioContext* cur_ctx_ = nullptr;  // set during process_audio

    // Gate edge detection for spread passthrough
    float    prev_gates_[kMaxVoices] = {};
    float    prev_notes_[kMaxVoices] = {};
    uint32_t prev_spread_len_        = 0;

    // MIDI voice allocation
    static constexpr int kMidiSlotBase = 128;
    struct MidiVoiceEntry {
        uint8_t note    = 0;
        bool    active  = false;
        int     slot    = -1;
    };
    MidiVoiceEntry midi_voices_[kMaxVoices] = {};

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
        out.push_back({"notes_in",      VIVID_PORT_SPREAD, VIVID_PORT_INPUT});   // 0
        out.push_back({"velocities_in", VIVID_PORT_SPREAD, VIVID_PORT_INPUT});   // 1
        out.push_back({"gates_in",      VIVID_PORT_SPREAD, VIVID_PORT_INPUT});   // 2
        out.push_back(VIVID_CUSTOM_REF_PORT("midi_in", VIVID_PORT_INPUT, VividMidiBuffer)); // 3
        // Outputs
        out.push_back({"notes",       VIVID_PORT_SPREAD, VIVID_PORT_OUTPUT});    // out 0
        out.push_back({"velocities",  VIVID_PORT_SPREAD, VIVID_PORT_OUTPUT});    // out 1
        out.push_back({"gates",       VIVID_PORT_SPREAD, VIVID_PORT_OUTPUT});    // out 2
        out.push_back({"frequencies", VIVID_PORT_SPREAD, VIVID_PORT_OUTPUT});    // out 3
        out.push_back({"lane_ids",    VIVID_PORT_SPREAD, VIVID_PORT_OUTPUT});    // out 4
    }

    // --- Helpers ---

    static float midi_to_freq(float note) {
        return 440.0f * std::pow(2.0f, (note - 69.0f) / 12.0f);
    }

    static float read_spread_slot(const VividSpreadPort* sp, int slot, float fallback = 0.0f) {
        if (sp && sp->data && slot >= 0 && static_cast<uint32_t>(slot) < sp->length)
            return sp->data[slot];
        return fallback;
    }

    // --- Voice management ---

    int find_free_voice() const {
        for (int i = 0; i < kMaxVoices; ++i)
            if (!voices_[i].active) return i;
        return -1;
    }

    int find_voice_to_steal() const {
        int idx = -1;
        uint64_t oldest = UINT64_MAX;
        for (int i = 0; i < kMaxVoices; ++i) {
            if (voices_[i].active && voices_[i].note_id < oldest) {
                oldest = voices_[i].note_id;
                idx = i;
            }
        }
        return idx;
    }

    void trigger_note_on(float note, float vel, int slot, float porta_ms) {
        // With portamento, reuse existing voice for this slot
        if (porta_ms > 0.0f) {
            for (int i = 0; i < kMaxVoices; ++i) {
                if (voices_[i].active && !voices_[i].releasing &&
                    voices_[i].gate_slot == slot) {
                    // Glide: update note/freq, don't retrigger
                    voices_[i].note = note;
                    voices_[i].velocity = vel;
                    voices_[i].freq = midi_to_freq(note);
                    voices_[i].note_id = ++note_counter_;
                    return;
                }
            }
        }

        int vi = find_free_voice();
        if (vi < 0) vi = find_voice_to_steal();
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
        // Allocate fresh lane_id for new voice
        if (cur_ctx_ && cur_ctx_->allocate_lane_id_fn)
            v.lane_id = cur_ctx_->allocate_lane_id_fn(cur_ctx_->lane_state_service);
    }

    void trigger_note_off_slot(int slot) {
        for (int i = 0; i < kMaxVoices; ++i) {
            if (voices_[i].active && !voices_[i].releasing &&
                voices_[i].gate_slot == slot) {
                voices_[i].releasing = true;
            }
        }
    }

    // --- MIDI processing ---

    void process_midi(const VividAudioContext* ctx) {
        if (!ctx->custom_inputs || ctx->custom_input_count == 0 || !ctx->custom_inputs[0])
            return;

        auto* midi = static_cast<const VividMidiBuffer*>(ctx->custom_inputs[0]);
        float porta_ms = portamento.value;

        for (uint32_t m = 0; m < midi->count; ++m) {
            const auto& msg = midi->messages[m];
            uint8_t status = msg.status & 0xF0;

            if (status == 0x90 && msg.data2 > 0) {
                float note = static_cast<float>(msg.data1);
                float vel  = msg.data2 / 127.0f;

                int entry = -1;
                for (int i = 0; i < kMaxVoices; ++i) {
                    if (midi_voices_[i].active && midi_voices_[i].note == msg.data1) {
                        entry = i; break;
                    }
                }
                if (entry < 0) {
                    for (int i = 0; i < kMaxVoices; ++i) {
                        if (!midi_voices_[i].active) { entry = i; break; }
                    }
                }
                if (entry < 0) {
                    entry = 0;
                    trigger_note_off_slot(midi_voices_[0].slot);
                    midi_voices_[0].active = false;
                }

                int slot = kMidiSlotBase + entry;
                if (midi_voices_[entry].active && midi_voices_[entry].note == msg.data1)
                    trigger_note_off_slot(slot);

                midi_voices_[entry].note   = msg.data1;
                midi_voices_[entry].active = true;
                midi_voices_[entry].slot   = slot;
                trigger_note_on(note, vel, slot, porta_ms);

            } else if (status == 0x80 || (status == 0x90 && msg.data2 == 0)) {
                for (int i = 0; i < kMaxVoices; ++i) {
                    if (midi_voices_[i].active && midi_voices_[i].note == msg.data1) {
                        trigger_note_off_slot(midi_voices_[i].slot);
                        midi_voices_[i].active = false;
                        break;
                    }
                }
            }
        }
    }

    // --- Gate passthrough processing ---

    void update_gates(const VividAudioContext* ctx) {
        if (!ctx->input_spreads) return;

        const auto& notes_sp = ctx->input_spreads[0];
        const auto& vel_sp   = ctx->input_spreads[1];
        const auto& gates_sp = ctx->input_spreads[2];

        uint32_t len = gates_sp.length;
        if (len > static_cast<uint32_t>(kMaxVoices)) len = kMaxVoices;

        float porta_ms = portamento.value;

        for (uint32_t i = 0; i < len; ++i) {
            float cur_gate = read_spread_slot(&gates_sp, static_cast<int>(i));
            float cur_note = read_spread_slot(&notes_sp, static_cast<int>(i));
            float cur_vel  = read_spread_slot(&vel_sp,   static_cast<int>(i), 0.8f);

            float prev_gate = (i < prev_spread_len_) ? prev_gates_[i] : 0.0f;
            float prev_note = (i < prev_spread_len_) ? prev_notes_[i] : 0.0f;

            bool on        = (cur_gate > 0.5f) && (prev_gate <= 0.5f);
            bool off       = (cur_gate <= 0.5f) && (prev_gate > 0.5f);
            bool retrigger = (cur_gate > 0.5f) && (prev_gate > 0.5f) &&
                             (std::abs(cur_note - prev_note) > 0.5f);

            if (on || retrigger) {
                if (retrigger && porta_ms <= 0.0f)
                    trigger_note_off_slot(static_cast<int>(i));
                trigger_note_on(cur_note, cur_vel, static_cast<int>(i),
                               retrigger ? porta_ms : 0.0f);
            } else if (off) {
                trigger_note_off_slot(static_cast<int>(i));
            }

            prev_gates_[i] = cur_gate;
            prev_notes_[i] = cur_note;
        }

        for (uint32_t i = len; i < prev_spread_len_; ++i) {
            if (prev_gates_[i] > 0.5f)
                trigger_note_off_slot(static_cast<int>(i));
            prev_gates_[i] = 0.0f;
            prev_notes_[i] = 0.0f;
        }

        prev_spread_len_ = len;
    }

    // --- Main process ---

    void process_audio(const VividAudioContext* ctx) override {
        cur_ctx_ = ctx;
        process_midi(ctx);
        update_gates(ctx);

        // Clean up fully-released voices
        for (int i = 0; i < kMaxVoices; ++i) {
            if (voices_[i].releasing) {
                // Mark inactive after one buffer of release signaling
                // Downstream nodes (envelopes) handle actual release timing
                // Keep active for a few buffers so downstream sees the gate-off
            }
        }

        // Write output spreads
        if (!ctx->output_spreads) return;

        auto& notes_out   = ctx->output_spreads[0];
        auto& vel_out     = ctx->output_spreads[1];
        auto& gates_out   = ctx->output_spreads[2];
        auto& freq_out    = ctx->output_spreads[3];
        auto& lane_id_out = ctx->output_spreads[4];

        uint32_t count = 0;
        for (int vi = 0; vi < kMaxVoices; ++vi) {
            if (!voices_[vi].active) continue;
            if (count >= notes_out.capacity) break;

            notes_out.data[count]   = voices_[vi].note;
            vel_out.data[count]     = voices_[vi].velocity;
            gates_out.data[count]   = voices_[vi].releasing ? 0.0f : 1.0f;
            freq_out.data[count]    = voices_[vi].freq;
            lane_id_out.data[count] = static_cast<float>(voices_[vi].lane_id);
            count++;
        }

        notes_out.length   = std::min(count, notes_out.capacity);
        vel_out.length     = std::min(count, vel_out.capacity);
        gates_out.length   = std::min(count, gates_out.capacity);
        freq_out.length    = std::min(count, freq_out.capacity);
        lane_id_out.length = std::min(count, lane_id_out.capacity);

        // Deactivate voices that have been releasing (gate=0 sent)
        for (int i = 0; i < kMaxVoices; ++i) {
            if (voices_[i].releasing) {
                // Retire lane_id — state can be cleaned up on next frame sweep
                if (voices_[i].lane_id != 0 && ctx->retire_lane_id_fn)
                    ctx->retire_lane_id_fn(ctx->lane_state_service, voices_[i].lane_id);
                voices_[i].active = false;
                voices_[i].releasing = false;
                voices_[i].gate_slot = -1;
                voices_[i].lane_id = 0;
            }
        }
        cur_ctx_ = nullptr;
    }
};

VIVID_REGISTER(PolyVoiceAllocator)
