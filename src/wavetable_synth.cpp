#include "operator_api/operator.h"
#include "operator_api/audio_operator.h"
#include "operator_api/bound_control_instance.h"
#include "operator_api/adsr.h"
#include "operator_api/audio_dsp.h"
#include "operator_api/midi_types.h"
#include "operator_api/type_id.h"
#include "wavetable_bank.h"
#include "wavetable_dsp.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <memory>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr float PI_F    = static_cast<float>(M_PI);
static constexpr float TWO_PI_F = 2.0f * PI_F;

namespace adsr = vivid::adsr;
using vivid_wavetable::bank::Wavetable;
using vivid_wavetable::bank::build_builtin_wavetables;
using vivid_wavetable::bank::kBuiltinWavetableCount;
using vivid_wavetable::bank::load_wavetable_from_wav;
using namespace vivid_wavetable::dsp;

static constexpr int kCustomWavetableIndex = kBuiltinWavetableCount;

// =============================================================================
// WavetableSynth operator
// =============================================================================

struct WavetableSynth : vivid::AudioOperatorBase {
    static constexpr const char* kName   = "WavetableSynth";
    static constexpr bool kTimeDependent = true;

    // --- Parameters ---

    // Core
    vivid::Param<int>   wavetable        {"wavetable",        0,        {"Basic", "Analog", "Digital", "Vocal", "Texture", "PWM", "Formant", "Harmonic", "Metallic", "Custom"}};
    vivid::Param<float> position         {"position",         0.0f,     0.0f, 1.0f};
    vivid::Param<float> amplitude        {"amplitude",        0.3f,     0.0f, 1.0f};

    // Warp
    vivid::Param<int>   warp_mode        {"warp_mode",        0,        {"None", "Sync", "BendPlus", "BendMinus", "Mirror", "Asym", "Quantize", "FM", "Flip"}};
    vivid::Param<float> warp_amount      {"warp_amount",      0.0f,     0.0f, 1.0f};

    // Unison
    vivid::Param<int>   unison_voices    {"unison_voices",    1,        1, 16};
    vivid::Param<float> unison_spread    {"unison_spread",    20.0f,    0.0f, 100.0f};
    vivid::Param<float> unison_stereo    {"unison_stereo",    1.0f,     0.0f, 1.0f};
    vivid::Param<int>   unison_spread_mode {"unison_spread_mode", 0, {"Linear", "Exponential", "Random"}};

    // Sub oscillator
    vivid::Param<float> sub_level        {"sub_level",        0.0f,     0.0f, 1.0f};
    vivid::Param<int>   sub_octave       {"sub_octave",       0,        {"-1", "-2"}};
    vivid::Param<int>   sub_waveform     {"sub_waveform",     0,        {"Sine", "Triangle", "Saw", "Square", "Noise"}};

    // Noise oscillator
    vivid::Param<float> noise_level      {"noise_level",      0.0f,     0.0f, 1.0f};
    vivid::Param<int>   noise_type       {"noise_type",       0,        {"White", "Pink"}};

    // Portamento
    vivid::Param<float> portamento       {"portamento",       0.0f,     0.0f, 2000.0f};

    // Amplitude envelope
    vivid::Param<float> attack           {"attack",           0.01f,    0.001f, 5.0f};
    vivid::Param<float> decay            {"decay",            0.1f,     0.001f, 5.0f};
    vivid::Param<float> sustain          {"sustain",          0.7f,     0.0f,   1.0f};
    vivid::Param<float> release          {"release",          0.3f,     0.001f, 10.0f};

    // Filter
    vivid::Param<int>   filter_type      {"filter_type",      1,        {"LP12", "LP24", "HP12", "BP", "Notch", "Comb", "Ladder", "Formant"}};
    vivid::Param<float> filter_cutoff    {"filter_cutoff",    20000.0f, 20.0f,  20000.0f};
    vivid::Param<float> filter_resonance {"filter_resonance", 0.0f,     0.0f,   1.0f};
    vivid::Param<float> filter_keytrack  {"filter_keytrack",  0.0f,     0.0f,   1.0f};
    vivid::Param<float> filter_drive     {"filter_drive",     0.0f,     0.0f,   1.0f};

    // Filter envelope
    vivid::Param<float> filter_attack    {"filter_attack",    0.01f,    0.001f, 10.0f};
    vivid::Param<float> filter_decay     {"filter_decay",     0.3f,     0.001f, 10.0f};
    vivid::Param<float> filter_sustain   {"filter_sustain",   0.0f,     0.0f,   1.0f};
    vivid::Param<float> filter_release   {"filter_release",   0.3f,     0.001f, 10.0f};
    vivid::Param<float> filter_env_amount{"filter_env_amount",0.0f,    -1.0f,   1.0f};

    // Position envelope
    vivid::Param<float> position_attack     {"position_attack",     0.01f, 0.001f, 10.0f};
    vivid::Param<float> position_decay      {"position_decay",      0.3f,  0.001f, 10.0f};
    vivid::Param<float> position_sustain    {"position_sustain",    0.0f,  0.0f,   1.0f};
    vivid::Param<float> position_release    {"position_release",    0.3f,  0.001f, 10.0f};
    vivid::Param<float> position_env_amount {"position_env_amount", 0.0f, -1.0f,   1.0f};

    // Velocity
    vivid::Param<float> vel_to_volume    {"vel_to_volume",    1.0f,     0.0f,   1.0f};
    vivid::Param<float> vel_to_attack    {"vel_to_attack",    0.0f,    -1.0f,   1.0f};

    // Stereo & misc
    vivid::Param<float> stereo_spread    {"stereo_spread",    0.5f,     0.0f,   1.0f};
    vivid::Param<float> detune           {"detune",           0.0f,     0.0f,   50.0f};
    vivid::Param<bool>  env_bypass       {"env_bypass",       false};

    // Custom wavetable
    vivid::Param<vivid::FilePath> wav_file {"wav_file"};

    std::atomic<Wavetable*> custom_table_{nullptr};
    Wavetable* deferred_delete_ = nullptr;
    std::string last_wav_path_;

    ~WavetableSynth() {
        delete custom_table_.load(std::memory_order_relaxed);
        delete deferred_delete_;
    }

    void main_thread_update(double /*time*/) override {
        // Delete old table from previous swap
        if (deferred_delete_) {
            delete deferred_delete_;
            deferred_delete_ = nullptr;
        }

        // Check if wav_file path changed
        if (wav_file.str_value != last_wav_path_) {
            last_wav_path_ = wav_file.str_value;

            if (last_wav_path_.empty()) {
                // Clear custom table
                Wavetable* old = custom_table_.exchange(nullptr, std::memory_order_release);
                deferred_delete_ = old;
            } else {
                Wavetable* new_table = load_wavetable_from_wav(last_wav_path_);
                Wavetable* old = custom_table_.exchange(new_table, std::memory_order_release);
                deferred_delete_ = old;
            }
        }
    }

    // --- Voice state ---

    static constexpr int kMaxVoices = 16;

    struct Voice {
        float  note           = 0;
        float  velocity       = 0;
        double phase          = 0;
        double sub_phase      = 0;
        float  current_freq   = 0;
        float  target_freq    = 0;
        float  detune_offset  = 0;  // cents offset for unison
        float  pan            = 0;  // -1..1 for unison stereo
        float  last_sample    = 0;  // FM warp feedback
        audio_dsp::WhiteNoise white_noise;
        audio_dsp::PinkNoise  pink_noise;
        uint64_t note_id      = 0;
        int    gate_slot      = -1;

        adsr::State amp_env;
        adsr::State filt_env;
        adsr::State pos_env;

        // Biquad filter state (2 cascaded stages for LP24)
        float fz1[2] = {};
        float fz2[2] = {};

        // Additional filter states
        CombFilterState    comb;
        LadderFilterState  ladder;
        FormantFilterState formant;

        bool is_active() const { return amp_env.is_active(); }

        void reset_filter() {
            fz1[0] = fz1[1] = 0.0f;
            fz2[0] = fz2[1] = 0.0f;
            comb.reset();
            ladder.reset();
            formant.reset();
        }
    };

    Voice    voices_[kMaxVoices] = {};
    uint64_t note_counter_       = 0;

    // Previous spread inputs for gate edge detection
    float    prev_gates_[kMaxVoices] = {};
    float    prev_notes_[kMaxVoices] = {};
    uint32_t prev_spread_len_        = 0;

    // MIDI voice allocation: maps active MIDI notes to spread slots
    static constexpr int kMidiSlotBase = 128; // offset to avoid collision with spread slots
    struct MidiVoiceEntry {
        uint8_t note    = 0;
        bool    active  = false;
        int     slot    = -1;  // virtual slot index (kMidiSlotBase + index)
    };
    MidiVoiceEntry midi_voices_[kMaxVoices] = {};

    // --- Embedded operator slot state ---
    static constexpr int kNumSlots = 8;
    static constexpr int kSlotAmpEnv = 0, kSlotFiltEnv = 1, kSlotPosEnv = 2,
                         kSlotPitchMod = 3, kSlotWtPosMod = 4,
                         kSlotFilterMod = 5, kSlotWarpMod = 6,
                         kSlotPanMod = 7;

    struct SlotState {
        std::string type_name;                              // cached for change detection
        void* (*create_fn)(void) = nullptr;
        void (*destroy_fn)(void*) = nullptr;
        std::unordered_map<std::string, float> template_params;
        bool assigned = false;

        // Per-voice instances (pre-created when slot is assigned)
        std::unique_ptr<vivid::BoundControlInstance> voice_inst[kMaxVoices];

        void clear_instances() {
            for (auto& inst : voice_inst) inst.reset();
            assigned = false;
        }
    };
    SlotState slots_[kNumSlots];

    int role_index_for_id(const char* id) const {
        if (!id) return -1;
        if (std::strcmp(id, "amp_env") == 0) return kSlotAmpEnv;
        if (std::strcmp(id, "filt_env") == 0) return kSlotFiltEnv;
        if (std::strcmp(id, "pos_env") == 0) return kSlotPosEnv;
        if (std::strcmp(id, "pitch_mod") == 0) return kSlotPitchMod;
        if (std::strcmp(id, "wt_pos_mod") == 0) return kSlotWtPosMod;
        if (std::strcmp(id, "filter_mod") == 0) return kSlotFilterMod;
        if (std::strcmp(id, "warp_mod") == 0) return kSlotWarpMod;
        if (std::strcmp(id, "pan_mod") == 0) return kSlotPanMod;
        return -1;
    }

    bool is_voice_active(int vi) const {
        if (slots_[kSlotAmpEnv].assigned) {
            auto& inst = slots_[kSlotAmpEnv].voice_inst[vi];
            return inst && inst->output("value") > 0.0001f;
        }
        return voices_[vi].amp_env.is_active();
    }

    // All wavetables pre-computed in constructor so process() never generates.
    Wavetable all_tables_[kBuiltinWavetableCount];

    WavetableSynth() {
        vivid::semantic_tag(position, "phase_01");
        vivid::semantic_shape(position, "scalar");
        vivid::semantic_intent(position, "wavetable_position");

        vivid::semantic_tag(amplitude, "amplitude_linear");
        vivid::semantic_shape(amplitude, "scalar");

        vivid::semantic_tag(portamento, "time_milliseconds");
        vivid::semantic_shape(portamento, "scalar");
        vivid::semantic_unit(portamento, "ms");

        vivid::semantic_tag(attack, "time_seconds");
        vivid::semantic_shape(attack, "scalar");
        vivid::semantic_unit(attack, "s");
        vivid::semantic_tag(decay, "time_seconds");
        vivid::semantic_shape(decay, "scalar");
        vivid::semantic_unit(decay, "s");
        vivid::semantic_tag(sustain, "amplitude_linear");
        vivid::semantic_shape(sustain, "scalar");
        vivid::semantic_tag(release, "time_seconds");
        vivid::semantic_shape(release, "scalar");
        vivid::semantic_unit(release, "s");

        vivid::semantic_tag(filter_cutoff, "frequency_hz");
        vivid::semantic_shape(filter_cutoff, "scalar");
        vivid::semantic_unit(filter_cutoff, "Hz");
        vivid::semantic_tag(filter_resonance, "resonance");
        vivid::semantic_shape(filter_resonance, "scalar");

        vivid::semantic_tag(filter_attack, "time_seconds");
        vivid::semantic_shape(filter_attack, "scalar");
        vivid::semantic_unit(filter_attack, "s");
        vivid::semantic_tag(filter_decay, "time_seconds");
        vivid::semantic_shape(filter_decay, "scalar");
        vivid::semantic_unit(filter_decay, "s");
        vivid::semantic_tag(filter_sustain, "amplitude_linear");
        vivid::semantic_shape(filter_sustain, "scalar");
        vivid::semantic_tag(filter_release, "time_seconds");
        vivid::semantic_shape(filter_release, "scalar");
        vivid::semantic_unit(filter_release, "s");

        build_builtin_wavetables(all_tables_, kBuiltinWavetableCount);
    }

    // --- Param / port registration ---

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        // -- Groups --
        param_group(wavetable,  "Core");
        param_group(wav_file,   "Core");
        param_group(position,   "Core");
        param_group(amplitude,  "Core");

        param_group(warp_mode,   "Warp");
        param_group(warp_amount, "Warp");

        param_group(unison_voices, "Unison");
        param_group(unison_spread, "Unison");
        param_group(unison_stereo, "Unison");
        param_group(unison_spread_mode, "Unison");

        param_group(sub_level,    "Sub");
        param_group(sub_octave,   "Sub");
        param_group(sub_waveform, "Sub");

        param_group(noise_level, "Noise");
        param_group(noise_type,  "Noise");

        param_group(portamento, "Portamento");

        param_group(attack,  "Amp Envelope");
        param_group(decay,   "Amp Envelope");
        param_group(sustain, "Amp Envelope");
        param_group(release, "Amp Envelope");

        param_group(filter_type,      "Filter");
        param_group(filter_cutoff,    "Filter");
        param_group(filter_resonance, "Filter");
        param_group(filter_keytrack,  "Filter");
        param_group(filter_drive,     "Filter");

        param_group(filter_attack,     "Filter Envelope");
        param_group(filter_decay,      "Filter Envelope");
        param_group(filter_sustain,    "Filter Envelope");
        param_group(filter_release,    "Filter Envelope");
        param_group(filter_env_amount, "Filter Envelope");

        param_group(position_attack,     "Position Envelope");
        param_group(position_decay,      "Position Envelope");
        param_group(position_sustain,    "Position Envelope");
        param_group(position_release,    "Position Envelope");
        param_group(position_env_amount, "Position Envelope");

        param_group(vel_to_volume, "Velocity");
        param_group(vel_to_attack, "Velocity");

        param_group(stereo_spread, "Output");
        param_group(detune,        "Output");
        param_group(env_bypass,    "Output");

        // -- Display hints --
        display_hint(attack,  VIVID_DISPLAY_KNOB);
        display_hint(decay,   VIVID_DISPLAY_KNOB);
        display_hint(sustain, VIVID_DISPLAY_KNOB);
        display_hint(release, VIVID_DISPLAY_KNOB);

        display_hint(filter_cutoff,    VIVID_DISPLAY_KNOB);
        display_hint(filter_resonance, VIVID_DISPLAY_KNOB);
        display_hint(filter_keytrack,  VIVID_DISPLAY_KNOB);
        display_hint(filter_drive,     VIVID_DISPLAY_KNOB);

        display_hint(noise_level, VIVID_DISPLAY_KNOB);

        display_hint(filter_attack,  VIVID_DISPLAY_KNOB);
        display_hint(filter_decay,   VIVID_DISPLAY_KNOB);
        display_hint(filter_sustain, VIVID_DISPLAY_KNOB);
        display_hint(filter_release, VIVID_DISPLAY_KNOB);

        display_hint(position_attack,  VIVID_DISPLAY_KNOB);
        display_hint(position_decay,   VIVID_DISPLAY_KNOB);
        display_hint(position_sustain, VIVID_DISPLAY_KNOB);
        display_hint(position_release, VIVID_DISPLAY_KNOB);

        // -- Multi-column layouts --
        // Amp ADSR: 4 columns
        layout_row(attack,  4, 0);
        layout_row(decay,   4, 1);
        layout_row(sustain, 4, 2);
        layout_row(release, 4, 3);

        // Filter knobs: 4 columns
        layout_row(filter_cutoff,    4, 0);
        layout_row(filter_resonance, 4, 1);
        layout_row(filter_keytrack,  4, 2);
        layout_row(filter_drive,     4, 3);

        // Noise: 2 columns
        layout_row(noise_level, 2, 0);
        layout_row(noise_type,  2, 1);

        // Filter Envelope ADSR: 4 columns
        layout_row(filter_attack,  4, 0);
        layout_row(filter_decay,   4, 1);
        layout_row(filter_sustain, 4, 2);
        layout_row(filter_release, 4, 3);

        // Position Envelope ADSR: 4 columns
        layout_row(position_attack,  4, 0);
        layout_row(position_decay,   4, 1);
        layout_row(position_sustain, 4, 2);
        layout_row(position_release, 4, 3);

        // Velocity: 2 columns
        layout_row(vel_to_volume, 2, 0);
        layout_row(vel_to_attack, 2, 1);

        // Output: 2 columns for spread/detune, env_bypass full-width
        layout_row(stereo_spread, 2, 0);
        layout_row(detune,        2, 1);

        out.push_back(&wavetable);
        out.push_back(&wav_file);
        out.push_back(&position);
        out.push_back(&amplitude);
        out.push_back(&warp_mode);
        out.push_back(&warp_amount);
        out.push_back(&unison_voices);
        out.push_back(&unison_spread);
        out.push_back(&unison_stereo);
        out.push_back(&unison_spread_mode);
        out.push_back(&sub_level);
        out.push_back(&sub_octave);
        out.push_back(&sub_waveform);
        out.push_back(&noise_level);
        out.push_back(&noise_type);
        out.push_back(&portamento);
        out.push_back(&attack);
        out.push_back(&decay);
        out.push_back(&sustain);
        out.push_back(&release);
        out.push_back(&filter_type);
        out.push_back(&filter_cutoff);
        out.push_back(&filter_resonance);
        out.push_back(&filter_keytrack);
        out.push_back(&filter_drive);
        out.push_back(&filter_attack);
        out.push_back(&filter_decay);
        out.push_back(&filter_sustain);
        out.push_back(&filter_release);
        out.push_back(&filter_env_amount);
        out.push_back(&position_attack);
        out.push_back(&position_decay);
        out.push_back(&position_sustain);
        out.push_back(&position_release);
        out.push_back(&position_env_amount);
        out.push_back(&vel_to_volume);
        out.push_back(&vel_to_attack);
        out.push_back(&stereo_spread);
        out.push_back(&detune);
        out.push_back(&env_bypass);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back({"notes",      VIVID_PORT_SPREAD, VIVID_PORT_INPUT});   // 0
        out.push_back({"velocities", VIVID_PORT_SPREAD, VIVID_PORT_INPUT});   // 1
        out.push_back({"gates",      VIVID_PORT_SPREAD, VIVID_PORT_INPUT});   // 2
        out.push_back({"filter_env", VIVID_PORT_SPREAD, VIVID_PORT_INPUT});   // 3
        out.push_back({"pitch_mod",  VIVID_PORT_SPREAD, VIVID_PORT_INPUT});   // 4
        out.push_back({"amp_mod",      VIVID_PORT_SPREAD, VIVID_PORT_INPUT});   // 5
        out.push_back({"position_mod", VIVID_PORT_SPREAD, VIVID_PORT_INPUT});  // 6
        out.push_back(VIVID_CUSTOM_REF_PORT("midi_in", VIVID_PORT_INPUT, VividMidiBuffer)); // 7
        out.push_back({"output", VIVID_PORT_AUDIO, VIVID_PORT_OUTPUT, VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 2}); // out 0 (stereo)
        out.push_back({"envelopes",    VIVID_PORT_SPREAD, VIVID_PORT_OUTPUT}); // out 1
    }

    void collect_role_bindings(std::vector<VividRoleBindingDescriptor>& out) override {
        // amp_env: per-voice amplitude envelope
        {
            static const char* allowed[] = {"Envelope", "MSEG"};
            VividRoleBindingDescriptor s{};
            s.role_id = "amp_env";
            s.label = "Amp Envelope";
            s.accepted_domain = VIVID_DOMAIN_CONTROL;
            s.runtime_scope = VIVID_ROLE_PER_VOICE;
            s.allowed_operator_types = allowed;
            s.allowed_operator_type_count = 2;
            s.preferred_output_name = "value";
            s.default_operator_type = "Envelope";
            out.push_back(s);
        }
        // filt_env: per-voice filter envelope
        {
            static const char* allowed[] = {"Envelope", "MSEG"};
            VividRoleBindingDescriptor s{};
            s.role_id = "filt_env";
            s.label = "Filter Envelope";
            s.accepted_domain = VIVID_DOMAIN_CONTROL;
            s.runtime_scope = VIVID_ROLE_PER_VOICE;
            s.allowed_operator_types = allowed;
            s.allowed_operator_type_count = 2;
            s.preferred_output_name = "value";
            s.default_operator_type = "Envelope";
            out.push_back(s);
        }
        // pos_env: per-voice position envelope
        {
            static const char* allowed[] = {"Envelope", "LFO", "MSEG", "RandomSH", "Macro"};
            VividRoleBindingDescriptor s{};
            s.role_id = "pos_env";
            s.label = "Position Envelope";
            s.accepted_domain = VIVID_DOMAIN_CONTROL;
            s.runtime_scope = VIVID_ROLE_PER_VOICE;
            s.allowed_operator_types = allowed;
            s.allowed_operator_type_count = 5;
            s.preferred_output_name = "value";
            s.default_operator_type = "Envelope";
            out.push_back(s);
        }
        // pitch_mod: per-voice pitch modulator (default empty)
        {
            static const char* allowed[] = {"LFO", "Envelope", "MSEG", "RandomSH", "Macro", "StepSeq"};
            VividRoleBindingDescriptor s{};
            s.role_id = "pitch_mod";
            s.label = "Pitch Modulator";
            s.accepted_domain = VIVID_DOMAIN_CONTROL;
            s.runtime_scope = VIVID_ROLE_PER_VOICE;
            s.allowed_operator_types = allowed;
            s.allowed_operator_type_count = 6;
            s.preferred_output_name = "value";
            s.default_operator_type = nullptr;
            out.push_back(s);
        }
        // wt_pos_mod: per-voice wavetable position modulator (default empty)
        {
            static const char* allowed[] = {"LFO", "Envelope", "MSEG", "RandomSH", "Macro", "StepSeq"};
            VividRoleBindingDescriptor s{};
            s.role_id = "wt_pos_mod";
            s.label = "WT Position Mod";
            s.accepted_domain = VIVID_DOMAIN_CONTROL;
            s.runtime_scope = VIVID_ROLE_PER_VOICE;
            s.allowed_operator_types = allowed;
            s.allowed_operator_type_count = 6;
            s.preferred_output_name = "value";
            s.default_operator_type = nullptr;
            out.push_back(s);
        }
        // filter_mod: per-voice filter cutoff modulator
        {
            static const char* allowed[] = {"LFO", "Envelope", "MSEG", "RandomSH", "Macro", "StepSeq"};
            VividRoleBindingDescriptor s{};
            s.role_id = "filter_mod";
            s.label = "Filter Modulator";
            s.accepted_domain = VIVID_DOMAIN_CONTROL;
            s.runtime_scope = VIVID_ROLE_PER_VOICE;
            s.allowed_operator_types = allowed;
            s.allowed_operator_type_count = 6;
            s.preferred_output_name = "value";
            s.default_operator_type = nullptr;
            out.push_back(s);
        }
        // warp_mod: per-voice warp amount modulator
        {
            static const char* allowed[] = {"LFO", "Envelope", "MSEG", "RandomSH", "Macro", "StepSeq"};
            VividRoleBindingDescriptor s{};
            s.role_id = "warp_mod";
            s.label = "Warp Modulator";
            s.accepted_domain = VIVID_DOMAIN_CONTROL;
            s.runtime_scope = VIVID_ROLE_PER_VOICE;
            s.allowed_operator_types = allowed;
            s.allowed_operator_type_count = 6;
            s.preferred_output_name = "value";
            s.default_operator_type = nullptr;
            out.push_back(s);
        }
        // pan_mod: per-voice pan modulator (default empty)
        {
            static const char* allowed[] = {"LFO", "Envelope", "MSEG", "RandomSH", "Macro", "StepSeq"};
            VividRoleBindingDescriptor s{};
            s.role_id = "pan_mod";
            s.label = "Pan Modulator";
            s.accepted_domain = VIVID_DOMAIN_CONTROL;
            s.runtime_scope = VIVID_ROLE_PER_VOICE;
            s.allowed_operator_types = allowed;
            s.allowed_operator_type_count = 6;
            s.preferred_output_name = "value";
            s.default_operator_type = nullptr;
            out.push_back(s);
        }
    }

    // --- Helpers ---

    static float cents_to_ratio(float cents) {
        return std::pow(2.0f, cents / 1200.0f);
    }

    static float read_spread_slot(const VividSpreadPort* sp, int slot, float fallback = 0.0f) {
        if (sp && sp->data && slot >= 0 && static_cast<uint32_t>(slot) < sp->length)
            return sp->data[slot];
        return fallback;
    }

    static float midi_to_freq(float note) {
        return 440.0f * std::pow(2.0f, (note - 69.0f) / 12.0f);
    }

    // --- Voice management ---

    int find_free_voice() const {
        for (int i = 0; i < kMaxVoices; ++i)
            if (!is_voice_active(i)) return i;
        return -1;
    }

    int find_voice_to_steal() const {
        int idx = -1;
        uint64_t oldest = UINT64_MAX;
        for (int i = 0; i < kMaxVoices; ++i) {
            if (is_voice_active(i) && voices_[i].note_id < oldest) {
                oldest = voices_[i].note_id;
                idx = i;
            }
        }
        return idx;
    }

    int find_voice_by_slot(int slot) const {
        for (int i = 0; i < kMaxVoices; ++i) {
            if (is_voice_active(i) &&
                voices_[i].amp_env.stage != adsr::RELEASE &&
                voices_[i].gate_slot == slot)
                return i;
        }
        return -1;
    }

    void trigger_note_on(float note, float vel, int slot, float porta_ms) {
        int num_uni   = unison_voices.int_value();
        float uni_spr = unison_spread.value;
        float uni_st  = unison_stereo.value;

        for (int u = 0; u < num_uni; ++u) {
            // Check if there's already a voice for this slot+unison index
            // For portamento: reuse existing voice instead of allocating new
            int vi = -1;
            if (porta_ms > 0.0f) {
                // Find existing voice for this slot with matching unison position
                int found = 0;
                for (int i = 0; i < kMaxVoices; ++i) {
                    if (is_voice_active(i) &&
                        voices_[i].amp_env.stage != adsr::RELEASE &&
                        voices_[i].gate_slot == slot) {
                        if (found == u) { vi = i; break; }
                        ++found;
                    }
                }
            }

            if (vi >= 0) {
                // Portamento: update target frequency, don't reset envelope
                voices_[vi].note = note;
                voices_[vi].target_freq = midi_to_freq(note);
                voices_[vi].velocity = vel;
                voices_[vi].note_id = ++note_counter_;
                continue;
            }

            // Allocate new voice
            vi = find_free_voice();
            if (vi < 0) vi = find_voice_to_steal();
            if (vi < 0) break;

            Voice& v = voices_[vi];
            v.note = note;
            v.velocity = vel;
            v.gate_slot = slot;
            v.note_id = ++note_counter_;

            // Unison detune & pan
            if (num_uni > 1) {
                float t = static_cast<float>(u) / static_cast<float>(num_uni - 1);
                float centered = t - 0.5f;

                int mode = unison_spread_mode.int_value();
                if (mode == 1) {
                    // Exponential: wider spacing at extremes
                    float sign = (centered >= 0.0f) ? 1.0f : -1.0f;
                    centered = sign * std::pow(std::abs(centered) * 2.0f, 1.5f) * 0.5f;
                } else if (mode == 2) {
                    // Random: deterministic hash from voice index + note
                    uint32_t seed = static_cast<uint32_t>(vi) * 1664525u
                                  + static_cast<uint32_t>(note * 100.0f);
                    seed ^= seed >> 16; seed *= 0x45d9f3bu; seed ^= seed >> 16;
                    centered = (static_cast<float>(seed & 0xFFFFu) / 65535.0f - 0.5f);
                }

                v.detune_offset = centered * uni_spr;
                v.pan = (t - 0.5f) * 2.0f * uni_st;
            } else {
                v.detune_offset = 0.0f;
                v.pan = 0.0f;
            }

            float freq = midi_to_freq(note);
            v.target_freq = freq;
            v.current_freq = freq; // no portamento for new voice
            v.phase = 0.0;
            v.sub_phase = 0.0;
            v.last_sample = 0.0f;
            v.white_noise.state = 12345u + static_cast<uint32_t>(vi) * 1664525u;
            v.pink_noise = {};
            v.pink_noise.white.state = 67890u + static_cast<uint32_t>(vi) * 1664525u;

            adsr::gate_on(v.amp_env);
            adsr::gate_on(v.filt_env);
            adsr::gate_on(v.pos_env);

            // Gate-on embedded slot instances for this voice
            for (int s = 0; s < kNumSlots; ++s) {
                if (slots_[s].assigned && slots_[s].voice_inst[vi]) {
                    auto& inst = *slots_[s].voice_inst[vi];
                    inst.reset();
                    inst.apply_template(slots_[s].template_params);
                    if (inst.has_input("gate")) inst.set_input("gate", 1.0f);
                }
            }

            v.reset_filter();
        }
    }

    void trigger_note_off_slot(int slot) {
        for (int i = 0; i < kMaxVoices; ++i) {
            if (is_voice_active(i) &&
                voices_[i].amp_env.stage != adsr::RELEASE &&
                voices_[i].gate_slot == slot) {
                adsr::gate_off(voices_[i].amp_env);
                adsr::gate_off(voices_[i].filt_env);
                adsr::gate_off(voices_[i].pos_env);

                // Gate-off embedded slot instances
                for (int s = 0; s < kNumSlots; ++s) {
                    if (slots_[s].assigned && slots_[s].voice_inst[i]) {
                        if (slots_[s].voice_inst[i]->has_input("gate"))
                            slots_[s].voice_inst[i]->set_input("gate", 0.0f);
                    }
                }
            }
        }
    }

    // --- Biquad filter ---

    float apply_biquad(Voice& v, float input, float cutoff_hz, float reso,
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
            case FILTER_LP12:
            case FILTER_LP24:
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

        // Normalize
        float inv_a0 = 1.0f / a0;
        b0 *= inv_a0; b1 *= inv_a0; b2 *= inv_a0;
        a1 *= inv_a0; a2 *= inv_a0;

        // Stage 1 (transposed direct form II)
        float out = b0 * input + v.fz1[0];
        v.fz1[0] = b1 * input - a1 * out + v.fz2[0];
        v.fz2[0] = b2 * input - a2 * out;

        // Stage 2 for LP24 (4-pole)
        if (ftype == FILTER_LP24) {
            float in2 = out;
            out = b0 * in2 + v.fz1[1];
            v.fz1[1] = b1 * in2 - a1 * out + v.fz2[1];
            v.fz2[1] = b2 * in2 - a2 * out;
        }

        return out;
    }

    // --- Filter dispatch ---

    float apply_filter(Voice& v, float input, float cutoff_hz, float reso,
                       int ftype, float sr) {
        switch (ftype) {
            case FILTER_LP12: case FILTER_LP24: case FILTER_HP12:
            case FILTER_BP:   case FILTER_NOTCH:
                return apply_biquad(v, input, cutoff_hz, reso, ftype, sr);
            case FILTER_COMB: {
                float delay_samples = sr / std::max(cutoff_hz, 20.0f);
                float feedback = reso * 0.98f;
                return v.comb.process(input, delay_samples, feedback);
            }
            case FILTER_LADDER:
                return v.ladder.process(input, cutoff_hz, reso, sr);
            case FILTER_FORMANT: {
                float morph = std::log2(cutoff_hz / 20.0f)
                            / std::log2(20000.0f / 20.0f);
                morph = std::clamp(morph, 0.0f, 1.0f);
                return v.formant.process(input, morph, reso, sr);
            }
            default:
                return input;
        }
    }

    // --- MIDI input processing ---

    void process_midi(const VividAudioContext* ctx) {
        if (!ctx->custom_inputs || ctx->custom_input_count == 0 || !ctx->custom_inputs[0])
            return;

        auto* midi = static_cast<const VividMidiBuffer*>(ctx->custom_inputs[0]);
        float porta_ms = portamento.value;

        for (uint32_t m = 0; m < midi->count; ++m) {
            const auto& msg = midi->messages[m];
            uint8_t status = msg.status & 0xF0;

            if (status == 0x90 && msg.data2 > 0) {
                // Note On — find or allocate a MIDI voice entry
                float note = static_cast<float>(msg.data1);
                float vel  = msg.data2 / 127.0f;

                // Check if this note is already active (retrigger)
                int entry = -1;
                for (int i = 0; i < kMaxVoices; ++i) {
                    if (midi_voices_[i].active && midi_voices_[i].note == msg.data1) {
                        entry = i;
                        break;
                    }
                }

                if (entry < 0) {
                    // Find a free MIDI voice entry
                    for (int i = 0; i < kMaxVoices; ++i) {
                        if (!midi_voices_[i].active) {
                            entry = i;
                            break;
                        }
                    }
                }

                if (entry < 0) {
                    // All slots full — steal the oldest (first active)
                    entry = 0;
                    trigger_note_off_slot(midi_voices_[0].slot);
                    midi_voices_[0].active = false;
                }

                int slot = kMidiSlotBase + entry;

                if (midi_voices_[entry].active && midi_voices_[entry].note == msg.data1) {
                    // Retrigger same note
                    trigger_note_off_slot(slot);
                }

                midi_voices_[entry].note   = msg.data1;
                midi_voices_[entry].active = true;
                midi_voices_[entry].slot   = slot;

                trigger_note_on(note, vel, slot, porta_ms);

            } else if (status == 0x80 || (status == 0x90 && msg.data2 == 0)) {
                // Note Off — find the matching MIDI voice entry
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

    // --- Gate processing ---

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
                trigger_note_on(cur_note, cur_vel, static_cast<int>(i), retrigger ? porta_ms : 0.0f);
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
        float* out_l = ctx->output_buffers[0];
        float* out_r = ctx->output_buffers[0] + ctx->buffer_size;
        uint32_t frames = ctx->buffer_size;
        float sr  = static_cast<float>(ctx->sample_rate);
        float dt  = 1.0f / sr;

        // Read params
        int   wt_idx       = std::clamp(wavetable.int_value(), 0, kCustomWavetableIndex);
        float pos          = position.value;
        float amp          = amplitude.value;
        int   warp_m       = warp_mode.int_value();
        float warp_a       = warp_amount.value;
        int   num_uni      = unison_voices.int_value();
        float sub_lvl      = sub_level.value;
        int   sub_oct      = sub_octave.int_value();
        int   sub_wave     = sub_waveform.int_value();
        float noise_lvl    = noise_level.value;
        int   noise_tp     = noise_type.int_value();
        float porta_ms     = portamento.value;
        float att          = attack.value;
        float dec          = decay.value;
        float sus          = sustain.value;
        float rel          = release.value;
        int   ftype        = filter_type.int_value();
        float f_cutoff     = filter_cutoff.value;
        float f_reso       = filter_resonance.value;
        float f_keytrack   = filter_keytrack.value;
        float f_drive      = filter_drive.value;
        float f_att        = filter_attack.value;
        float f_dec        = filter_decay.value;
        float f_sus        = filter_sustain.value;
        float f_rel        = filter_release.value;
        float f_env_amt    = filter_env_amount.value;
        float p_att        = position_attack.value;
        float p_dec        = position_decay.value;
        float p_sus        = position_sustain.value;
        float p_rel        = position_release.value;
        float p_env_amt    = position_env_amount.value;
        float v2vol        = vel_to_volume.value;
        float v2atk        = vel_to_attack.value;
        float spread       = stereo_spread.value;
        float det_cents    = detune.value;
        bool  bypass       = env_bypass.value > 0.5f;

        const Wavetable* wt_ptr;
        if (wt_idx == kCustomWavetableIndex) {
            wt_ptr = custom_table_.load(std::memory_order_acquire);
            if (!wt_ptr) wt_ptr = &all_tables_[0]; // fallback to Basic
        } else {
            wt_ptr = &all_tables_[wt_idx];
        }
        const Wavetable& wt = *wt_ptr;

        // Modulation spread inputs
        const VividSpreadPort* filter_env_sp = ctx->input_spreads ? &ctx->input_spreads[3] : nullptr;
        const VividSpreadPort* pitch_mod_sp  = ctx->input_spreads ? &ctx->input_spreads[4] : nullptr;
        const VividSpreadPort* amp_mod_sp    = ctx->input_spreads ? &ctx->input_spreads[5] : nullptr;
        const VividSpreadPort* position_mod_sp = ctx->input_spreads ? &ctx->input_spreads[6] : nullptr;

        process_midi(ctx);
        update_gates(ctx);

        // Sync role binding config from audio context
        if (ctx->role_binding_configs) {
            for (uint32_t si = 0; si < ctx->role_binding_count; ++si) {
                const auto& cfg = ctx->role_binding_configs[si];
                int idx = role_index_for_id(cfg.role_id);
                if (idx < 0) continue;
                auto& slot = slots_[idx];

                // Detect assignment change
                bool type_changed = (slot.type_name != cfg.bound_node_type);
                if (type_changed) {
                    slot.clear_instances();
                    slot.type_name = cfg.bound_node_type;
                    slot.create_fn = cfg.create_fn;
                    slot.destroy_fn = cfg.destroy_fn;
                    slot.assigned = (cfg.bound_node_type[0] != '\0' && cfg.create_fn);
                    if (slot.assigned) {
                        // Pre-create instances for all voice slots
                        for (int v = 0; v < kMaxVoices; ++v) {
                            auto* raw = static_cast<vivid::OperatorBase*>(cfg.create_fn());
                            slot.voice_inst[v] = std::make_unique<vivid::BoundControlInstance>(
                                raw, [d = cfg.destroy_fn](vivid::OperatorBase* p) {
                                    d(static_cast<void*>(p));
                                });
                        }
                    }
                }

                // Update template params (always — param values may change between frames)
                slot.template_params.clear();
                for (uint32_t p = 0; p < cfg.param_count; ++p)
                    slot.template_params[cfg.param_names[p]] = cfg.param_values[p];

                // Apply updated params to all existing voice instances
                if (slot.assigned) {
                    for (int v = 0; v < kMaxVoices; ++v) {
                        if (slot.voice_inst[v])
                            slot.voice_inst[v]->apply_template(slot.template_params);
                    }
                }
            }
        }

        // Portamento rate (per-sample exponential glide)
        float porta_rate = 1.0f;
        if (porta_ms > 0.0f) {
            float porta_samples = porta_ms * 0.001f * sr;
            porta_rate = 1.0f - std::exp(-4.0f / porta_samples);
        }

        // Sub oscillator divisor
        float sub_div = (sub_oct == 1) ? 4.0f : 2.0f; // choice 0="-1"(÷2), 1="-2"(÷4)

        // Filter active check
        bool filter_active = (ftype >= FILTER_COMB) ||
                             (f_cutoff < 19999.0f) || (f_reso > 0.01f) ||
                             (std::abs(f_env_amt) > 0.001f) || (f_drive > 0.001f);
        bool pos_env_active = p_env_amt != 0.0f;

        float norm = 1.0f / std::sqrt(static_cast<float>(kMaxVoices));

        // Pre-compute per-voice stereo pan gains
        uint32_t spread_len = prev_spread_len_;
        float voice_gain_l[kMaxVoices] = {};
        float voice_gain_r[kMaxVoices] = {};
        float voice_base_pan[kMaxVoices] = {};

        for (int vi = 0; vi < kMaxVoices; ++vi) {
            Voice& v = voices_[vi];
            if (!is_voice_active(vi)) continue;

            // Combine slot-based stereo spread and unison pan
            float pan = v.pan; // unison pan
            if (num_uni <= 1 && spread_len > 1 && v.gate_slot >= 0) {
                // No unison: use slot-based spread (like original Polysynth)
                pan = (static_cast<float>(v.gate_slot) /
                       static_cast<float>(spread_len - 1) * 2.0f - 1.0f) * spread;
            }
            voice_base_pan[vi] = pan;
            float theta = (pan + 1.0f) * PI_F * 0.25f;
            voice_gain_l[vi] = std::cos(theta);
            voice_gain_r[vi] = std::sin(theta);
        }

        std::memset(out_l, 0, frames * sizeof(float));
        std::memset(out_r, 0, frames * sizeof(float));

        for (uint32_t s = 0; s < frames; ++s) {
            float left_mix  = 0.0f;
            float right_mix = 0.0f;

            // Build a synthetic VividProcessContext for stepping embedded ops
            VividProcessContext emb_ctx{};
            emb_ctx.time = static_cast<double>(ctx->frame + s) / sr;
            emb_ctx.delta_time = static_cast<double>(dt);
            emb_ctx.frame = ctx->frame + s;

            for (int vi = 0; vi < kMaxVoices; ++vi) {
                Voice& v = voices_[vi];
                if (!is_voice_active(vi)) continue;

                // Velocity→attack modulation
                float eff_att = att;
                if (v2atk != 0.0f) {
                    float vel_mod = v2atk * (1.0f - v.velocity);
                    eff_att *= std::pow(2.0f, vel_mod * 2.0f);
                    eff_att = std::clamp(eff_att, 0.001f, 10.0f);
                }

                // Dual-path amplitude envelope
                float amp_env_val;
                if (slots_[kSlotAmpEnv].assigned) {
                    auto& inst = *slots_[kSlotAmpEnv].voice_inst[vi];
                    inst.process(&emb_ctx);
                    amp_env_val = inst.output("value");
                } else {
                    adsr::advance(v.amp_env, dt, eff_att, dec, sus, rel);
                    amp_env_val = bypass ? 1.0f : v.amp_env.env_value;
                }
                // Check if voice died (only for fallback path)
                if (!slots_[kSlotAmpEnv].assigned && !v.is_active()) continue;
                if (slots_[kSlotAmpEnv].assigned && amp_env_val <= 0.0001f &&
                    v.amp_env.stage == adsr::RELEASE) continue;

                // Dual-path filter envelope
                float filt_env_val = 0.0f;
                if (filter_active) {
                    if (slots_[kSlotFiltEnv].assigned) {
                        auto& inst = *slots_[kSlotFiltEnv].voice_inst[vi];
                        inst.process(&emb_ctx);
                        filt_env_val = inst.output("value");
                    } else {
                        adsr::advance(v.filt_env, dt, f_att, f_dec, f_sus, f_rel);
                        filt_env_val = v.filt_env.env_value;
                    }
                }

                // Dual-path position envelope
                float pos_env_val = 0.0f;
                if (pos_env_active || (slots_[kSlotPosEnv].assigned)) {
                    if (slots_[kSlotPosEnv].assigned) {
                        auto& inst = *slots_[kSlotPosEnv].voice_inst[vi];
                        inst.process(&emb_ctx);
                        pos_env_val = inst.output("value");
                    } else {
                        adsr::advance(v.pos_env, dt, p_att, p_dec, p_sus, p_rel);
                        pos_env_val = v.pos_env.env_value;
                    }
                }

                // Embedded pitch modulator
                float pitch_mod_embedded = 0.0f;
                if (slots_[kSlotPitchMod].assigned) {
                    auto& inst = *slots_[kSlotPitchMod].voice_inst[vi];
                    inst.process(&emb_ctx);
                    pitch_mod_embedded = inst.output("value");
                }

                // Embedded WT position modulator
                float wt_pos_mod_embedded = 0.0f;
                if (slots_[kSlotWtPosMod].assigned) {
                    auto& inst = *slots_[kSlotWtPosMod].voice_inst[vi];
                    inst.process(&emb_ctx);
                    wt_pos_mod_embedded = inst.output("value");
                }

                // Embedded filter cutoff modulator
                float filter_mod_embedded = 0.0f;
                if (slots_[kSlotFilterMod].assigned) {
                    auto& inst = *slots_[kSlotFilterMod].voice_inst[vi];
                    inst.process(&emb_ctx);
                    filter_mod_embedded = inst.output("value");
                }

                // Embedded warp amount modulator
                float warp_mod_embedded = 0.0f;
                if (slots_[kSlotWarpMod].assigned) {
                    auto& inst = *slots_[kSlotWarpMod].voice_inst[vi];
                    inst.process(&emb_ctx);
                    warp_mod_embedded = inst.output("value");
                }

                // Embedded pan modulator
                float pan_mod_embedded = 0.0f;
                if (slots_[kSlotPanMod].assigned) {
                    auto& inst = *slots_[kSlotPanMod].voice_inst[vi];
                    inst.process(&emb_ctx);
                    pan_mod_embedded = inst.output("value");
                }

                // Portamento: glide current_freq toward target_freq
                if (porta_ms > 0.0f && v.current_freq != v.target_freq) {
                    v.current_freq += (v.target_freq - v.current_freq) * porta_rate;
                    if (std::abs(v.current_freq - v.target_freq) < 0.01f)
                        v.current_freq = v.target_freq;
                }

                // Pitch modulation (external spread + embedded)
                float pitch_offset = read_spread_slot(pitch_mod_sp, v.gate_slot);
                pitch_offset += pitch_mod_embedded;
                float freq = v.current_freq *
                             cents_to_ratio(v.detune_offset + det_cents) *
                             std::pow(2.0f, pitch_offset / 12.0f);
                if (!std::isfinite(freq) || freq <= 0.0f) freq = v.current_freq;

                float phase_inc = static_cast<float>(freq) / sr;

                // Phase warp + wavetable sample (with embedded warp modulator)
                float eff_warp = std::clamp(warp_a + warp_mod_embedded, 0.0f, 1.0f);
                float warped = warp_phase(static_cast<float>(v.phase), warp_m, eff_warp, v.last_sample);

                // Position modulation (internal envelope + external spread + embedded)
                float effective_pos = pos;
                effective_pos += pos_env_val * p_env_amt;
                float ext_pos = read_spread_slot(position_mod_sp, v.gate_slot);
                effective_pos += ext_pos + wt_pos_mod_embedded;
                effective_pos = std::clamp(effective_pos, 0.0f, 1.0f);

                float sig = wt.sample(warped, effective_pos, freq, sr);
                v.last_sample = sig;

                // Sub oscillator
                if (sub_lvl > 0.0f) {
                    float sub_freq = v.current_freq / sub_div;
                    float sub_inc  = sub_freq / sr;
                    float sub_sig;
                    if (sub_wave == 4) {
                        sub_sig = v.white_noise.next();
                    } else {
                        // Map param order (Sine=0, Tri=1, Saw=2, Sq=3)
                        // to audio_dsp::waveform order (sine=0, saw=1, sq=2, tri=3)
                        static constexpr int wf_map[] = {0, 3, 1, 2};
                        sub_sig = static_cast<float>(audio_dsp::waveform(v.sub_phase, wf_map[sub_wave]));
                    }
                    sig = sig * (1.0f - sub_lvl) + sub_sig * sub_lvl;
                    v.sub_phase += static_cast<double>(sub_inc);
                    if (v.sub_phase >= 1.0) v.sub_phase -= 1.0;
                    if (!std::isfinite(v.sub_phase)) v.sub_phase = 0.0;
                }

                // Noise oscillator
                if (noise_lvl > 0.001f) {
                    float n = (noise_tp == 0) ? v.white_noise.next()
                                              : v.pink_noise.next();
                    sig += n * noise_lvl;
                }

                // Per-voice filter (dual-path filter envelope)
                if (filter_active) {
                    float cutoff = f_cutoff;

                    // Filter envelope modulation (bipolar)
                    float env_mod = filt_env_val * f_env_amt;
                    cutoff *= std::pow(2.0f, env_mod * 4.0f);

                    // External filter envelope modulation
                    float ext_fenv = read_spread_slot(filter_env_sp, v.gate_slot);
                    if (ext_fenv != 0.0f)
                        cutoff *= std::pow(2.0f, ext_fenv * 4.0f);

                    // Embedded filter modulator (bipolar, ±4 octave range)
                    if (filter_mod_embedded != 0.0f)
                        cutoff *= std::pow(2.0f, filter_mod_embedded * 4.0f);

                    // Keytracking
                    if (f_keytrack > 0.0f) {
                        float oct_from_c4 = std::log2(v.current_freq / 261.63f);
                        cutoff *= std::pow(2.0f, oct_from_c4 * f_keytrack);
                    }

                    // Filter drive (gain-compensated soft clip)
                    if (f_drive > 0.001f) {
                        float d = 1.0f + f_drive * 7.0f;
                        sig = std::tanh(sig * d) / std::tanh(d);
                    }

                    sig = apply_filter(v, sig, cutoff, f_reso, ftype, sr);
                }

                // Envelope & velocity (dual-path amplitude)
                float vel_vol = 1.0f - v2vol * (1.0f - v.velocity);
                sig *= amp_env_val * vel_vol;
                sig *= read_spread_slot(amp_mod_sp, v.gate_slot, 1.0f);

                float gl = voice_gain_l[vi];
                float gr = voice_gain_r[vi];
                if (pan_mod_embedded != 0.0f) {
                    float mod_pan = std::clamp(voice_base_pan[vi] + pan_mod_embedded, -1.0f, 1.0f);
                    float theta = (mod_pan + 1.0f) * PI_F * 0.25f;
                    gl = std::cos(theta);
                    gr = std::sin(theta);
                }
                left_mix  += sig * gl;
                right_mix += sig * gr;

                // Advance phase
                v.phase += static_cast<double>(phase_inc);
                if (v.phase >= 1.0) v.phase -= 1.0;
                if (!std::isfinite(v.phase)) v.phase = 0.0;
            }

            out_l[s] = left_mix * amp * norm;
            out_r[s] = right_mix * amp * norm;
        }

        // Write per-voice envelope values to output spread
        if (ctx->output_spreads) {
            auto& env_sp = ctx->output_spreads[1];
            uint32_t active_count = 0;
            for (int vi = 0; vi < kMaxVoices; ++vi) {
                if (is_voice_active(vi)) {
                    if (active_count < env_sp.capacity) {
                        if (slots_[kSlotAmpEnv].assigned &&
                            slots_[kSlotAmpEnv].voice_inst[vi]) {
                            env_sp.data[active_count] = slots_[kSlotAmpEnv].voice_inst[vi]->output("value");
                        } else {
                            env_sp.data[active_count] = voices_[vi].amp_env.env_value;
                        }
                    }
                    active_count++;
                }
            }
            env_sp.length = std::min(active_count, env_sp.capacity);
        }
    }
};

VIVID_REGISTER(WavetableSynth)
