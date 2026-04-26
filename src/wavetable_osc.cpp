#include "wavetable_osc_internal.h"

#include "lane_audio_utils.h"
#include "wavetable_voice_utils.h"
#include "operator_api/type_id.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

using vivid_wavetable::bank::build_builtin_wavetables;
using vivid_wavetable::bank::load_wavetable_from_wav;
using vivid_wavetable::bank::resolve_builtin_wavetable;

WavetableOsc::WavetableOsc() {
    vivid::semantic_tag(position, "phase_01");
    vivid::semantic_intent(position, "wavetable_position");
    vivid::semantic_tag(amplitude, "amplitude_linear");
    vivid::semantic_tag(portamento, "time_milliseconds");
    vivid::semantic_unit(portamento, "ms");
    vivid::semantic_tag(position_smooth_ms, "time_milliseconds");
    vivid::semantic_unit(position_smooth_ms, "ms");
    vivid::semantic_tag(warp_smooth_ms, "time_milliseconds");
    vivid::semantic_unit(warp_smooth_ms, "ms");
    vivid::semantic_tag(start_phase, "phase_01");
    vivid::semantic_tag(stereo_phase_offset, "phase_01");
    vivid::semantic_tag(drift_rate_hz, "frequency_hz");
    vivid::semantic_unit(drift_rate_hz, "Hz");
    vivid::semantic_tag(interaction_input_gain, "gain_linear");
}

WavetableOsc::~WavetableOsc() {
    delete custom_table_.load(std::memory_order_relaxed);
    delete deferred_delete_;
    release_thumb_gpu();
}

void WavetableOsc::prepare_instance_assets() {
    (void)builtin_tables();
}

void WavetableOsc::main_thread_update(double) {
    if (deferred_delete_) {
        delete deferred_delete_;
        deferred_delete_ = nullptr;
    }
    if (wav_file.str_value != last_wav_path_) {
        last_wav_path_ = wav_file.str_value;
        if (last_wav_path_.empty()) {
            deferred_delete_ = custom_table_.exchange(nullptr, std::memory_order_release);
        } else {
            Wavetable* next = load_wavetable_from_wav(last_wav_path_);
            deferred_delete_ = custom_table_.exchange(next, std::memory_order_release);
        }
    }
}

void WavetableOsc::collect_params(std::vector<vivid::ParamBase*>& out) {
    param_group(wavetable_source, "Core");
    param_group(wavetable_family, "Core");
    param_group(wavetable_member, "Core");
    param_group(wav_file, "Core");
    param_group(position, "Core");
    param_group(amplitude, "Core");
    param_group(warp_mode, "Warp");
    param_group(warp_amount, "Warp");
    param_group(position_smooth_ms, "Motion");
    param_group(warp_smooth_ms, "Motion");
    param_group(phase_reset_mode, "Phase");
    param_group(start_phase, "Phase");
    param_group(phase_random, "Phase");
    param_group(stereo_phase_offset, "Phase");
    param_group(drift_amount, "Motion");
    param_group(drift_rate_hz, "Motion");
    param_group(unison_voices, "Unison");
    param_group(unison_spread, "Unison");
    param_group(unison_stereo, "Unison");
    param_group(unison_spread_mode, "Unison");
    param_group(unison_output_mode, "Unison");
    param_group(detune, "Output");
    param_group(portamento, "Portamento");
    param_group(interaction_mode, "Interaction");
    param_group(interaction_depth, "Interaction");
    param_group(interaction_input_gain, "Interaction");
    param_group(interaction_tracking, "Interaction");
    param_group(attack,  "Envelope");
    param_group(decay,   "Envelope");
    param_group(sustain, "Envelope");
    param_group(release, "Envelope");

    out.push_back(&wavetable_source);
    out.push_back(&wavetable_family);
    out.push_back(&wavetable_member);
    out.push_back(&wav_file);
    out.push_back(&position);
    out.push_back(&amplitude);
    out.push_back(&warp_mode);
    out.push_back(&warp_amount);
    out.push_back(&position_smooth_ms);
    out.push_back(&warp_smooth_ms);
    out.push_back(&phase_reset_mode);
    out.push_back(&start_phase);
    out.push_back(&phase_random);
    out.push_back(&stereo_phase_offset);
    out.push_back(&drift_amount);
    out.push_back(&drift_rate_hz);
    out.push_back(&unison_voices);
    out.push_back(&unison_spread);
    out.push_back(&unison_stereo);
    out.push_back(&unison_spread_mode);
    out.push_back(&unison_output_mode);
    out.push_back(&detune);
    out.push_back(&portamento);
    out.push_back(&interaction_mode);
    out.push_back(&interaction_depth);
    out.push_back(&interaction_input_gain);
    out.push_back(&interaction_tracking);
    out.push_back(&attack);
    out.push_back(&decay);
    out.push_back(&sustain);
    out.push_back(&release);
}

void WavetableOsc::collect_ports(std::vector<VividPortDescriptor>& out) {
    // Canonical native note input — drive directly from Tracker/NotePattern/etc.
    // First port so it's the obvious primary connection.
    out.push_back(VIVID_CUSTOM_REF_PORT("notes_in", VIVID_PORT_INPUT, VividNoteBuffer)); // 0
    out.push_back({"frequencies",  VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 1
    out.push_back({"gates",        VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 2
    out.push_back({"velocities",   VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 3
    out.push_back({"pitch_mod",    VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 4
    out.push_back({"position_mod", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 5
    out.push_back({"warp_mod",     VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 6
    out.push_back({"lane_ids",     VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 7
    out.push_back({"mod_input", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                   VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});         // 8
    out.push_back({"pitch_mod_audio", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                   VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});         // 9
    out.push_back({"position_mod_audio", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                   VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});         // 10
    out.push_back({"warp_mod_audio", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                   VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});         // 11

    // Primary stereo output — sum of all active voices. Default user path
    // (notes_in → output → audio_out) sees this. Output 0.
    out.push_back({"output", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_OUTPUT,
                   VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 2});

    // Advanced per-voice breakouts. voices_out is the multichannel buffer
    // (one channel per voice slot) that previously lived on the `output`
    // port; downstream VoiceMixer/VoiceDrive consumers wire to voices_out
    // now. In MIDI mode channels are sorted by note_id; in lane-driven
    // mode they follow the input lane positions.
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

const Wavetable* WavetableOsc::resolve_table() const {
    const auto& builtins = builtin_tables();
    int source_mode = std::clamp(wavetable_source.int_value(),
                                 static_cast<int>(SOURCE_BUILTIN),
                                 static_cast<int>(SOURCE_CUSTOM));
    if (source_mode == SOURCE_CUSTOM) {
        const Wavetable* custom = custom_table_.load(std::memory_order_acquire);
        if (custom && custom->frame_count > 0) return custom;
    }
    const Wavetable* builtin = resolve_builtin_wavetable(
        builtins.data(), wavetable_family.int_value(), wavetable_member.int_value());
    return builtin ? builtin : &builtins[0];
}

const std::array<Wavetable, kBuiltinWavetableCount>& WavetableOsc::builtin_tables() {
    static const std::array<Wavetable, kBuiltinWavetableCount> tables = []() {
        std::array<Wavetable, kBuiltinWavetableCount> built{};
        build_builtin_wavetables(built.data(), built.size());
        return built;
    }();
    return tables;
}

float WavetableOsc::wrap_phase(double phase) {
    return vivid_wavetable::voice::wrap_phase(phase);
}

float WavetableOsc::smoothing_coeff(float sample_rate, float smooth_ms) {
    return vivid_wavetable::voice::smoothing_coeff(sample_rate, smooth_ms);
}

uint32_t WavetableOsc::hash_u32(uint32_t x) {
    return vivid_wavetable::voice::hash_u32(x);
}

float WavetableOsc::hash01(uint32_t seed) {
    return vivid_wavetable::voice::hash01(seed);
}

float WavetableOsc::normalized_unison_position(int index, int count) {
    return vivid_wavetable::voice::normalized_unison_position(index, count);
}

float WavetableOsc::unison_detune_offset(int index, int count, float spread_cents, int spread_mode, uint32_t lane_seed) {
    return vivid_wavetable::voice::unison_detune_offset(index, count, spread_cents, spread_mode, lane_seed);
}

float WavetableOsc::unison_pan_position(int index, int count, float stereo_depth) {
    return vivid_wavetable::voice::unison_pan_position(index, count, stereo_depth);
}

float WavetableOsc::base_phase_offset(int index, int count, bool stereo_pairs, float stereo_phase, uint32_t lane_seed) {
    return vivid_wavetable::voice::base_phase_offset(index, count, stereo_pairs, stereo_phase, lane_seed);
}

float WavetableOsc::stereo_pair_phase_shift(int index, int count, float stereo_phase, uint32_t lane_seed) {
    return vivid_wavetable::voice::stereo_pair_phase_shift(index, count, stereo_phase, lane_seed);
}

float WavetableOsc::gate_on_phase(PhaseResetMode mode,
                                  float start_phase_value,
                                  float phase_random_amount,
                                  float base_offset,
                                  int index,
                                  uint32_t lane_seed) {
    return vivid_wavetable::voice::gate_on_phase(static_cast<int>(mode), start_phase_value,
                                                  phase_random_amount, base_offset, index, lane_seed);
}

VIVID_REGISTER(WavetableOsc)
VIVID_THUMBNAIL(WavetableOsc)
VIVID_EDITOR(WavetableOsc)
