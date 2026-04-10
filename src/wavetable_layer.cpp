#include "wavetable_layer_internal.h"

#include "lane_audio_utils.h"
#include "wavetable_voice_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using vivid_wavetable::bank::build_builtin_wavetables;
using vivid_wavetable::bank::load_wavetable_from_wav;
using vivid_wavetable::bank::resolve_builtin_wavetable;
using namespace vivid_wavetable::layer;

namespace {

int decode_layer_warp_mode(int param_value) {
    switch (param_value) {
        case 1: return vivid_wavetable::dsp::WARP_SYNC;
        case 2: return vivid_wavetable::dsp::WARP_BEND_PLUS;
        case 3: return vivid_wavetable::dsp::WARP_BEND_MINUS;
        case 4: return vivid_wavetable::dsp::WARP_MIRROR;
        case 5: return vivid_wavetable::dsp::WARP_ASYM;
        case 6: return vivid_wavetable::dsp::WARP_QUANTIZE;
        case 7: return vivid_wavetable::dsp::WARP_FLIP;
        default: return vivid_wavetable::dsp::WARP_NONE;
    }
}

} // namespace

WavetableLayer::WavetableLayer() {
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
}

WavetableLayer::~WavetableLayer() {
    delete custom_table_.load(std::memory_order_relaxed);
    delete deferred_delete_;
}

void WavetableLayer::prepare_instance_assets() {
    (void)builtin_tables();
}

void WavetableLayer::main_thread_update(double) {
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

void WavetableLayer::collect_params(std::vector<vivid::ParamBase*>& out) {
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
    param_group(drift_amount, "Motion");
    param_group(drift_rate_hz, "Motion");
    param_group(phase_reset_mode, "Phase");
    param_group(start_phase, "Phase");
    param_group(phase_random, "Phase");
    param_group(stereo_phase_offset, "Phase");
    param_group(unison_voices, "Unison");
    param_group(unison_spread, "Unison");
    param_group(unison_stereo, "Unison");
    param_group(unison_spread_mode, "Unison");
    param_group(detune, "Output");
    param_group(portamento, "Portamento");

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
    out.push_back(&drift_amount);
    out.push_back(&drift_rate_hz);
    out.push_back(&phase_reset_mode);
    out.push_back(&start_phase);
    out.push_back(&phase_random);
    out.push_back(&stereo_phase_offset);
    out.push_back(&unison_voices);
    out.push_back(&unison_spread);
    out.push_back(&unison_stereo);
    out.push_back(&unison_spread_mode);
    out.push_back(&detune);
    out.push_back(&portamento);
}

void WavetableLayer::collect_ports(std::vector<VividPortDescriptor>& out) {
    // Lane-array inputs (per-voice control data from PolyVoiceAllocator)
    out.push_back({"frequencies",  VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});
    out.push_back({"gates",        VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});
    out.push_back({"velocities",   VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});
    out.push_back({"lane_ids",     VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});
    out.push_back({"pitch_mod",    VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});
    out.push_back({"position_mod", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});
    out.push_back({"warp_mod",     VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});

    // Audio-rate modulation inputs (auto-channel, one channel per voice)
    out.push_back({"pitch_mod_audio", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                   VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});
    out.push_back({"position_mod_audio", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                   VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});
    out.push_back({"warp_mod_audio", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                   VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});
    out.push_back({"voice_gain_audio", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                   VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});

    // Stereo output (always 2 channels)
    out.push_back({"output", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_OUTPUT,
                   VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 2});
}

void WavetableLayer::process_audio(const VividAudioContext* ctx) {
    uint32_t frames = ctx->buffer_size;
    float sr = static_cast<float>(ctx->sample_rate);
    float* out = ctx->output_buffers[0];
    std::memset(out, 0, 2 * frames * sizeof(float));

    const Wavetable* wt = resolve_table();
    if (!wt || wt->frame_count == 0) return;

    // Lazy rebuild guard-sample storage when table pointer changes
    if (wt != cached_table_) {
        prepared_wt_.prepare_from(*wt);
        cached_table_ = wt;
    }

    // Read lane inputs (port indices match collect_ports order)
    const VividLaneView* freq_lane = ctx->input_lanes ? &ctx->input_lanes[0] : nullptr;
    const VividLaneView* gates_lane = ctx->input_lanes ? &ctx->input_lanes[1] : nullptr;
    const VividLaneView* vel_lane = ctx->input_lanes ? &ctx->input_lanes[2] : nullptr;
    const VividLaneView* lid_lane = ctx->input_lanes ? &ctx->input_lanes[3] : nullptr;
    const VividLaneView* pitch_lane = ctx->input_lanes ? &ctx->input_lanes[4] : nullptr;
    const VividLaneView* pos_lane = ctx->input_lanes ? &ctx->input_lanes[5] : nullptr;
    const VividLaneView* warp_lane = ctx->input_lanes ? &ctx->input_lanes[6] : nullptr;

    // Audio-rate mod buffers (port indices 7-10)
    float* pitch_mod_buf = ctx->input_buffers[7];
    uint32_t pitch_mod_ch = pitch_mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[7] : 0;
    float* pos_mod_buf = ctx->input_buffers[8];
    uint32_t pos_mod_ch = pos_mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[8] : 0;
    float* warp_mod_buf = ctx->input_buffers[9];
    uint32_t warp_mod_ch = warp_mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[9] : 0;
    float* gain_mod_buf = ctx->input_buffers[10];
    uint32_t gain_mod_ch = gain_mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[10] : 0;

    uint32_t voice_count = freq_lane ? freq_lane->length : 0;
    if (voice_count > static_cast<uint32_t>(kMaxVoices)) voice_count = kMaxVoices;

    // Extract params
    RenderParams rp;
    rp.warp_mode = decode_layer_warp_mode(warp_mode.int_value());
    rp.amplitude = amplitude.value;
    rp.position_base = std::clamp(position.value, 0.0f, 1.0f);
    rp.warp_base = std::clamp(warp_amount.value, 0.0f, 1.0f);
    rp.drift_amount = std::clamp(drift_amount.value, 0.0f, 1.0f);
    rp.drift_rate_hz = std::clamp(drift_rate_hz.value, 0.02f, 2.0f);
    rp.drift_enabled = rp.drift_amount > 1.0e-6f;
    rp.pos_smooth_coeff = vivid_wavetable::voice::smoothing_coeff(sr, position_smooth_ms.value);
    rp.warp_smooth_coeff = vivid_wavetable::voice::smoothing_coeff(sr, warp_smooth_ms.value);
    rp.num_unison = std::clamp(unison_voices.int_value(), 1, kMaxUnisonVoices);
    rp.unison_spread = unison_spread.value;
    rp.unison_stereo = unison_stereo.value;
    rp.unison_spread_mode = unison_spread_mode.int_value();
    rp.detune_cents = detune.value;
    rp.portamento_ms = portamento.value;
    rp.phase_reset_mode = std::clamp(phase_reset_mode.int_value(), 0, 2);
    rp.start_phase = std::clamp(start_phase.value, 0.0f, 1.0f);
    rp.phase_random = std::clamp(phase_random.value, 0.0f, 1.0f);
    rp.stereo_phase_offset = std::clamp(stereo_phase_offset.value, 0.0f, 1.0f);

    float unison_gain = rp.amplitude / std::sqrt(static_cast<float>(rp.num_unison));
    float porta_rate = 1.0f;
    if (rp.portamento_ms > 0.0f) {
        float porta_samples = rp.portamento_ms * 0.001f * sr;
        porta_rate = 1.0f - std::exp(-4.0f / porta_samples);
    }

    // Build active render list
    render_units_.clear();
    voice_block_ = VoiceBlock{};
    voice_block_.voice_count = static_cast<int>(voice_count);

    int slot = 0;
    for (uint32_t vi = 0; vi < voice_count; ++vi) {
        using namespace vivid_wavetable::lane_audio;
        float gate = read_lane(gates_lane, vi, 0.0f);
        float freq_target = read_lane(freq_lane, vi, 0.0f);
        if (!std::isfinite(freq_target) || freq_target <= 0.0f) continue;

        uint32_t lid = resolve_lane_id(lid_lane, vi);
        Voice& v = *vivid_lane_state(ctx, lid, Voice);

        // Gate-on handling
        bool gate_on = gate > 0.5f;
        if (gate_on && !v.was_gated) {
            if (!v.initialized || rp.phase_reset_mode != PHASE_FREE_RUN) {
                for (int ui = 0; ui < rp.num_unison; ++ui) {
                    float offset = vivid_wavetable::voice::base_phase_offset(
                        ui, rp.num_unison, true, rp.stereo_phase_offset, lid);
                    v.phase[ui] = vivid_wavetable::voice::gate_on_phase(
                        rp.phase_reset_mode, rp.start_phase, rp.phase_random, offset, ui, lid);
                    v.drift_phase[ui] = vivid_wavetable::voice::hash01(lid + static_cast<uint32_t>(ui * 211))
                                        * 2.0f * static_cast<float>(M_PI);
                }
                // Read lane modulation for smoother reset targets
                float pos_mod_val = read_lane(pos_lane, vi, 0.0f);
                float warp_mod_val = read_lane(warp_lane, vi, 0.0f);
                v.pos_smoother.reset(std::clamp(rp.position_base + pos_mod_val, 0.0f, 1.0f));
                v.warp_smoother.reset(std::clamp(rp.warp_base + warp_mod_val, 0.0f, 1.0f));
                v.declick_remaining = kDeClickSamples;
                v.initialized = true;
            }
            if (!v.initialized) {
                v.current_freq = freq_target;
                v.initialized = true;
            }
        }
        v.was_gated = gate_on;
        v.target_freq = freq_target;

        // Portamento
        if (rp.portamento_ms > 0.0f && v.current_freq > 0.0f) {
            if (std::abs(v.target_freq - v.current_freq) > 0.01f) {
                v.current_freq += (v.target_freq - v.current_freq) * porta_rate;
            } else {
                v.current_freq = v.target_freq;
            }
        } else {
            v.current_freq = v.target_freq;
        }

        float base_freq = v.current_freq;

        // Resolve audio-rate mod channel pointers for this voice
        voice_block_.pitch_mod_audio[vi] = resolve_mod_channel(pitch_mod_buf, pitch_mod_ch, vi, frames);
        voice_block_.position_mod_audio[vi] = resolve_mod_channel(pos_mod_buf, pos_mod_ch, vi, frames);
        voice_block_.warp_mod_audio[vi] = resolve_mod_channel(warp_mod_buf, warp_mod_ch, vi, frames);
        voice_block_.voice_gain_audio[vi] = resolve_mod_channel(gain_mod_buf, gain_mod_ch, vi, frames);
        voice_block_.pos_smoother[vi] = &v.pos_smoother;
        voice_block_.warp_smoother[vi] = &v.warp_smoother;
        voice_block_.declick_remaining[vi] = v.declick_remaining;
        voice_block_.pitch_lane_base[vi] = read_lane(pitch_lane, vi, 0.0f);
        voice_block_.position_lane_base[vi] = read_lane(pos_lane, vi, 0.0f);
        voice_block_.warp_lane_base[vi] = read_lane(warp_lane, vi, 0.0f);

        if (!v.pos_smoother.initialized) {
            v.pos_smoother.reset(std::clamp(rp.position_base + voice_block_.position_lane_base[vi], 0.0f, 1.0f));
        }
        if (!v.warp_smoother.initialized) {
            v.warp_smoother.reset(std::clamp(rp.warp_base + voice_block_.warp_lane_base[vi], 0.0f, 1.0f));
        }

        // Initialize sub-block interpolation from the current smoother state.
        voice_block_.pos_to[vi] = v.pos_smoother.value;
        voice_block_.pos_from[vi] = voice_block_.pos_to[vi];
        voice_block_.warp_to[vi] = v.warp_smoother.value;
        voice_block_.warp_from[vi] = voice_block_.warp_to[vi];

        // Fill SoA render units for each unison sub-voice
        for (int ui = 0; ui < rp.num_unison; ++ui) {
            if (slot >= kMaxRenderUnits) break;

            float det_offset = vivid_wavetable::voice::unison_detune_offset(
                ui, rp.num_unison, rp.unison_spread, rp.unison_spread_mode, lid);
            float total_detune = rp.detune_cents + det_offset;
            float detune_ratio = vivid_wavetable::dsp::cents_to_ratio(total_detune);
            float unit_freq = base_freq * detune_ratio;

            render_units_.phase[slot] = v.phase[ui];
            render_units_.phase_inc[slot] = unit_freq / sr;
            render_units_.drift_phase[slot] = v.drift_phase[ui];
            render_units_.drift_phase_inc[slot] = rp.drift_rate_hz * 2.0f * static_cast<float>(M_PI) / sr;
            render_units_.mip_level[slot] = quantized_mip_level(unit_freq, sr);

            // Pan: always stereo pairs for WavetableLayer
            float pan_pos = vivid_wavetable::voice::unison_pan_position(
                ui, rp.num_unison, rp.unison_stereo);
            float angle = (pan_pos * 0.5f + 0.5f) * static_cast<float>(M_PI) * 0.5f;
            render_units_.pan_l[slot] = std::cos(angle);
            render_units_.pan_r[slot] = std::sin(angle);

            render_units_.gain[slot] = unison_gain;
            render_units_.voice_idx[slot] = static_cast<int32_t>(vi);
            ++slot;
        }
    }
    render_units_.active_count = slot;

    // Zero-pad SoA remainder for safe SIMD over-read
    for (int i = slot; i < slot + 16 && i < kMaxRenderUnits; ++i) {
            render_units_.phase[i] = 0.0f;
            render_units_.phase_inc[i] = 0.0f;
            render_units_.gain[i] = 0.0f;
            render_units_.pan_l[i] = 0.0f;
            render_units_.pan_r[i] = 0.0f;
            render_units_.mip_level[i] = 0;
            render_units_.voice_idx[i] = 0;
        }

    if (render_units_.active_count == 0) return;

    // Render
#ifdef VIVID_HAS_HIGHWAY
    if (render_units_.active_count >= 4) {
        render_block_simd(out, frames, sr, render_units_, voice_block_, prepared_wt_, rp);
    } else {
        render_block_scalar(out, frames, sr, render_units_, voice_block_, prepared_wt_, rp);
    }
#else
    render_block_scalar(out, frames, sr, render_units_, voice_block_, prepared_wt_, rp);
#endif

    // Write back phase and drift state to identity-keyed voices
    slot = 0;
    for (uint32_t vi = 0; vi < voice_count; ++vi) {
        using namespace vivid_wavetable::lane_audio;
        float gate = read_lane(gates_lane, vi, 0.0f);
        float freq_target = read_lane(freq_lane, vi, 0.0f);
        if (!std::isfinite(freq_target) || freq_target <= 0.0f) continue;

        uint32_t lid = resolve_lane_id(lid_lane, vi);
        Voice& v = *vivid_lane_state(ctx, lid, Voice);

        for (int ui = 0; ui < rp.num_unison; ++ui) {
            if (slot >= render_units_.active_count) break;
            v.phase[ui] = render_units_.phase[slot];
            v.drift_phase[ui] = render_units_.drift_phase[slot];
            ++slot;
        }
        v.declick_remaining = voice_block_.declick_remaining[vi];
    }
}

const Wavetable* WavetableLayer::resolve_table() const {
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

const std::array<Wavetable, kBuiltinWavetableCount>& WavetableLayer::builtin_tables() {
    static const std::array<Wavetable, kBuiltinWavetableCount> tables = []() {
        std::array<Wavetable, kBuiltinWavetableCount> built{};
        build_builtin_wavetables(built.data(), built.size());
        return built;
    }();
    return tables;
}

VIVID_REGISTER(WavetableLayer)
