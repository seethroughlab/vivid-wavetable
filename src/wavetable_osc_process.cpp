#include "wavetable_osc_internal.h"

#include "lane_audio_utils.h"
#include "voice_breakouts.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace vivid_wavetable::dsp;

// Always-MIDI dispatcher. Port layout (post-Phase 3 trim): notes_in=0,
// mod_input=1, pitch_mod_audio=2, position_mod_audio=3, warp_mod_audio=4,
// output=5, voices_out=6, voice_ids/gates/velocities/freqs (lane outs).
// process_audio_lane_driven is now an internal renderer driven by
// process_audio_midi via a synthesized sub-context.
void WavetableOsc::process_audio(const VividAudioContext* ctx) {
    process_audio_midi(ctx);
}

void WavetableOsc::process_audio_lane_driven(const VividAudioContext* ctx) {
    constexpr uint32_t kControlSubblock = 4;

    uint32_t frames = ctx->buffer_size;
    float sr = static_cast<float>(ctx->sample_rate);

    const Wavetable& wt = *resolve_table();

    float pos = position.value;
    float amp = amplitude.value;
    int warp_m = warp_mode.int_value();
    float warp_a = warp_amount.value;
    float pos_smooth_coeff = smoothing_coeff(sr, position_smooth_ms.value);
    float warp_smooth_coeff = smoothing_coeff(sr, warp_smooth_ms.value);
    PhaseResetMode reset_mode = static_cast<PhaseResetMode>(std::clamp(phase_reset_mode.int_value(), 0, 2));
    float start_phase_value = std::clamp(start_phase.value, 0.0f, 1.0f);
    float phase_random_amount = std::clamp(phase_random.value, 0.0f, 1.0f);
    float stereo_phase = std::clamp(stereo_phase_offset.value, 0.0f, 1.0f);
    float drift_amt = std::clamp(drift_amount.value, 0.0f, 1.0f);
    float drift_rate = std::clamp(drift_rate_hz.value, 0.02f, 2.0f);
    int num_uni = std::clamp(unison_voices.int_value(), 1, kMaxUnisonVoices);
    float uni_spr = unison_spread.value;
    float uni_stereo = unison_stereo.value;
    int uni_spread_mode = unison_spread_mode.int_value();
    int uni_output_mode = std::clamp(unison_output_mode.int_value(), 0, 1);
    float det = detune.value;
    float porta_ms = portamento.value;
    InteractionMode interaction = static_cast<InteractionMode>(std::clamp(interaction_mode.int_value(), 0, 4));
    float interaction_depth_value = interaction_depth.value;
    float interaction_input_gain_value = interaction_input_gain.value;
    float interaction_tracking_value = interaction_tracking.value;
    bool stereo_pairs = (uni_output_mode == UNISON_OUTPUT_STEREO_PAIRS);

    const VividLaneView* freq_lane = ctx->input_lanes ? &ctx->input_lanes[1] : nullptr;
    const VividLaneView* gates_lane = ctx->input_lanes ? &ctx->input_lanes[2] : nullptr;
    const VividLaneView* pitch_lane = ctx->input_lanes ? &ctx->input_lanes[4] : nullptr;
    const VividLaneView* pos_mod_lane = ctx->input_lanes ? &ctx->input_lanes[5] : nullptr;
    const VividLaneView* warp_mod_lane = ctx->input_lanes ? &ctx->input_lanes[6] : nullptr;
    const VividLaneView* lane_id_lane = ctx->input_lanes ? &ctx->input_lanes[7] : nullptr;

    uint32_t voice_count = freq_lane ? freq_lane->length : 0;
    uint32_t max_note_voices = stereo_pairs ? static_cast<uint32_t>(kMaxStereoPairVoices)
                                            : static_cast<uint32_t>(kMaxVoices);
    if (voice_count > max_note_voices) voice_count = max_note_voices;

    float porta_rate = 1.0f;
    if (porta_ms > 0.0f) {
        float porta_samples = porta_ms * 0.001f * sr;
        porta_rate = 1.0f - std::exp(-4.0f / porta_samples);
    }

    const bool interaction_enabled =
        (interaction > INTERACTION_OFF && interaction_depth_value > 1.0e-6f && ctx->input_buffers[8]);
    float* mod_buf = interaction_enabled ? ctx->input_buffers[8] : nullptr;
    uint32_t mod_channels = mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[8] : 0;
    float* pitch_mod_buf = ctx->input_buffers[9];
    uint32_t pitch_mod_ch = pitch_mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[9] : 0;
    float* pos_mod_buf = ctx->input_buffers[10];
    uint32_t pos_mod_ch = pos_mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[10] : 0;
    float* warp_mod_buf = ctx->input_buffers[11];
    uint32_t warp_mod_ch = warp_mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[11] : 0;

    // voices_out is the multichannel per-voice audio buffer (post-Phase-2:
    // moved off `output` to its own advanced port). The stereo `output`
    // (port 0) gets the summed mix at the end of this function.
    float* voices_out_buf = (ctx->output_buffers && ctx->output_buffers[1])
                            ? ctx->output_buffers[1] : nullptr;
    float* out_buf = voices_out_buf ? voices_out_buf : ctx->output_buffers[0];
    std::memset(out_buf, 0, kMaxVoices * frames * sizeof(float));

    float unison_gain = amp / std::sqrt(static_cast<float>(num_uni));
    const bool warp_uses_feedback = (warp_m == WARP_FM);
    const bool drift_enabled = drift_amt > 1.0e-6f;
    const bool warp_identity = (warp_m == WARP_NONE);

    for (uint32_t vi = 0; vi < voice_count; ++vi) {
        float gate = vivid_wavetable::lane_audio::read_lane(gates_lane, vi, 0.0f);
        float freq_target = vivid_wavetable::lane_audio::read_lane(freq_lane, vi, 0.0f);
        if (!std::isfinite(freq_target) || freq_target <= 0.0f) continue;

        uint32_t lid = vivid_wavetable::lane_audio::resolve_lane_id(lane_id_lane, vi);
        Voice& v = *vivid_lane_state(ctx, lid, Voice);

        bool gate_on = gate > 0.5f;
        if (gate_on && !v.was_gated) {
            if (!v.initialized || reset_mode != PHASE_FREE_RUN) {
                for (int ui = 0; ui < num_uni; ++ui) {
                    float phase = gate_on_phase(reset_mode,
                                                start_phase_value,
                                                phase_random_amount,
                                                base_phase_offset(ui, num_uni, stereo_pairs, stereo_phase, lid),
                                                ui,
                                                lid);
                    if (reset_mode == PHASE_FREE_RUN && v.initialized) {
                        phase = wrap_phase(v.phase[ui]);
                    }
                    v.phase[ui] = phase;
                    v.last_sample[ui] = 0.0f;
                    v.drift_phase[ui] = hash01(lid + static_cast<uint32_t>(ui * 211))
                        * static_cast<float>(2.0 * M_PI);
                }
                float target_pos = std::clamp(
                    pos + vivid_wavetable::lane_audio::read_lane(pos_mod_lane, vi, 0.0f), 0.0f, 1.0f);
                float target_warp = std::clamp(
                    warp_a + vivid_wavetable::lane_audio::read_lane(warp_mod_lane, vi, 0.0f), 0.0f, 1.0f);
                v.pos_smoother.reset(target_pos);
                v.warp_smoother.reset(target_warp);
                v.declick_remaining = (reset_mode == PHASE_FREE_RUN) ? 0 : kDeClickSamples;
            }
            if (!v.initialized) {
                v.current_freq = freq_target;
                v.target_freq = freq_target;
            }
            v.initialized = true;
            v.interaction_dc.reset();
        }
        v.was_gated = gate_on;
        v.target_freq = freq_target;
        if (!std::isfinite(v.current_freq) || v.current_freq <= 0.0f) {
            v.current_freq = freq_target;
        }

        float* ch_out_l = out_buf + (stereo_pairs ? vi * 2 : vi) * frames;
        float* ch_out_r = stereo_pairs ? (out_buf + (vi * 2 + 1) * frames) : nullptr;

        float pitch_offset_lane = vivid_wavetable::lane_audio::read_lane(pitch_lane, vi, 0.0f);
        float pos_mod_lane_val = vivid_wavetable::lane_audio::read_lane(pos_mod_lane, vi, 0.0f);
        float warp_mod_lane_val = vivid_wavetable::lane_audio::read_lane(warp_mod_lane, vi, 0.0f);

        float* mod_ch = vivid_wavetable::lane_audio::resolve_mod_channel(mod_buf, mod_channels, vi, frames);
        float* pitch_mod_voice = vivid_wavetable::lane_audio::resolve_mod_channel(
            pitch_mod_buf, pitch_mod_ch, vi, frames);
        float* pos_mod_voice = vivid_wavetable::lane_audio::resolve_mod_channel(
            pos_mod_buf, pos_mod_ch, vi, frames);
        float* warp_mod_voice = vivid_wavetable::lane_audio::resolve_mod_channel(
            warp_mod_buf, warp_mod_ch, vi, frames);

        const bool has_pitch_mod = pitch_mod_voice || std::abs(pitch_offset_lane) > 1.0e-6f;
        const bool has_pos_mod = pos_mod_voice || std::abs(pos_mod_lane_val) > 1.0e-6f;
        const bool has_warp_mod = warp_mod_voice || std::abs(warp_mod_lane_val) > 1.0e-6f;
        const bool block_stable_frequency = (!interaction_enabled && !drift_enabled && porta_ms <= 0.0f && !has_pitch_mod);

        float unison_detune_ratio[kMaxUnisonVoices] = {};
        float unison_pan_left[kMaxUnisonVoices] = {};
        float unison_pan_right[kMaxUnisonVoices] = {};
        float unison_phase_shift[kMaxUnisonVoices] = {};
        bool unison_has_phase_shift[kMaxUnisonVoices] = {};
        float unison_drift_phase_inc[kMaxUnisonVoices] = {};
        float stable_phase_inc[kMaxUnisonVoices] = {};
        vivid_wavetable::bank::PreparedMipPlan stable_mip_plan[kMaxUnisonVoices] = {};
        for (int ui = 0; ui < num_uni; ++ui) {
            float detune_cents = det + unison_detune_offset(ui, num_uni, uni_spr, uni_spread_mode, lid);
            unison_detune_ratio[ui] = cents_to_ratio(detune_cents);
            if (stereo_pairs) {
                float pan = std::clamp(unison_pan_position(ui, num_uni, uni_stereo), -1.0f, 1.0f);
                float theta = (pan + 1.0f) * static_cast<float>(M_PI) * 0.25f;
                unison_pan_left[ui] = std::cos(theta);
                unison_pan_right[ui] = std::sin(theta);
                unison_phase_shift[ui] = stereo_pair_phase_shift(ui, num_uni, stereo_phase, lid);
                unison_has_phase_shift[ui] = std::abs(unison_phase_shift[ui]) > 0.00001f;
            }
            if (drift_enabled) {
                float drift_seed = hash01(lid + static_cast<uint32_t>(ui * 379));
                float drift_rate_scale = 0.7f + drift_seed * 0.8f;
                unison_drift_phase_inc[ui] = static_cast<float>(
                    (2.0 * M_PI * static_cast<double>(drift_rate) * static_cast<double>(drift_rate_scale))
                    / static_cast<double>(sr));
            }
        }

        auto update_frequency = [&]() {
            if (porta_ms > 0.0f && v.current_freq != v.target_freq) {
                v.current_freq += (v.target_freq - v.current_freq) * porta_rate;
                if (std::abs(v.current_freq - v.target_freq) < 0.01f) {
                    v.current_freq = v.target_freq;
                }
            } else {
                v.current_freq = v.target_freq;
            }
        };

        auto current_pitch_ratio = [&](uint32_t s) {
            if (!has_pitch_mod) return 1.0f;
            float pitch_offset = pitch_mod_voice ? pitch_mod_voice[s] : pitch_offset_lane;
            return (std::abs(pitch_offset) > 1.0e-6f)
                ? std::pow(2.0f, pitch_offset / 12.0f)
                : 1.0f;
        };
        auto gate_gain = [&]() {
            if (v.declick_remaining <= 0) return 1.0f;
            float gain = static_cast<float>(kDeClickSamples - v.declick_remaining + 1)
                / static_cast<float>(kDeClickSamples);
            --v.declick_remaining;
            return gain;
        };
        auto wrap_phase_fast = [](float phase) {
            if (phase >= 1.0f) {
                phase -= 1.0f;
                if (phase >= 1.0f) phase -= std::floor(phase);
            } else if (phase < 0.0f) {
                phase += 1.0f;
                if (phase < 0.0f) phase -= std::floor(phase);
            }
            return phase;
        };
        auto advance_phase = [&](int ui, float phase_inc) {
            double next = v.phase[ui] + static_cast<double>(phase_inc);
            if (!std::isfinite(next)) {
                v.phase[ui] = 0.0;
                return;
            }
            if (next >= 1.0) {
                next -= 1.0;
                if (next >= 1.0) next -= std::floor(next);
            } else if (next < 0.0) {
                next += 1.0;
                if (next < 0.0) next -= std::floor(next);
            }
            v.phase[ui] = next;
        };
        auto render_subblock_targets = [&](uint32_t s0, float& pos_from, float& pos_to, float& warp_from, float& warp_to) {
            pos_from = v.pos_smoother.value;
            warp_from = v.warp_smoother.value;
            float pos_mod_val = has_pos_mod ? (pos_mod_voice ? pos_mod_voice[s0] : pos_mod_lane_val) : 0.0f;
            float warp_mod_val = has_warp_mod ? (warp_mod_voice ? warp_mod_voice[s0] : warp_mod_lane_val) : 0.0f;
            pos_to = v.pos_smoother.process(std::clamp(pos + pos_mod_val, 0.0f, 1.0f), pos_smooth_coeff);
            warp_to = v.warp_smoother.process(std::clamp(warp_a + warp_mod_val, 0.0f, 1.0f), warp_smooth_coeff);
        };
        auto render_sample_value = [&](float phase,
                                       float smooth_warp,
                                       float feedback_sample,
                                       const vivid_wavetable::bank::PreparedFramePlan& frame_plan,
                                       const vivid_wavetable::bank::PreparedMipPlan& mip_plan) {
            float warped = warp_identity ? phase : warp_phase(phase, warp_m, smooth_warp, feedback_sample);
            return wt.sample_prepared(warped, frame_plan, mip_plan);
        };

        if (block_stable_frequency) {
            update_frequency();
            for (int ui = 0; ui < num_uni; ++ui) {
                float base_freq = v.current_freq * unison_detune_ratio[ui];
                if (!std::isfinite(base_freq) || base_freq <= 0.0f) {
                    base_freq = std::max(v.current_freq, 1.0f);
                }
                stable_phase_inc[ui] = base_freq / sr;
                stable_mip_plan[ui] = wt.prepare_mip_plan(base_freq, sr, true);
            }

            if (stereo_pairs) {
                for (uint32_t sb = 0; sb < frames; sb += kControlSubblock) {
                    uint32_t block_len = std::min(kControlSubblock, frames - sb);
                    float pos_from, pos_to, warp_from, warp_to;
                    render_subblock_targets(sb, pos_from, pos_to, warp_from, warp_to);

                    for (uint32_t offset = 0; offset < block_len; ++offset) {
                        uint32_t s = sb + offset;
                        float t = static_cast<float>(offset + 1) / static_cast<float>(block_len);
                        float smooth_pos = pos_from + (pos_to - pos_from) * t;
                        float smooth_warp = warp_from + (warp_to - warp_from) * t;
                        vivid_wavetable::bank::PreparedFramePlan frame_plan = wt.prepare_frame_plan(smooth_pos);
                        float stereo_l = 0.0f;
                        float stereo_r = 0.0f;

                        for (int ui = 0; ui < num_uni; ++ui) {
                            float phase = wrap_phase_fast(static_cast<float>(v.phase[ui]));
                            float feedback_sample = v.last_sample[ui];
                            float left_sig;
                            float right_sig;
                            if (unison_has_phase_shift[ui]) {
                                float left_phase = wrap_phase_fast(phase + unison_phase_shift[ui]);
                                float right_phase = wrap_phase_fast(phase - unison_phase_shift[ui]);
                                left_sig = render_sample_value(left_phase, smooth_warp, feedback_sample, frame_plan, stable_mip_plan[ui]);
                                right_sig = render_sample_value(right_phase, smooth_warp, feedback_sample, frame_plan, stable_mip_plan[ui]);
                            } else {
                                left_sig = render_sample_value(phase, smooth_warp, feedback_sample, frame_plan, stable_mip_plan[ui]);
                                right_sig = left_sig;
                            }
                            if (warp_uses_feedback) {
                                v.last_sample[ui] = 0.5f * (left_sig + right_sig);
                            }
                            stereo_l += (left_sig * unison_gain) * unison_pan_left[ui];
                            stereo_r += (right_sig * unison_gain) * unison_pan_right[ui];
                            advance_phase(ui, stable_phase_inc[ui]);
                        }

                        float g = gate_gain();
                        ch_out_l[s] = stereo_l * g;
                        ch_out_r[s] = stereo_r * g;
                    }
                }
            } else {
                for (uint32_t sb = 0; sb < frames; sb += kControlSubblock) {
                    uint32_t block_len = std::min(kControlSubblock, frames - sb);
                    float pos_from, pos_to, warp_from, warp_to;
                    render_subblock_targets(sb, pos_from, pos_to, warp_from, warp_to);

                    for (uint32_t offset = 0; offset < block_len; ++offset) {
                        uint32_t s = sb + offset;
                        float t = static_cast<float>(offset + 1) / static_cast<float>(block_len);
                        float smooth_pos = pos_from + (pos_to - pos_from) * t;
                        float smooth_warp = warp_from + (warp_to - warp_from) * t;
                        vivid_wavetable::bank::PreparedFramePlan frame_plan = wt.prepare_frame_plan(smooth_pos);
                        float mono_sum = 0.0f;

                        for (int ui = 0; ui < num_uni; ++ui) {
                            float phase = wrap_phase_fast(static_cast<float>(v.phase[ui]));
                            float sig = render_sample_value(phase, smooth_warp, v.last_sample[ui], frame_plan, stable_mip_plan[ui]);
                            if (warp_uses_feedback) {
                                v.last_sample[ui] = sig;
                            }
                            mono_sum += sig * unison_gain;
                            advance_phase(ui, stable_phase_inc[ui]);
                        }

                        ch_out_l[s] = mono_sum * gate_gain();
                    }
                }
            }
            continue;
        }

        if (!interaction_enabled && !drift_enabled) {
            if (stereo_pairs) {
                for (uint32_t sb = 0; sb < frames; sb += kControlSubblock) {
                    uint32_t block_len = std::min(kControlSubblock, frames - sb);
                    float pos_from, pos_to, warp_from, warp_to;
                    render_subblock_targets(sb, pos_from, pos_to, warp_from, warp_to);

                    for (uint32_t offset = 0; offset < block_len; ++offset) {
                        uint32_t s = sb + offset;
                        update_frequency();
                        float pitch_ratio = current_pitch_ratio(s);
                        float t = static_cast<float>(offset + 1) / static_cast<float>(block_len);
                        float smooth_pos = pos_from + (pos_to - pos_from) * t;
                        float smooth_warp = warp_from + (warp_to - warp_from) * t;
                        vivid_wavetable::bank::PreparedFramePlan frame_plan = wt.prepare_frame_plan(smooth_pos);
                        float stereo_l = 0.0f;
                        float stereo_r = 0.0f;

                        for (int ui = 0; ui < num_uni; ++ui) {
                            float base_freq = v.current_freq * pitch_ratio * unison_detune_ratio[ui];
                            if (!std::isfinite(base_freq) || base_freq <= 0.0f) {
                                base_freq = std::max(v.current_freq, 1.0f);
                            }
                            vivid_wavetable::bank::PreparedMipPlan mip_plan = wt.prepare_mip_plan(base_freq, sr, false);
                            float phase = wrap_phase_fast(static_cast<float>(v.phase[ui]));
                            float feedback_sample = v.last_sample[ui];
                            float left_sig;
                            float right_sig;
                            if (unison_has_phase_shift[ui]) {
                                float left_phase = wrap_phase_fast(phase + unison_phase_shift[ui]);
                                float right_phase = wrap_phase_fast(phase - unison_phase_shift[ui]);
                                left_sig = render_sample_value(left_phase, smooth_warp, feedback_sample, frame_plan, mip_plan);
                                right_sig = render_sample_value(right_phase, smooth_warp, feedback_sample, frame_plan, mip_plan);
                            } else {
                                left_sig = render_sample_value(phase, smooth_warp, feedback_sample, frame_plan, mip_plan);
                                right_sig = left_sig;
                            }
                            if (warp_uses_feedback) {
                                v.last_sample[ui] = 0.5f * (left_sig + right_sig);
                            }
                            stereo_l += (left_sig * unison_gain) * unison_pan_left[ui];
                            stereo_r += (right_sig * unison_gain) * unison_pan_right[ui];
                            advance_phase(ui, base_freq / sr);
                        }

                        float g = gate_gain();
                        ch_out_l[s] = stereo_l * g;
                        ch_out_r[s] = stereo_r * g;
                    }
                }
            } else {
                for (uint32_t sb = 0; sb < frames; sb += kControlSubblock) {
                    uint32_t block_len = std::min(kControlSubblock, frames - sb);
                    float pos_from, pos_to, warp_from, warp_to;
                    render_subblock_targets(sb, pos_from, pos_to, warp_from, warp_to);

                    for (uint32_t offset = 0; offset < block_len; ++offset) {
                        uint32_t s = sb + offset;
                        update_frequency();
                        float pitch_ratio = current_pitch_ratio(s);
                        float t = static_cast<float>(offset + 1) / static_cast<float>(block_len);
                        float smooth_pos = pos_from + (pos_to - pos_from) * t;
                        float smooth_warp = warp_from + (warp_to - warp_from) * t;
                        vivid_wavetable::bank::PreparedFramePlan frame_plan = wt.prepare_frame_plan(smooth_pos);
                        float mono_sum = 0.0f;

                        for (int ui = 0; ui < num_uni; ++ui) {
                            float base_freq = v.current_freq * pitch_ratio * unison_detune_ratio[ui];
                            if (!std::isfinite(base_freq) || base_freq <= 0.0f) {
                                base_freq = std::max(v.current_freq, 1.0f);
                            }
                            vivid_wavetable::bank::PreparedMipPlan mip_plan = wt.prepare_mip_plan(base_freq, sr, false);
                            float phase = wrap_phase_fast(static_cast<float>(v.phase[ui]));
                            float sig = render_sample_value(phase, smooth_warp, v.last_sample[ui], frame_plan, mip_plan);
                            if (warp_uses_feedback) {
                                v.last_sample[ui] = sig;
                            }
                            mono_sum += sig * unison_gain;
                            advance_phase(ui, base_freq / sr);
                        }

                        ch_out_l[s] = mono_sum * gate_gain();
                    }
                }
            }
            continue;
        }

        if (stereo_pairs) {
            for (uint32_t s = 0; s < frames; ++s) {
                update_frequency();
                float pitch_ratio = current_pitch_ratio(s);
                float pos_mod_val = has_pos_mod ? (pos_mod_voice ? pos_mod_voice[s] : pos_mod_lane_val) : 0.0f;
                float warp_mod_val = has_warp_mod ? (warp_mod_voice ? warp_mod_voice[s] : warp_mod_lane_val) : 0.0f;
                float smooth_pos = v.pos_smoother.process(std::clamp(pos + pos_mod_val, 0.0f, 1.0f), pos_smooth_coeff);
                float smooth_warp = v.warp_smoother.process(std::clamp(warp_a + warp_mod_val, 0.0f, 1.0f), warp_smooth_coeff);
                vivid_wavetable::bank::PreparedFramePlan frame_plan = wt.prepare_frame_plan(smooth_pos);
                float stereo_l = 0.0f;
                float stereo_r = 0.0f;
                float mod_sample = mod_ch ? mod_ch[s] : 0.0f;

                for (int ui = 0; ui < num_uni; ++ui) {
                    float base_ratio = unison_detune_ratio[ui];
                    if (drift_enabled) {
                        float drift_cents = std::sin(v.drift_phase[ui]) * drift_amt * kMaxDriftCents;
                        base_ratio *= cents_to_ratio(drift_cents);
                        v.drift_phase[ui] += static_cast<double>(unison_drift_phase_inc[ui]);
                        if (!std::isfinite(v.drift_phase[ui])) {
                            v.drift_phase[ui] = 0.0;
                        } else if (v.drift_phase[ui] > 2.0 * M_PI) {
                            v.drift_phase[ui] -= 2.0 * M_PI;
                        }
                    }

                    float base_freq = v.current_freq * pitch_ratio * base_ratio;
                    if (!std::isfinite(base_freq) || base_freq <= 0.0f) {
                        base_freq = std::max(v.current_freq, 1.0f);
                    }
                    vivid_wavetable::bank::PreparedMipPlan mip_plan = wt.prepare_mip_plan(base_freq, sr, false);
                    float phase_inc = base_freq / sr;
                    float interaction_phase = wrap_phase_fast(static_cast<float>(v.phase[ui]));
                    float gain_comp = unison_gain;
                    InteractionSignal interaction_signal{};
                    if (interaction_enabled) {
                        interaction_signal = prepare_interaction_signal(
                            interaction,
                            base_freq,
                            interaction_depth_value,
                            interaction_input_gain_value,
                            interaction_tracking_value,
                            mod_sample,
                            true,
                            v.interaction_dc);
                        if (interaction == INTERACTION_FM) {
                            phase_inc += interaction_fm_phase_delta(interaction_signal, sr);
                        }
                        if (interaction == INTERACTION_PM) {
                            interaction_phase = wrap_phase_fast(static_cast<float>(v.phase[ui]) + interaction_pm_offset(interaction_signal));
                        }
                        gain_comp *= interaction_output_compensation(interaction, interaction_signal.amount);
                    }

                    float feedback_sample = v.last_sample[ui];
                    float left_sig;
                    float right_sig;
                    if (unison_has_phase_shift[ui]) {
                        float left_phase = wrap_phase_fast(interaction_phase + unison_phase_shift[ui]);
                        float right_phase = wrap_phase_fast(interaction_phase - unison_phase_shift[ui]);
                        left_sig = render_sample_value(left_phase, smooth_warp, feedback_sample, frame_plan, mip_plan);
                        right_sig = render_sample_value(right_phase, smooth_warp, feedback_sample, frame_plan, mip_plan);
                    } else {
                        left_sig = render_sample_value(interaction_phase, smooth_warp, feedback_sample, frame_plan, mip_plan);
                        right_sig = left_sig;
                    }

                    if (warp_uses_feedback) {
                        v.last_sample[ui] = 0.5f * (left_sig + right_sig);
                    }
                    if (interaction_enabled) {
                        if (interaction == INTERACTION_RM) {
                            left_sig = interaction_rm_sample(left_sig, interaction_signal);
                            right_sig = interaction_rm_sample(right_sig, interaction_signal);
                        } else if (interaction == INTERACTION_AM) {
                            float gain = interaction_am_gain(interaction_signal);
                            left_sig *= gain;
                            right_sig *= gain;
                        }
                    }

                    stereo_l += (left_sig * gain_comp) * unison_pan_left[ui];
                    stereo_r += (right_sig * gain_comp) * unison_pan_right[ui];
                    advance_phase(ui, phase_inc);
                }

                float g = gate_gain();
                ch_out_l[s] = stereo_l * g;
                ch_out_r[s] = stereo_r * g;
            }
        } else {
            for (uint32_t s = 0; s < frames; ++s) {
                update_frequency();
                float pitch_ratio = current_pitch_ratio(s);
                float pos_mod_val = has_pos_mod ? (pos_mod_voice ? pos_mod_voice[s] : pos_mod_lane_val) : 0.0f;
                float warp_mod_val = has_warp_mod ? (warp_mod_voice ? warp_mod_voice[s] : warp_mod_lane_val) : 0.0f;
                float smooth_pos = v.pos_smoother.process(std::clamp(pos + pos_mod_val, 0.0f, 1.0f), pos_smooth_coeff);
                float smooth_warp = v.warp_smoother.process(std::clamp(warp_a + warp_mod_val, 0.0f, 1.0f), warp_smooth_coeff);
                vivid_wavetable::bank::PreparedFramePlan frame_plan = wt.prepare_frame_plan(smooth_pos);
                float mono_sum = 0.0f;
                float mod_sample = mod_ch ? mod_ch[s] : 0.0f;

                for (int ui = 0; ui < num_uni; ++ui) {
                    float base_ratio = unison_detune_ratio[ui];
                    if (drift_enabled) {
                        float drift_cents = std::sin(v.drift_phase[ui]) * drift_amt * kMaxDriftCents;
                        base_ratio *= cents_to_ratio(drift_cents);
                        v.drift_phase[ui] += static_cast<double>(unison_drift_phase_inc[ui]);
                        if (!std::isfinite(v.drift_phase[ui])) {
                            v.drift_phase[ui] = 0.0;
                        } else if (v.drift_phase[ui] > 2.0 * M_PI) {
                            v.drift_phase[ui] -= 2.0 * M_PI;
                        }
                    }

                    float base_freq = v.current_freq * pitch_ratio * base_ratio;
                    if (!std::isfinite(base_freq) || base_freq <= 0.0f) {
                        base_freq = std::max(v.current_freq, 1.0f);
                    }
                    vivid_wavetable::bank::PreparedMipPlan mip_plan = wt.prepare_mip_plan(base_freq, sr, false);
                    float phase_inc = base_freq / sr;
                    float interaction_phase = wrap_phase_fast(static_cast<float>(v.phase[ui]));
                    float gain_comp = unison_gain;
                    InteractionSignal interaction_signal{};
                    if (interaction_enabled) {
                        interaction_signal = prepare_interaction_signal(
                            interaction,
                            base_freq,
                            interaction_depth_value,
                            interaction_input_gain_value,
                            interaction_tracking_value,
                            mod_sample,
                            true,
                            v.interaction_dc);
                        if (interaction == INTERACTION_FM) {
                            phase_inc += interaction_fm_phase_delta(interaction_signal, sr);
                        }
                        if (interaction == INTERACTION_PM) {
                            interaction_phase = wrap_phase_fast(static_cast<float>(v.phase[ui]) + interaction_pm_offset(interaction_signal));
                        }
                        gain_comp *= interaction_output_compensation(interaction, interaction_signal.amount);
                    }

                    float sig = render_sample_value(interaction_phase, smooth_warp, v.last_sample[ui], frame_plan, mip_plan);
                    if (warp_uses_feedback) {
                        v.last_sample[ui] = sig;
                    }
                    if (interaction_enabled) {
                        if (interaction == INTERACTION_RM) {
                            sig = interaction_rm_sample(sig, interaction_signal);
                        } else if (interaction == INTERACTION_AM) {
                            sig *= interaction_am_gain(interaction_signal);
                        }
                    }

                    mono_sum += sig * gain_comp;
                    advance_phase(ui, phase_inc);
                }

                ch_out_l[s] = mono_sum * gate_gain();
            }
        }

        // Snapshot voice-0's smoothed effective position for the editor
        // so the stacked-frames view can highlight the live sweep frame.
        if (vi == 0) {
            editor_effective_position_.store(
                std::clamp(v.pos_smoother.value, 0.0f, 1.0f),
                std::memory_order_relaxed);
        }
    }

    // Sum per-voice audio (in voices_out_buf) into the stereo `output`
    // (port 0). When called from process_audio_midi() the recursive call
    // also goes through here, so the stereo sum happens automatically;
    // the MIDI post-step then re-applies ADSR per voice and re-sums into
    // output_buffers[0].
    if (voices_out_buf && ctx->output_buffers && ctx->output_buffers[0]) {
        float* stereo_out = ctx->output_buffers[0];
        std::memset(stereo_out, 0, 2 * frames * sizeof(float));
        const uint32_t channels = stereo_pairs ? voice_count * 2 : voice_count;
        for (uint32_t ch = 0; ch < channels && ch < static_cast<uint32_t>(kMaxVoices); ++ch) {
            const float* src = voices_out_buf + ch * frames;
            // Mono voices: route to both stereo channels. Stereo pairs:
            // even ch -> L, odd ch -> R.
            const bool is_right = stereo_pairs && (ch & 1);
            float* dst_l = is_right ? nullptr : (stereo_out);
            float* dst_r = is_right ? (stereo_out + frames) : (stereo_pairs ? nullptr : (stereo_out + frames));
            for (uint32_t s = 0; s < frames; ++s) {
                if (dst_l) dst_l[s] += src[s];
                if (dst_r) dst_r[s] += src[s];
            }
        }
    }
}

// MIDI-driven path: ingest MIDI events into our internal allocator, build
// synthetic lane-array views, dispatch through process_audio_lane_driven,
// then apply per-slot ADSR and sum to stereo (channels 0/1 of the output).
//
// This pattern preserves the full feature surface (unison, drift, motion
// smoothers, interaction modes, declick) of the lane-driven render, while
// presenting a clean midi_in → stereo experience to the user.
void WavetableOsc::process_audio_midi(const VividAudioContext* ctx) {
    const uint32_t frames = ctx->buffer_size;
    const float sr = static_cast<float>(ctx->sample_rate);
    const auto* notes = static_cast<const VividNoteBuffer*>(ctx->custom_inputs[0]);

    // Drive the allocator. On note-on, reset the envelope; on note-off,
    // start the release tail. Per-voice phase / unison reset happens
    // automatically inside process_audio_lane_driven via the gate edge.
    // Per-note expression (pitch_bend, pressure, timbre) is consumed by
    // the allocator (mutates the matching slot) but the wavetable render
    // path doesn't yet route any of it; that is Phase 2 work.
    midi_allocator_.process_note_buffer(notes, midi_frame_counter_,
        [this](int slot, int /*note*/, float /*vel*/, uint32_t /*offset*/, uint64_t /*note_id*/) {
            vivid::adsr::gate_on(midi_voices_[slot].env);
        },
        [this](int slot, int /*note*/, uint64_t /*note_id*/) {
            vivid::adsr::gate_off(midi_voices_[slot].env);
        },
        [](int /*slot*/, VividNoteEventType /*kind*/, float /*value*/) {
            // Expression events recorded on the slot for future phases.
        });

    // Pack active slots into synthetic lane buffers, sorted by note_id
    // ascending so voices_out + voice_* breakouts share the cross-cutting
    // contract: synths fed the same note stream produce per-voice surfaces
    // in the same order.
    int sorted_idx[kMaxVoices];
    int n_active_int = 0;
    for (int i = 0; i < kMaxVoices; ++i) {
        if (midi_allocator_.slots[i].active) sorted_idx[n_active_int++] = i;
    }
    std::sort(sorted_idx, sorted_idx + n_active_int,
              [this](int a, int b) {
                  return midi_allocator_.slots[a].note_id <
                         midi_allocator_.slots[b].note_id;
              });

    float synth_freqs[kMaxVoices] = {};
    float synth_gates[kMaxVoices] = {};
    float synth_vels [kMaxVoices] = {};
    float synth_zeros[kMaxVoices] = {};
    float synth_lane_ids[kMaxVoices] = {};
    int   slot_for_voice[kMaxVoices] = {};
    uint32_t n_active = static_cast<uint32_t>(n_active_int);
    for (uint32_t k = 0; k < n_active; ++k) {
        const int i = sorted_idx[k];
        const auto& slot = midi_allocator_.slots[i];
        const float voice_freq = 440.0f *
            std::pow(2.0f, (static_cast<float>(slot.note) - 69.0f
                            + slot.pitch_bend_semis) / 12.0f);
        synth_freqs[k]    = voice_freq;
        synth_gates[k]    = slot.gate ? 1.0f : 0.0f;
        synth_vels [k]    = slot.velocity;
        synth_zeros[k]    = 0.0f;
        synth_lane_ids[k] = static_cast<float>(kMidiLaneIdBase + static_cast<uint32_t>(i));
        slot_for_voice[k] = i;
    }

    // Stereo `output` (port 0) and multichannel `voices_out` (port 1).
    // process_audio_lane_driven writes per-voice into voices_out and
    // computes a stereo sum into output; we then re-apply per-voice ADSR
    // here, so we need to re-do both summing steps after envelope.
    float* stereo_out  = ctx->output_buffers && ctx->output_buffers[0]
                         ? ctx->output_buffers[0] : nullptr;
    float* voices_buf  = ctx->output_buffers && ctx->output_buffers[1]
                         ? ctx->output_buffers[1] : nullptr;
    if (n_active == 0) {
        if (stereo_out) std::memset(stereo_out, 0, 2 * frames * sizeof(float));
        if (voices_buf) std::memset(voices_buf, 0, kMaxVoices * frames * sizeof(float));
        midi_frame_counter_ += frames;
        // Still emit the (empty) breakout lanes so downstream lane-driven
        // consumers see consistent shape.
        if (ctx->output_lanes) {
            // Lane outputs are indexed by lane-output-port position only.
            // WavetableOsc output port order: output(0), voices_out(1) are
            // audio buffers; lane outputs are voice_ids(0), voice_gates(1),
            // voice_velocities(2), voice_freqs(3).
            VividLaneOutput lanes[vivid_sequencers::kVoiceBreakoutLaneCount] = {
                ctx->output_lanes[0], ctx->output_lanes[1],
                ctx->output_lanes[2], ctx->output_lanes[3],
            };
            vivid_sequencers::emit_voice_breakouts(midi_allocator_, lanes);
        }
        return;
    }

    // Build a sub-context that points at the synthetic lanes. The internal
    // renderer (process_audio_lane_driven) keeps its historical input
    // indexing — lanes 1..7, audio inputs 8..11 — for stability; we map
    // real ctx audio buffers (now at indices 1..4) into those slots.
    VividLaneView synth_lanes[8] = {};
    synth_lanes[1] = {synth_freqs,    n_active, 0, 0};   // frequencies
    synth_lanes[2] = {synth_gates,    n_active, 0, 0};   // gates
    synth_lanes[3] = {synth_vels,     n_active, 0, 0};   // velocities
    synth_lanes[4] = {synth_zeros,    n_active, 0, 0};   // pitch_mod
    synth_lanes[5] = {synth_zeros,    n_active, 0, 0};   // position_mod
    synth_lanes[6] = {synth_zeros,    n_active, 0, 0};   // warp_mod
    synth_lanes[7] = {synth_lane_ids, n_active, 0, 0};   // lane_ids

    constexpr int kSubInputPorts = 12;  // 1 midi + 7 lanes + 4 audio
    float* sub_input_buffers[kSubInputPorts] = {};
    uint8_t sub_input_channels[kSubInputPorts] = {};
    if (ctx->input_buffers) {
        sub_input_buffers[8]  = ctx->input_buffers[1];   // mod_input
        sub_input_buffers[9]  = ctx->input_buffers[2];   // pitch_mod_audio
        sub_input_buffers[10] = ctx->input_buffers[3];   // position_mod_audio
        sub_input_buffers[11] = ctx->input_buffers[4];   // warp_mod_audio
    }
    if (ctx->input_channel_counts) {
        sub_input_channels[8]  = ctx->input_channel_counts[1];
        sub_input_channels[9]  = ctx->input_channel_counts[2];
        sub_input_channels[10] = ctx->input_channel_counts[3];
        sub_input_channels[11] = ctx->input_channel_counts[4];
    }

    VividAudioContext sub_ctx = *ctx;
    sub_ctx.input_lanes          = synth_lanes;
    sub_ctx.input_buffers        = sub_input_buffers;
    sub_ctx.input_channel_counts = sub_input_channels;
    sub_ctx.custom_inputs        = nullptr;
    sub_ctx.custom_input_count   = 0;

    process_audio_lane_driven(&sub_ctx);

    // Per-sample ADSR: advance each active voice's envelope, multiply the
    // voice's voices_out channel in place by env_value, then re-sum into
    // the stereo `output`. The envelope is applied AFTER the lane-render's
    // pre-envelope summing in voices_out, so voices_out reflects the
    // enveloped per-voice signal that downstream consumers expect.
    const float dt = 1.0f / sr;
    if (voices_buf && stereo_out) {
        // Reset stereo sum — lane-render wrote a pre-envelope sum we must replace.
        std::memset(stereo_out, 0, 2 * frames * sizeof(float));
        for (uint32_t s = 0; s < frames; ++s) {
            float sum = 0.0f;
            for (uint32_t v = 0; v < n_active; ++v) {
                const int slot = slot_for_voice[v];
                vivid::adsr::advance(midi_voices_[slot].env, dt,
                                     attack.value, decay.value,
                                     sustain.value, release.value);
                const float env = midi_voices_[slot].env.env_value;
                voices_buf[v * frames + s] *= env;
                sum += voices_buf[v * frames + s];
            }
            stereo_out[s]            = sum;
            stereo_out[frames + s]   = sum;
        }
        // Zero unused voices_out channels so downstream consumers see clean
        // silence on inactive voice slots.
        if (n_active < static_cast<uint32_t>(kMaxVoices)) {
            std::memset(voices_buf + n_active * frames, 0,
                        (kMaxVoices - n_active) * frames * sizeof(float));
        }
    }

    // Mark voices whose release tail has finished as inactive so the next
    // block does not waste cycles on idle envelopes.
    for (int i = 0; i < kMaxVoices; ++i) {
        if (midi_allocator_.slots[i].active &&
            midi_voices_[i].env.stage == vivid::adsr::IDLE) {
            midi_allocator_.slots[i].active = false;
        }
    }

    // Emit the four voice_* control breakouts in note_id-sorted order so
    // they line up with voices_out channels.
    if (ctx->output_lanes) {
        VividLaneOutput lanes[vivid_sequencers::kVoiceBreakoutLaneCount] = {
            ctx->output_lanes[1], ctx->output_lanes[2],
            ctx->output_lanes[3], ctx->output_lanes[4],
        };
        vivid_sequencers::emit_voice_breakouts(midi_allocator_, lanes);
    }

    midi_frame_counter_ += frames;
}
