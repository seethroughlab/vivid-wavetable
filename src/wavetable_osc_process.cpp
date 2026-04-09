#include "wavetable_osc_internal.h"

#include "lane_audio_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace vivid_wavetable::dsp;

void WavetableOsc::process_audio(const VividAudioContext* ctx) {
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

    const VividLaneView* freq_lane = ctx->input_lanes ? &ctx->input_lanes[0] : nullptr;
    const VividLaneView* gates_lane = ctx->input_lanes ? &ctx->input_lanes[1] : nullptr;
    const VividLaneView* pitch_lane = ctx->input_lanes ? &ctx->input_lanes[3] : nullptr;
    const VividLaneView* pos_mod_lane = ctx->input_lanes ? &ctx->input_lanes[4] : nullptr;
    const VividLaneView* warp_mod_lane = ctx->input_lanes ? &ctx->input_lanes[5] : nullptr;
    const VividLaneView* lane_id_lane = ctx->input_lanes ? &ctx->input_lanes[6] : nullptr;

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
        (interaction > INTERACTION_OFF && interaction_depth_value > 1.0e-6f && ctx->input_buffers[7]);
    float* mod_buf = interaction_enabled ? ctx->input_buffers[7] : nullptr;
    uint32_t mod_channels = mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[7] : 0;
    float* pitch_mod_buf = ctx->input_buffers[8];
    uint32_t pitch_mod_ch = pitch_mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[8] : 0;
    float* pos_mod_buf = ctx->input_buffers[9];
    uint32_t pos_mod_ch = pos_mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[9] : 0;
    float* warp_mod_buf = ctx->input_buffers[10];
    uint32_t warp_mod_ch = warp_mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[10] : 0;

    float* out_buf = ctx->output_buffers[0];
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
    }
}
