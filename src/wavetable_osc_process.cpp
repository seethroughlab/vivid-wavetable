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

    const VividLanePort* freq_lane = ctx->input_lanes ? &ctx->input_lanes[0] : nullptr;
    const VividLanePort* gates_lane = ctx->input_lanes ? &ctx->input_lanes[1] : nullptr;
    const VividLanePort* pitch_lane = ctx->input_lanes ? &ctx->input_lanes[3] : nullptr;
    const VividLanePort* pos_mod_lane = ctx->input_lanes ? &ctx->input_lanes[4] : nullptr;
    const VividLanePort* warp_mod_lane = ctx->input_lanes ? &ctx->input_lanes[5] : nullptr;
    const VividLanePort* lane_id_lane = ctx->input_lanes ? &ctx->input_lanes[6] : nullptr;

    uint32_t voice_count = freq_lane ? freq_lane->length : 0;
    uint32_t max_note_voices = stereo_pairs ? static_cast<uint32_t>(kMaxStereoPairVoices)
                                            : static_cast<uint32_t>(kMaxVoices);
    if (voice_count > max_note_voices) voice_count = max_note_voices;

    float porta_rate = 1.0f;
    if (porta_ms > 0.0f) {
        float porta_samples = porta_ms * 0.001f * sr;
        porta_rate = 1.0f - std::exp(-4.0f / porta_samples);
    }

    float* mod_buf = (interaction > INTERACTION_OFF && interaction_depth_value > 0.0f && ctx->input_buffers[7])
        ? ctx->input_buffers[7] : nullptr;
    uint32_t mod_channels = mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[7] : 0;
    float* pitch_mod_buf = ctx->input_buffers[8];
    uint32_t pitch_mod_ch = pitch_mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[8] : 0;
    float* pos_mod_buf = ctx->input_buffers[9];
    uint32_t pos_mod_ch = pos_mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[9] : 0;
    float* warp_mod_buf = ctx->input_buffers[10];
    uint32_t warp_mod_ch = warp_mod_buf && ctx->input_channel_counts ? ctx->input_channel_counts[10] : 0;

    float* out_buf = ctx->output_buffers[0];
    std::memset(out_buf, 0, kMaxVoices * frames * sizeof(float));

    float unison_norm = 1.0f / std::sqrt(static_cast<float>(num_uni));

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

        for (uint32_t s = 0; s < frames; ++s) {
            if (porta_ms > 0.0f && v.current_freq != v.target_freq) {
                v.current_freq += (v.target_freq - v.current_freq) * porta_rate;
                if (std::abs(v.current_freq - v.target_freq) < 0.01f) {
                    v.current_freq = v.target_freq;
                }
            } else {
                v.current_freq = v.target_freq;
            }

            float pitch_offset = pitch_mod_voice ? pitch_mod_voice[s] : pitch_offset_lane;
            float pos_mod_val = pos_mod_voice ? pos_mod_voice[s] : pos_mod_lane_val;
            float warp_mod_val = warp_mod_voice ? warp_mod_voice[s] : warp_mod_lane_val;

            float pitch_ratio = std::pow(2.0f, pitch_offset / 12.0f);
            float smooth_pos = v.pos_smoother.process(std::clamp(pos + pos_mod_val, 0.0f, 1.0f), pos_smooth_coeff);
            float smooth_warp = v.warp_smoother.process(std::clamp(warp_a + warp_mod_val, 0.0f, 1.0f), warp_smooth_coeff);

            float mono_sum = 0.0f;
            float stereo_l = 0.0f;
            float stereo_r = 0.0f;
            for (int ui = 0; ui < num_uni; ++ui) {
                float detune_cents = det + unison_detune_offset(ui, num_uni, uni_spr, uni_spread_mode, lid);
                float drift_seed = hash01(lid + static_cast<uint32_t>(ui * 379));
                float drift_rate_scale = 0.7f + drift_seed * 0.8f;
                float drift_cents = std::sin(v.drift_phase[ui]) * drift_amt * kMaxDriftCents;
                v.drift_phase[ui] += static_cast<double>(
                    (2.0f * static_cast<float>(M_PI) * drift_rate * drift_rate_scale) / sr);
                if (!std::isfinite(v.drift_phase[ui])) v.drift_phase[ui] = 0.0;
                if (v.drift_phase[ui] > 2.0 * M_PI) v.drift_phase[ui] -= 2.0 * M_PI;

                float base_freq = v.current_freq * pitch_ratio * cents_to_ratio(detune_cents + drift_cents);
                if (!std::isfinite(base_freq) || base_freq <= 0.0f) {
                    base_freq = std::max(v.current_freq, 1.0f);
                }

                InteractionSignal interaction_signal = prepare_interaction_signal(
                    interaction,
                    base_freq,
                    interaction_depth_value,
                    interaction_input_gain_value,
                    interaction_tracking_value,
                    mod_ch ? mod_ch[s] : 0.0f,
                    mod_ch != nullptr,
                    v.interaction_dc);

                float phase_inc = base_freq / sr;
                if (interaction == INTERACTION_FM && mod_ch) {
                    phase_inc += interaction_fm_phase_delta(interaction_signal, sr);
                }

                float interaction_phase = wrap_phase(v.phase[ui]);
                if (interaction == INTERACTION_PM && mod_ch) {
                    interaction_phase = wrap_phase(v.phase[ui] + interaction_pm_offset(interaction_signal));
                }

                float warped = warp_phase(interaction_phase, warp_m, smooth_warp, v.last_sample[ui]);
                float sig = wt.sample(warped, smooth_pos, base_freq, sr);
                v.last_sample[ui] = sig;

                float left_sig = sig;
                float right_sig = sig;
                if (stereo_pairs) {
                    float phase_shift = stereo_pair_phase_shift(ui, num_uni, stereo_phase, lid);
                    if (std::abs(phase_shift) > 0.00001f) {
                        float left_phase = interaction_phase + phase_shift;
                        float right_phase = interaction_phase - phase_shift;
                        float left_warped = warp_phase(wrap_phase(left_phase), warp_m, smooth_warp, v.last_sample[ui]);
                        float right_warped = warp_phase(wrap_phase(right_phase), warp_m, smooth_warp, v.last_sample[ui]);
                        left_sig = wt.sample(left_warped, smooth_pos, base_freq, sr);
                        right_sig = wt.sample(right_warped, smooth_pos, base_freq, sr);
                    }
                }

                if (mod_ch && interaction > INTERACTION_OFF) {
                    if (interaction == INTERACTION_RM) {
                        sig = interaction_rm_sample(sig, interaction_signal);
                        left_sig = interaction_rm_sample(left_sig, interaction_signal);
                        right_sig = interaction_rm_sample(right_sig, interaction_signal);
                    } else if (interaction == INTERACTION_AM) {
                        float gain = interaction_am_gain(interaction_signal);
                        sig *= gain;
                        left_sig *= gain;
                        right_sig *= gain;
                    }
                }

                if (mod_ch && interaction > INTERACTION_OFF) {
                    float comp = interaction_output_compensation(interaction, interaction_signal.amount);
                    sig *= comp;
                    left_sig *= comp;
                    right_sig *= comp;
                }

                sig *= amp * unison_norm;
                left_sig *= amp * unison_norm;
                right_sig *= amp * unison_norm;
                if (stereo_pairs) {
                    float pan = std::clamp(unison_pan_position(ui, num_uni, uni_stereo), -1.0f, 1.0f);
                    float theta = (pan + 1.0f) * static_cast<float>(M_PI) * 0.25f;
                    stereo_l += left_sig * std::cos(theta);
                    stereo_r += right_sig * std::sin(theta);
                } else {
                    mono_sum += sig;
                }

                v.phase[ui] += static_cast<double>(phase_inc);
                if (!std::isfinite(v.phase[ui])) {
                    v.phase[ui] = 0.0;
                } else {
                    v.phase[ui] -= std::floor(v.phase[ui]);
                }
            }

            float gate_gain = 1.0f;
            if (v.declick_remaining > 0) {
                gate_gain = static_cast<float>(kDeClickSamples - v.declick_remaining + 1)
                    / static_cast<float>(kDeClickSamples);
                v.declick_remaining--;
            }

            if (stereo_pairs) {
                ch_out_l[s] = stereo_l * gate_gain;
                ch_out_r[s] = stereo_r * gate_gain;
            } else {
                ch_out_l[s] = mono_sum * gate_gain;
            }
        }
    }
}
