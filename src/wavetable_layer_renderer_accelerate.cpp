#ifdef VIVID_HAS_ACCELERATE

#include "wavetable_layer_renderer.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>

namespace vivid_wavetable::layer {
namespace {

static inline float wrap_phase_scalar(float phase) {
    if (phase >= 1.0f) {
        phase -= 1.0f;
        if (phase >= 1.0f) phase -= std::floor(phase);
    } else if (phase < 0.0f) {
        phase += 1.0f;
        if (phase < 0.0f) phase -= std::floor(phase);
    }
    return phase;
}

static bool can_render_accelerate(const RenderUnit& ru,
                                  const VoiceBlock& vb,
                                  const PreparedWavetable& pwt,
                                  const RenderParams& params) {
    if (ru.active_count == 0 || pwt.frame_count == 0) return false;
    if (params.warp_mode != vivid_wavetable::dsp::WARP_NONE) return false;
    if (params.drift_enabled) return false;

    for (int slot = 0; slot < ru.active_count; ++slot) {
        const int vi = ru.voice_idx[slot];
        if (vi < 0 || vi >= vb.voice_count || vi >= kMaxVoices) return false;
        if (std::abs(vb.pitch_lane_base[vi]) > 1.0e-6f) return false;
        if (vb.pitch_mod_audio[vi]) return false;
    }
    return true;
}

static void update_subblock_targets(uint32_t sb,
                                    const RenderParams& params,
                                    VoiceBlock& vb) {
    for (int v = 0; v < vb.voice_count; ++v) {
        float pos_target = params.position_base + vb.position_lane_base[v];
        if (vb.position_mod_audio[v]) pos_target += vb.position_mod_audio[v][sb];
        vb.pos_from[v] = vb.pos_to[v];
        pos_target = std::clamp(pos_target, 0.0f, 1.0f);
        vb.pos_to[v] = vb.pos_smoother[v]
            ? vb.pos_smoother[v]->process(pos_target, params.pos_smooth_coeff)
            : pos_target;

        float warp_target = params.warp_base + vb.warp_lane_base[v];
        if (vb.warp_mod_audio[v]) warp_target += vb.warp_mod_audio[v][sb];
        vb.warp_from[v] = vb.warp_to[v];
        warp_target = std::clamp(warp_target, 0.0f, 1.0f);
        vb.warp_to[v] = vb.warp_smoother[v]
            ? vb.warp_smoother[v]->process(warp_target, params.warp_smooth_coeff)
            : warp_target;
    }
}

static inline float lookup_guarded_scalar(const PreparedWavetable& pwt,
                                          int level,
                                          uint32_t f0,
                                          uint32_t f1,
                                          float frame_blend,
                                          float phase) {
    const float sp = phase * static_cast<float>(bank::kSamplesPerFrame);
    const int i = std::clamp(static_cast<int>(sp), 0, static_cast<int>(bank::kSamplesPerFrame) - 1);
    const float frac = sp - static_cast<float>(i);
    const float* d0 = pwt.frame_data(level, f0);
    const float s0 = d0[i] + (d0[i + 1] - d0[i]) * frac;
    if (f0 == f1) return s0;
    const float* d1 = pwt.frame_data(level, f1);
    const float s1 = d1[i] + (d1[i + 1] - d1[i]) * frac;
    return s0 + (s1 - s0) * frame_blend;
}

static void sample_same_frames_vdsp(const PreparedWavetable& pwt,
                                    int mip,
                                    uint32_t f0,
                                    uint32_t f1,
                                    const float* phase,
                                    const float* frame_blend,
                                    float frame_blend_const,
                                    bool use_frame_blend_vector,
                                    float* out,
                                    uint32_t n) {
    float index[kControlSubBlock] = {};
    float f0_samples[kControlSubBlock] = {};
    const float table_size = static_cast<float>(bank::kSamplesPerFrame);
    vDSP_vsmul(phase, 1, &table_size, index, 1, n);
    vDSP_vlint(pwt.frame_data(mip, f0), index, 1, f0_samples, 1, n, PreparedWavetable::kGuardedFrameSize);

    if (f0 == f1) {
        std::copy(f0_samples, f0_samples + n, out);
        return;
    }

    float f1_samples[kControlSubBlock] = {};
    float diff[kControlSubBlock] = {};
    vDSP_vlint(pwt.frame_data(mip, f1), index, 1, f1_samples, 1, n, PreparedWavetable::kGuardedFrameSize);
    vDSP_vsub(f0_samples, 1, f1_samples, 1, diff, 1, n);
    if (use_frame_blend_vector) {
        float blended[kControlSubBlock] = {};
        vDSP_vmul(diff, 1, frame_blend, 1, blended, 1, n);
        vDSP_vadd(f0_samples, 1, blended, 1, out, 1, n);
    } else {
        vDSP_vsma(diff, 1, &frame_blend_const, f0_samples, 1, out, 1, n);
    }
}

} // namespace

bool render_block_accelerate(
    float* stereo_out,
    uint32_t frames,
    float sample_rate,
    RenderUnit& ru,
    VoiceBlock& vb,
    const PreparedWavetable& pwt,
    const RenderParams& params
) {
    (void)sample_rate;
    if (!can_render_accelerate(ru, vb, pwt, params)) return false;

    float* out_l = stereo_out;
    float* out_r = stereo_out + frames;
    const float frame_count_m1 = static_cast<float>(std::max(pwt.frame_count, 1u) - 1);

    for (uint32_t sb = 0; sb < frames; sb += kControlSubBlock) {
        const uint32_t block_len = std::min(static_cast<uint32_t>(kControlSubBlock), frames - sb);
        const float inv_block = 1.0f / static_cast<float>(block_len);
        update_subblock_targets(sb, params, vb);

        for (int slot = 0; slot < ru.active_count; ++slot) {
            const int vi = ru.voice_idx[slot];
            const int mip = ru.mip_level[slot];
            float phase_start = ru.phase[slot];
            const float phase_inc = ru.phase_inc[slot];
            const float gain = ru.gain[slot];
            const float pan_l = ru.pan_l[slot];
            const float pan_r = ru.pan_r[slot];
            const int declick_start = vb.declick_remaining[vi];
            const bool has_declick = declick_start > 0;
            const bool has_voice_gain = vb.voice_gain_audio[vi] != nullptr;
            const float pos_delta = vb.pos_to[vi] - vb.pos_from[vi];
            const bool stable_position = std::abs(pos_delta) <= 1.0e-7f;

            uint32_t offset = 0;
            for (; offset < block_len; offset += kControlSubBlock) {
                const uint32_t n = std::min(static_cast<uint32_t>(kControlSubBlock), block_len - offset);
                const uint32_t s = sb + offset;

                float phase[kControlSubBlock] = {};
                vDSP_vramp(&phase_start, &phase_inc, phase, 1, n);
                for (uint32_t i = 0; i < n; ++i) {
                    phase[i] = wrap_phase_scalar(phase[i]);
                }
                phase_start = wrap_phase_scalar(phase_start + phase_inc * static_cast<float>(n));

                float samples[kControlSubBlock] = {};
                if (stable_position) {
                    const float frame_pos = std::clamp(vb.pos_from[vi], 0.0f, 1.0f) * frame_count_m1;
                    const uint32_t f0 = std::min(static_cast<uint32_t>(frame_pos), pwt.frame_count - 1);
                    const uint32_t f1 = std::min(f0 + 1, pwt.frame_count - 1);
                    const float frame_blend = frame_pos - static_cast<float>(f0);
                    sample_same_frames_vdsp(pwt, mip, f0, f1, phase, nullptr, frame_blend, false, samples, n);
                } else {
                    float smooth_pos[kControlSubBlock] = {};
                    float frame_blend[kControlSubBlock] = {};
                    const float pos_start = vb.pos_from[vi]
                        + pos_delta * (static_cast<float>(offset + 1) * inv_block);
                    const float pos_inc = pos_delta * inv_block;
                    vDSP_vramp(&pos_start, &pos_inc, smooth_pos, 1, n);

                    const float frame_first = std::clamp(smooth_pos[0], 0.0f, 1.0f) * frame_count_m1;
                    const float frame_last = std::clamp(smooth_pos[n - 1], 0.0f, 1.0f) * frame_count_m1;
                    const uint32_t f0_first = std::min(static_cast<uint32_t>(frame_first), pwt.frame_count - 1);
                    const uint32_t f0_last = std::min(static_cast<uint32_t>(frame_last), pwt.frame_count - 1);
                    const uint32_t f1_first = std::min(f0_first + 1, pwt.frame_count - 1);
                    const uint32_t f1_last = std::min(f0_last + 1, pwt.frame_count - 1);

                    if (f0_first == f0_last && f1_first == f1_last) {
                        for (uint32_t i = 0; i < n; ++i) {
                            const float frame_pos = std::clamp(smooth_pos[i], 0.0f, 1.0f) * frame_count_m1;
                            frame_blend[i] = frame_pos - static_cast<float>(f0_first);
                        }
                        sample_same_frames_vdsp(
                            pwt, mip, f0_first, f1_first, phase, frame_blend, 0.0f, true, samples, n);
                    } else {
                        for (uint32_t i = 0; i < n; ++i) {
                            const float frame_pos = std::clamp(smooth_pos[i], 0.0f, 1.0f) * frame_count_m1;
                            const uint32_t f0 = std::min(static_cast<uint32_t>(frame_pos), pwt.frame_count - 1);
                            const uint32_t f1 = std::min(f0 + 1, pwt.frame_count - 1);
                            const float blend = frame_pos - static_cast<float>(f0);
                            samples[i] = lookup_guarded_scalar(pwt, mip, f0, f1, blend, phase[i]);
                        }
                    }
                }

                if (has_voice_gain) {
                    vDSP_vmul(samples, 1, vb.voice_gain_audio[vi] + s, 1, samples, 1, n);
                }
                if (has_declick) {
                    float declick[kControlSubBlock] = {};
                    for (uint32_t i = 0; i < n; ++i) {
                        const int rem = declick_start - static_cast<int>(offset + i);
                        declick[i] = rem > 0
                            ? static_cast<float>(kDeClickSamples - rem + 1) / static_cast<float>(kDeClickSamples)
                            : 1.0f;
                    }
                    vDSP_vmul(samples, 1, declick, 1, samples, 1, n);
                }

                const float gain_l = gain * pan_l;
                const float gain_r = gain * pan_r;
                float scaled[kControlSubBlock] = {};
                vDSP_vsmul(samples, 1, &gain_l, scaled, 1, n);
                vDSP_vadd(out_l + s, 1, scaled, 1, out_l + s, 1, n);
                vDSP_vsmul(samples, 1, &gain_r, scaled, 1, n);
                vDSP_vadd(out_r + s, 1, scaled, 1, out_r + s, 1, n);
            }

            ru.phase[slot] = phase_start;
        }

        for (int v = 0; v < vb.voice_count; ++v) {
            vb.declick_remaining[v] = std::max(0, vb.declick_remaining[v] - static_cast<int>(block_len));
        }
    }

    return true;
}

} // namespace vivid_wavetable::layer

#endif // VIVID_HAS_ACCELERATE
