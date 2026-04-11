// WavetableLayer SIMD render backend using Google Highway.
// Processes render units (voice × unison) in fixed-width SIMD batches.

#ifdef VIVID_HAS_HIGHWAY

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "wavetable_layer_renderer_simd.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

#include "wavetable_layer_renderer.h"
#include "wavetable_dsp.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

HWY_BEFORE_NAMESPACE();
namespace vivid_wavetable::layer::HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

using D = hn::ScalableTag<float>;
using DI = hn::RebindToSigned<D>;

using vivid_wavetable::dsp::MotionSmoother;

namespace {

static HWY_INLINE hn::Vec<D> warp_none(D, hn::Vec<D> phase, hn::Vec<D>) {
    return phase;
}

static HWY_INLINE hn::Vec<D> warp_sync(D d, hn::Vec<D> phase, hn::Vec<D> amt) {
    auto r = hn::MulAdd(amt, hn::Set(d, 7.0f), hn::Set(d, 1.0f));
    auto sp = hn::Mul(phase, r);
    return hn::Sub(sp, hn::Floor(sp));
}

static HWY_INLINE hn::Vec<D> warp_flip(D d, hn::Vec<D> phase, hn::Vec<D> amt) {
    auto half = hn::Set(d, 0.5f);
    auto one = hn::Set(d, 1.0f);
    auto flipped = hn::Sub(one, phase);
    auto blended = hn::MulAdd(hn::Sub(flipped, phase), amt, phase);
    auto mask = hn::Ge(phase, half);
    return hn::IfThenElse(mask, blended, phase);
}

static HWY_INLINE hn::Vec<D> warp_quantize(D d, hn::Vec<D> phase, hn::Vec<D> amt) {
    auto steps_f = hn::Max(hn::Set(d, 4.0f),
                           hn::NegMulAdd(amt, hn::Set(d, 252.0f), hn::Set(d, 256.0f)));
    auto inv_steps = hn::Div(hn::Set(d, 1.0f), steps_f);
    return hn::Mul(hn::Floor(hn::Mul(phase, steps_f)), inv_steps);
}

static HWY_INLINE hn::Vec<D> warp_asym(D d, hn::Vec<D> phase, hn::Vec<D> amt) {
    auto half = hn::Set(d, 0.5f);
    auto stretch = hn::MulAdd(amt, hn::Set(d, 0.3f), half);
    auto one = hn::Set(d, 1.0f);
    auto two = hn::Set(d, 2.0f);
    auto lo = hn::Mul(hn::Mul(phase, two), stretch);
    auto hi = hn::MulAdd(hn::Mul(hn::Sub(phase, half), two), hn::Sub(one, stretch), stretch);
    auto mask = hn::Lt(phase, half);
    return hn::IfThenElse(mask, lo, hi);
}

static HWY_INLINE hn::Vec<D> warp_mirror(D d, hn::Vec<D> phase, hn::Vec<D> amt) {
    auto mid = hn::NegMulAdd(amt, hn::Set(d, 0.3f), hn::Set(d, 0.5f));
    auto mirrored = hn::Sub(hn::Add(mid, mid), phase);
    auto half = hn::Set(d, 0.5f);
    auto scaled = hn::Mul(hn::Div(phase, hn::Max(mid, hn::Set(d, 1e-6f))), half);
    auto mask = hn::Gt(phase, mid);
    return hn::IfThenElse(mask, mirrored, scaled);
}

static HWY_INLINE hn::Vec<D> warp_bend_plus(D d, hn::Vec<D> phase, hn::Vec<D> amt) {
    const size_t lanes = hn::Lanes(d);
    HWY_ALIGN float p[hn::MaxLanes(d)];
    HWY_ALIGN float a[hn::MaxLanes(d)];
    HWY_ALIGN float out[hn::MaxLanes(d)];
    hn::Store(phase, d, p);
    hn::Store(amt, d, a);
    for (size_t i = 0; i < lanes; ++i) {
        out[i] = std::pow(p[i], 1.0f + a[i] * 3.0f);
    }
    return hn::Load(d, out);
}

static HWY_INLINE hn::Vec<D> warp_bend_minus(D d, hn::Vec<D> phase, hn::Vec<D> amt) {
    const size_t lanes = hn::Lanes(d);
    HWY_ALIGN float p[hn::MaxLanes(d)];
    HWY_ALIGN float a[hn::MaxLanes(d)];
    HWY_ALIGN float out[hn::MaxLanes(d)];
    hn::Store(phase, d, p);
    hn::Store(amt, d, a);
    for (size_t i = 0; i < lanes; ++i) {
        out[i] = std::pow(p[i], 1.0f / (1.0f + a[i] * 3.0f));
    }
    return hn::Load(d, out);
}

using WarpFn = hn::Vec<D> (*)(D, hn::Vec<D>, hn::Vec<D>);

static WarpFn select_warp_fn(int mode) {
    switch (mode) {
        case dsp::WARP_SYNC:       return warp_sync;
        case dsp::WARP_BEND_PLUS:  return warp_bend_plus;
        case dsp::WARP_BEND_MINUS: return warp_bend_minus;
        case dsp::WARP_MIRROR:     return warp_mirror;
        case dsp::WARP_ASYM:       return warp_asym;
        case dsp::WARP_QUANTIZE:   return warp_quantize;
        case dsp::WARP_FLIP:       return warp_flip;
        default:                   return warp_none;
    }
}

static void update_subblock_targets(
    uint32_t sb,
    const RenderParams& params,
    VoiceBlock& vb
) {
    for (int v = 0; v < vb.voice_count; ++v) {
        float pos_target = params.position_base + vb.position_lane_base[v];
        float warp_target = params.warp_base + vb.warp_lane_base[v];
        if (vb.position_mod_audio[v]) pos_target += vb.position_mod_audio[v][sb];
        if (vb.warp_mod_audio[v]) warp_target += vb.warp_mod_audio[v][sb];

        vb.pos_from[v] = vb.pos_to[v];
        pos_target = std::clamp(pos_target, 0.0f, 1.0f);
        if (vb.pos_smoother[v]) {
            vb.pos_to[v] = vb.pos_smoother[v]->process(pos_target, params.pos_smooth_coeff);
        } else {
            vb.pos_to[v] = pos_target;
        }

        vb.warp_from[v] = vb.warp_to[v];
        warp_target = std::clamp(warp_target, 0.0f, 1.0f);
        if (vb.warp_smoother[v]) {
            vb.warp_to[v] = vb.warp_smoother[v]->process(warp_target, params.warp_smooth_coeff);
        } else {
            vb.warp_to[v] = warp_target;
        }
    }
}

template <bool HasWarp, bool HasDrift>
void RenderBlockSimdImpl(
    float* stereo_out,
    uint32_t frames,
    float sample_rate,
    RenderUnit& ru,
    VoiceBlock& vb,
    const PreparedWavetable& pwt,
    const RenderParams& params,
    WarpFn warp_fn
) {
    (void)sample_rate;
    if (ru.active_count == 0 || pwt.frame_count == 0) return;

    const D d;
    const DI di;
    const size_t lanes = hn::Lanes(d);

    float* out_l = stereo_out;
    float* out_r = stereo_out + frames;

    const float drift_scale = params.drift_amount * 8.0f;
    const float frame_count_m1 = static_cast<float>(std::max(pwt.frame_count, 1u) - 1);
    const float* flat_base = pwt.base();

    HWY_ALIGN int32_t level_offsets[bank::kNumMipLevels];
    for (int l = 0; l < bank::kNumMipLevels; ++l) {
        level_offsets[l] = static_cast<int32_t>(pwt.level_offset[l]);
    }

    const auto v_frame_count_m1 = hn::Set(d, frame_count_m1);
    const auto v_table_size = hn::Set(d, static_cast<float>(bank::kSamplesPerFrame));
    const auto v_one = hn::Set(d, 1.0f);
    const auto v_zero = hn::Set(d, 0.0f);
    const auto v_max_frame = hn::Set(di, static_cast<int32_t>(pwt.frame_count - 1));
    const auto v_max_idx = hn::Set(di, static_cast<int32_t>(bank::kSamplesPerFrame - 1));
    const auto v_one_i = hn::Set(di, 1);
    const auto v_two_pi = hn::Set(d, 2.0f * static_cast<float>(M_PI));

    const int padded_count = static_cast<int>((ru.active_count + lanes - 1) / lanes * lanes);

    for (uint32_t sb = 0; sb < frames; sb += kControlSubBlock) {
        const uint32_t block_len = std::min(static_cast<uint32_t>(kControlSubBlock), frames - sb);
        const float inv_block = 1.0f / static_cast<float>(block_len);

        update_subblock_targets(sb, params, vb);

        for (uint32_t offset = 0; offset < block_len; ++offset) {
            const uint32_t s = sb + offset;
            const auto v_t = hn::Set(d, static_cast<float>(offset + 1) * inv_block);

            float sum_l = 0.0f;
            float sum_r = 0.0f;

            for (int slot = 0; slot < padded_count; slot += static_cast<int>(lanes)) {
                auto v_phase = hn::Load(d, &ru.phase[slot]);
                auto v_phase_inc = hn::Load(d, &ru.phase_inc[slot]);
                auto v_pan_l = hn::Load(d, &ru.pan_l[slot]);
                auto v_pan_r = hn::Load(d, &ru.pan_r[slot]);
                auto v_gain = hn::Load(d, &ru.gain[slot]);
                auto v_mip = hn::Load(di, &ru.mip_level[slot]);
                auto v_voice_idx = hn::Load(di, &ru.voice_idx[slot]);

                HWY_ALIGN float pos_from_arr[hn::MaxLanes(d)];
                HWY_ALIGN float pos_to_arr[hn::MaxLanes(d)];
                HWY_ALIGN float warp_from_arr[hn::MaxLanes(d)];
                HWY_ALIGN float warp_to_arr[hn::MaxLanes(d)];
                HWY_ALIGN float voice_gain_arr[hn::MaxLanes(d)];
                HWY_ALIGN float declick_arr[hn::MaxLanes(d)];
                HWY_ALIGN float pitch_mod_arr[hn::MaxLanes(d)];
                HWY_ALIGN int32_t vi_arr[hn::MaxLanes(DI())];

                hn::Store(v_voice_idx, di, vi_arr);

                for (size_t i = 0; i < lanes; ++i) {
                    const int vi = vi_arr[i];
                    pos_from_arr[i] = vb.pos_from[vi];
                    pos_to_arr[i] = vb.pos_to[vi];
                    warp_from_arr[i] = vb.warp_from[vi];
                    warp_to_arr[i] = vb.warp_to[vi];
                    voice_gain_arr[i] = vb.voice_gain_audio[vi] ? vb.voice_gain_audio[vi][s] : 1.0f;
                    pitch_mod_arr[i] = vb.pitch_lane_base[vi]
                                     + (vb.pitch_mod_audio[vi] ? vb.pitch_mod_audio[vi][s] : 0.0f);
                    declick_arr[i] = (vb.declick_remaining[vi] > 0)
                        ? static_cast<float>(kDeClickSamples - vb.declick_remaining[vi] + 1)
                          / static_cast<float>(kDeClickSamples)
                        : 1.0f;
                }

                auto v_pos_from = hn::Load(d, pos_from_arr);
                auto v_pos_to = hn::Load(d, pos_to_arr);
                auto v_warp_from = hn::Load(d, warp_from_arr);
                auto v_warp_to = hn::Load(d, warp_to_arr);
                auto v_voice_gain = hn::Load(d, voice_gain_arr);
                auto v_declick = hn::Load(d, declick_arr);
                auto v_pitch_mod = hn::Load(d, pitch_mod_arr);

                {
                    HWY_ALIGN float pi_arr[hn::MaxLanes(d)];
                    HWY_ALIGN float pm_arr[hn::MaxLanes(d)];
                    hn::Store(v_phase_inc, d, pi_arr);
                    hn::Store(v_pitch_mod, d, pm_arr);
                    for (size_t i = 0; i < lanes; ++i) {
                        if (std::abs(pm_arr[i]) > 1e-6f) {
                            pi_arr[i] *= std::pow(2.0f, pm_arr[i] / 12.0f);
                        }
                    }
                    v_phase_inc = hn::Load(d, pi_arr);
                }

                if constexpr (HasDrift) {
                    HWY_ALIGN float dp_arr[hn::MaxLanes(d)];
                    HWY_ALIGN float pi_arr[hn::MaxLanes(d)];
                    auto v_drift_phase = hn::Load(d, &ru.drift_phase[slot]);
                    auto v_drift_inc = hn::Load(d, &ru.drift_phase_inc[slot]);
                    hn::Store(v_drift_phase, d, dp_arr);
                    hn::Store(v_phase_inc, d, pi_arr);
                    for (size_t i = 0; i < lanes; ++i) {
                        const float drift_cents = std::sin(dp_arr[i]) * drift_scale;
                        pi_arr[i] *= dsp::cents_to_ratio(drift_cents);
                    }
                    v_phase_inc = hn::Load(d, pi_arr);
                    v_drift_phase = hn::Add(v_drift_phase, v_drift_inc);
                    auto wrap_mask = hn::Ge(v_drift_phase, v_two_pi);
                    v_drift_phase = hn::Sub(v_drift_phase, hn::IfThenElseZero(wrap_mask, v_two_pi));
                    hn::Store(v_drift_phase, d, &ru.drift_phase[slot]);
                }

                const auto v_smooth_pos = hn::MulAdd(hn::Sub(v_pos_to, v_pos_from), v_t, v_pos_from);
                const auto v_smooth_warp = hn::MulAdd(hn::Sub(v_warp_to, v_warp_from), v_t, v_warp_from);

                const auto v_frame_pos = hn::Mul(v_smooth_pos, v_frame_count_m1);
                const auto v_f0 = hn::Min(hn::ConvertTo(di, hn::Floor(v_frame_pos)), v_max_frame);
                const auto v_f1 = hn::Min(hn::Add(v_f0, v_one_i), v_max_frame);
                const auto v_frame_blend = hn::Sub(v_frame_pos, hn::ConvertTo(d, v_f0));

                v_phase = hn::Sub(v_phase, hn::Floor(v_phase));

                auto v_lookup_phase = v_phase;
                if constexpr (HasWarp) {
                    v_lookup_phase = warp_fn(d, v_phase, v_smooth_warp);
                    v_lookup_phase = hn::Clamp(v_lookup_phase, v_zero, hn::Set(d, 0.999999f));
                }

                const auto v_sp = hn::Mul(v_lookup_phase, v_table_size);
                const auto v_i = hn::Min(hn::ConvertTo(di, hn::Floor(v_sp)), v_max_idx);
                const auto v_frac = hn::Sub(v_sp, hn::ConvertTo(d, v_i));

                HWY_ALIGN int32_t mip_arr[hn::MaxLanes(DI())];
                HWY_ALIGN int32_t f0_arr[hn::MaxLanes(DI())];
                HWY_ALIGN int32_t f1_arr[hn::MaxLanes(DI())];
                HWY_ALIGN int32_t i_arr[hn::MaxLanes(DI())];
                HWY_ALIGN int32_t off0_arr[hn::MaxLanes(DI())];
                HWY_ALIGN int32_t off1_arr[hn::MaxLanes(DI())];

                hn::Store(v_mip, di, mip_arr);
                hn::Store(v_f0, di, f0_arr);
                hn::Store(v_f1, di, f1_arr);
                hn::Store(v_i, di, i_arr);

                for (size_t k = 0; k < lanes; ++k) {
                    const int32_t level_offset = level_offsets[mip_arr[k]];
                    off0_arr[k] = level_offset
                                + f0_arr[k] * static_cast<int32_t>(PreparedWavetable::kGuardedFrameSize)
                                + i_arr[k];
                    off1_arr[k] = level_offset
                                + f1_arr[k] * static_cast<int32_t>(PreparedWavetable::kGuardedFrameSize)
                                + i_arr[k];
                }

                const auto v_off0 = hn::Load(di, off0_arr);
                const auto v_off1 = hn::Load(di, off1_arr);
                const auto v_s0_lo = hn::GatherIndex(d, flat_base, v_off0);
                const auto v_s0_hi = hn::GatherIndex(d, flat_base, hn::Add(v_off0, v_one_i));
                const auto v_sample_f0 = hn::MulAdd(hn::Sub(v_s0_hi, v_s0_lo), v_frac, v_s0_lo);
                const auto v_s1_lo = hn::GatherIndex(d, flat_base, v_off1);
                const auto v_s1_hi = hn::GatherIndex(d, flat_base, hn::Add(v_off1, v_one_i));
                const auto v_sample_f1 = hn::MulAdd(hn::Sub(v_s1_hi, v_s1_lo), v_frac, v_s1_lo);
                const auto v_sample = hn::MulAdd(hn::Sub(v_sample_f1, v_sample_f0), v_frame_blend, v_sample_f0);

                const auto v_scaled = hn::Mul(hn::Mul(hn::Mul(v_sample, v_gain), v_voice_gain), v_declick);
                sum_l += hn::ReduceSum(d, hn::Mul(v_scaled, v_pan_l));
                sum_r += hn::ReduceSum(d, hn::Mul(v_scaled, v_pan_r));

                v_phase = hn::Add(v_phase, v_phase_inc);
                v_phase = hn::Sub(v_phase, hn::IfThenElseZero(hn::Ge(v_phase, v_one), v_one));
                v_phase = hn::Add(v_phase, hn::IfThenElseZero(hn::Lt(v_phase, v_zero), v_one));
                hn::Store(v_phase, d, &ru.phase[slot]);
            }

            out_l[s] += sum_l;
            out_r[s] += sum_r;

            for (int v = 0; v < vb.voice_count; ++v) {
                if (vb.declick_remaining[v] > 0) --vb.declick_remaining[v];
            }
        }
    }
}

static HWY_INLINE float wrap_phase_scalar(float phase) {
    if (phase >= 1.0f) {
        phase -= 1.0f;
        if (phase >= 1.0f) phase -= std::floor(phase);
    } else if (phase < 0.0f) {
        phase += 1.0f;
        if (phase < 0.0f) phase -= std::floor(phase);
    }
    return phase;
}

static HWY_INLINE float lookup_guarded_scalar(const PreparedWavetable& pwt,
                                              int level,
                                              uint32_t f0,
                                              uint32_t f1,
                                              float frame_blend,
                                              float phase) {
    float sp = phase * static_cast<float>(bank::kSamplesPerFrame);
    int i = std::clamp(static_cast<int>(sp), 0, static_cast<int>(bank::kSamplesPerFrame) - 1);
    float frac = sp - static_cast<float>(i);
    const float* d0 = pwt.frame_data(level, f0);
    float s0 = d0[i] + (d0[i + 1] - d0[i]) * frac;
    if (f0 == f1) return s0;
    const float* d1 = pwt.frame_data(level, f1);
    float s1 = d1[i] + (d1[i + 1] - d1[i]) * frac;
    return s0 + (s1 - s0) * frame_blend;
}

template <bool HasWarp, bool HasDrift>
void RenderBlockSimdSampleAxisImpl(
    float* stereo_out,
    uint32_t frames,
    float sample_rate,
    RenderUnit& ru,
    VoiceBlock& vb,
    const PreparedWavetable& pwt,
    const RenderParams& params,
    WarpFn warp_fn
) {
    (void)sample_rate;
    if (ru.active_count == 0 || pwt.frame_count == 0) return;

    const D d;
    const DI di;
    const size_t lanes = hn::Lanes(d);
    if (lanes == 0) return;

    float* out_l = stereo_out;
    float* out_r = stereo_out + frames;
    const float* flat_base = pwt.base();
    const float drift_scale = params.drift_amount * 8.0f;
    const float frame_count_m1 = static_cast<float>(std::max(pwt.frame_count, 1u) - 1);

    HWY_ALIGN int32_t level_offsets[bank::kNumMipLevels];
    for (int l = 0; l < bank::kNumMipLevels; ++l) {
        level_offsets[l] = static_cast<int32_t>(pwt.level_offset[l]);
    }

    const auto v_table_size = hn::Set(d, static_cast<float>(bank::kSamplesPerFrame));
    const auto v_frame_count_m1 = hn::Set(d, frame_count_m1);
    const auto v_one = hn::Set(d, 1.0f);
    const auto v_zero = hn::Set(d, 0.0f);
    const auto v_max_frame = hn::Set(di, static_cast<int32_t>(pwt.frame_count - 1));
    const auto v_max_idx = hn::Set(di, static_cast<int32_t>(bank::kSamplesPerFrame - 1));
    const auto v_one_i = hn::Set(di, 1);
    const auto v_two_pi = hn::Set(d, 2.0f * static_cast<float>(M_PI));

    for (uint32_t sb = 0; sb < frames; sb += kControlSubBlock) {
        const uint32_t block_len = std::min(static_cast<uint32_t>(kControlSubBlock), frames - sb);
        const float inv_block = 1.0f / static_cast<float>(block_len);

        update_subblock_targets(sb, params, vb);

        for (int slot = 0; slot < ru.active_count; ++slot) {
            const int vi = ru.voice_idx[slot];
            const float pan_l = ru.pan_l[slot];
            const float pan_r = ru.pan_r[slot];
            const float gain = ru.gain[slot];
            const int mip = ru.mip_level[slot];
            const int32_t level_offset = level_offsets[mip];
            float phase = ru.phase[slot];
            float drift_phase = ru.drift_phase[slot];
            const float drift_inc = ru.drift_phase_inc[slot];
            const float base_phase_inc = ru.phase_inc[slot];
            const int declick_start = vb.declick_remaining[vi];
            const bool has_pitch_mod = std::abs(vb.pitch_lane_base[vi]) > 1.0e-6f || vb.pitch_mod_audio[vi];
            const bool has_voice_gain = vb.voice_gain_audio[vi] != nullptr;
            const float pos_delta = vb.pos_to[vi] - vb.pos_from[vi];
            const bool stable_position = std::abs(pos_delta) <= 1.0e-7f;
            const bool has_declick = declick_start > 0;
            const float frame_pos_const = std::clamp(vb.pos_from[vi], 0.0f, 1.0f) * frame_count_m1;
            const uint32_t f0_const = std::min(static_cast<uint32_t>(frame_pos_const), pwt.frame_count - 1);
            const uint32_t f1_const = std::min(f0_const + 1, pwt.frame_count - 1);
            const float frame_blend_const = frame_pos_const - static_cast<float>(f0_const);
            const int32_t off0_const_base = level_offset
                + static_cast<int32_t>(f0_const) * static_cast<int32_t>(PreparedWavetable::kGuardedFrameSize);
            const int32_t off1_const_base = level_offset
                + static_cast<int32_t>(f1_const) * static_cast<int32_t>(PreparedWavetable::kGuardedFrameSize);

            uint32_t offset = 0;
            for (; offset + lanes <= block_len; offset += static_cast<uint32_t>(lanes)) {
                const uint32_t s = sb + offset;
                const auto v_lane = hn::Iota(d, 0.0f);
                auto v_sample_offset = v_lane;
                if constexpr (HasWarp) {
                    v_sample_offset = hn::Add(v_lane, hn::Set(d, static_cast<float>(offset)));
                } else if (!stable_position || has_declick) {
                    v_sample_offset = hn::Add(v_lane, hn::Set(d, static_cast<float>(offset)));
                }
                auto v_t = v_zero;
                if constexpr (HasWarp) {
                    v_t = hn::Mul(hn::Add(v_sample_offset, hn::Set(d, 1.0f)), hn::Set(d, inv_block));
                } else if (!stable_position) {
                    v_t = hn::Mul(hn::Add(v_sample_offset, hn::Set(d, 1.0f)), hn::Set(d, inv_block));
                }

                auto v_phase = hn::MulAdd(v_lane, hn::Set(d, base_phase_inc), hn::Set(d, phase));
                if constexpr (HasDrift) {
                    HWY_ALIGN float phase_arr[hn::MaxLanes(d)];
                    for (size_t i = 0; i < lanes; ++i) {
                        float phase_inc = base_phase_inc;
                        if (has_pitch_mod) {
                            float pitch_offset = vb.pitch_lane_base[vi]
                                + (vb.pitch_mod_audio[vi] ? vb.pitch_mod_audio[vi][s + i] : 0.0f);
                            if (std::abs(pitch_offset) > 1.0e-6f) {
                                phase_inc *= std::pow(2.0f, pitch_offset / 12.0f);
                            }
                        }
                        float dp = drift_phase + drift_inc * static_cast<float>(offset + i);
                        if (dp >= 2.0f * static_cast<float>(M_PI)) {
                            dp -= 2.0f * static_cast<float>(M_PI);
                            if (dp >= 2.0f * static_cast<float>(M_PI)) {
                                dp = std::fmod(dp, 2.0f * static_cast<float>(M_PI));
                            }
                        }
                        phase_inc *= dsp::cents_to_ratio(std::sin(dp) * drift_scale);
                        phase_arr[i] = phase;
                        phase = wrap_phase_scalar(phase + phase_inc);
                    }
                    v_phase = hn::Load(d, phase_arr);
                } else if (has_pitch_mod) {
                    HWY_ALIGN float phase_arr[hn::MaxLanes(d)];
                    for (size_t i = 0; i < lanes; ++i) {
                        float phase_inc = base_phase_inc;
                        float pitch_offset = vb.pitch_lane_base[vi]
                            + (vb.pitch_mod_audio[vi] ? vb.pitch_mod_audio[vi][s + i] : 0.0f);
                        if (std::abs(pitch_offset) > 1.0e-6f) {
                            phase_inc *= std::pow(2.0f, pitch_offset / 12.0f);
                        }
                        phase_arr[i] = phase;
                        phase = wrap_phase_scalar(phase + phase_inc);
                    }
                    v_phase = hn::Load(d, phase_arr);
                } else {
                    phase = wrap_phase_scalar(phase + base_phase_inc * static_cast<float>(lanes));
                }

                v_phase = hn::Sub(v_phase, hn::IfThenElseZero(hn::Ge(v_phase, v_one), v_one));
                v_phase = hn::Add(v_phase, hn::IfThenElseZero(hn::Lt(v_phase, v_zero), v_one));
                auto v_lookup_phase = v_phase;
                if constexpr (HasWarp) {
                    const auto v_smooth_warp = hn::MulAdd(
                        hn::Set(d, vb.warp_to[vi] - vb.warp_from[vi]), v_t, hn::Set(d, vb.warp_from[vi]));
                    v_lookup_phase = warp_fn(d, v_phase, v_smooth_warp);
                    v_lookup_phase = hn::Clamp(v_lookup_phase, v_zero, hn::Set(d, 0.999999f));
                }

                const auto v_sp = hn::Mul(v_lookup_phase, v_table_size);
                const auto v_i = hn::Min(hn::ConvertTo(di, hn::Floor(v_sp)), v_max_idx);
                const auto v_frac = hn::Sub(v_sp, hn::ConvertTo(d, v_i));

                auto v_frame_blend = hn::Set(d, frame_blend_const);
                auto v_off0 = hn::Add(hn::Set(di, off0_const_base), v_i);
                auto v_off1 = hn::Add(hn::Set(di, off1_const_base), v_i);
                if (!stable_position) {
                    const auto v_smooth_pos = hn::MulAdd(
                        hn::Set(d, pos_delta), v_t, hn::Set(d, vb.pos_from[vi]));
                    const float pos_first = std::clamp(
                        vb.pos_from[vi] + pos_delta * (static_cast<float>(offset + 1) * inv_block), 0.0f, 1.0f);
                    const float pos_last = std::clamp(
                        vb.pos_from[vi] + pos_delta * (static_cast<float>(offset + lanes) * inv_block), 0.0f, 1.0f);
                    const float frame_first = pos_first * frame_count_m1;
                    const float frame_last = pos_last * frame_count_m1;
                    const uint32_t f0_first = std::min(static_cast<uint32_t>(frame_first), pwt.frame_count - 1);
                    const uint32_t f0_last = std::min(static_cast<uint32_t>(frame_last), pwt.frame_count - 1);
                    const uint32_t f1_first = std::min(f0_first + 1, pwt.frame_count - 1);
                    const uint32_t f1_last = std::min(f0_last + 1, pwt.frame_count - 1);
                    const bool frame_indices_stable = (f0_first == f0_last && f1_first == f1_last);

                    v_frame_blend = hn::Sub(
                        hn::Mul(v_smooth_pos, v_frame_count_m1),
                        hn::Set(d, static_cast<float>(f0_first)));
                    v_off0 = hn::Add(
                        hn::Set(di, level_offset + static_cast<int32_t>(f0_first) * static_cast<int32_t>(PreparedWavetable::kGuardedFrameSize)),
                        v_i);
                    v_off1 = hn::Add(
                        hn::Set(di, level_offset + static_cast<int32_t>(f1_first) * static_cast<int32_t>(PreparedWavetable::kGuardedFrameSize)),
                        v_i);
                    if (!frame_indices_stable) {
                        const auto v_frame_pos = hn::Mul(v_smooth_pos, v_frame_count_m1);
                        const auto v_f0 = hn::Min(hn::ConvertTo(di, hn::Floor(v_frame_pos)), v_max_frame);
                        const auto v_f1 = hn::Min(hn::Add(v_f0, v_one_i), v_max_frame);
                        v_frame_blend = hn::Sub(v_frame_pos, hn::ConvertTo(d, v_f0));

                        HWY_ALIGN int32_t f0_arr[hn::MaxLanes(DI())];
                        HWY_ALIGN int32_t f1_arr[hn::MaxLanes(DI())];
                        HWY_ALIGN int32_t i_arr[hn::MaxLanes(DI())];
                        HWY_ALIGN int32_t off0_arr[hn::MaxLanes(DI())];
                        HWY_ALIGN int32_t off1_arr[hn::MaxLanes(DI())];
                        hn::Store(v_f0, di, f0_arr);
                        hn::Store(v_f1, di, f1_arr);
                        hn::Store(v_i, di, i_arr);
                        for (size_t k = 0; k < lanes; ++k) {
                            off0_arr[k] = level_offset
                                + f0_arr[k] * static_cast<int32_t>(PreparedWavetable::kGuardedFrameSize)
                                + i_arr[k];
                            off1_arr[k] = level_offset
                                + f1_arr[k] * static_cast<int32_t>(PreparedWavetable::kGuardedFrameSize)
                                + i_arr[k];
                        }
                        v_off0 = hn::Load(di, off0_arr);
                        v_off1 = hn::Load(di, off1_arr);
                    }
                }
                const auto v_s0_lo = hn::GatherIndex(d, flat_base, v_off0);
                const auto v_s0_hi = hn::GatherIndex(d, flat_base, hn::Add(v_off0, v_one_i));
                const auto v_sample_f0 = hn::MulAdd(hn::Sub(v_s0_hi, v_s0_lo), v_frac, v_s0_lo);
                const auto v_s1_lo = hn::GatherIndex(d, flat_base, v_off1);
                const auto v_s1_hi = hn::GatherIndex(d, flat_base, hn::Add(v_off1, v_one_i));
                const auto v_sample_f1 = hn::MulAdd(hn::Sub(v_s1_hi, v_s1_lo), v_frac, v_s1_lo);
                auto v_sample = hn::MulAdd(hn::Sub(v_sample_f1, v_sample_f0), v_frame_blend, v_sample_f0);

                auto v_voice_gain = hn::Set(d, 1.0f);
                if (has_voice_gain) {
                    v_voice_gain = hn::LoadU(d, vb.voice_gain_audio[vi] + s);
                }

                auto v_declick = v_one;
                if (has_declick) {
                    const auto v_declick_rem = hn::Sub(
                        hn::Set(d, static_cast<float>(declick_start)), v_sample_offset);
                    const auto v_declick_ramp = hn::Mul(
                        hn::Sub(hn::Set(d, static_cast<float>(kDeClickSamples + 1)), v_declick_rem),
                        hn::Set(d, 1.0f / static_cast<float>(kDeClickSamples)));
                    v_declick = hn::IfThenElse(
                        hn::Gt(v_declick_rem, v_zero),
                        v_declick_ramp,
                        v_one);
                }
                const auto v_scaled = hn::Mul(hn::Mul(hn::Mul(v_sample, hn::Set(d, gain)), v_voice_gain), v_declick);
                auto v_l = hn::LoadU(d, out_l + s);
                auto v_r = hn::LoadU(d, out_r + s);
                v_l = hn::MulAdd(v_scaled, hn::Set(d, pan_l), v_l);
                v_r = hn::MulAdd(v_scaled, hn::Set(d, pan_r), v_r);
                hn::StoreU(v_l, d, out_l + s);
                hn::StoreU(v_r, d, out_r + s);
            }

            for (; offset < block_len; ++offset) {
                const uint32_t s = sb + offset;
                float phase_inc = base_phase_inc;
                if (has_pitch_mod) {
                    float pitch_offset = vb.pitch_lane_base[vi]
                        + (vb.pitch_mod_audio[vi] ? vb.pitch_mod_audio[vi][s] : 0.0f);
                    if (std::abs(pitch_offset) > 1.0e-6f) {
                        phase_inc *= std::pow(2.0f, pitch_offset / 12.0f);
                    }
                }
                if constexpr (HasDrift) {
                    float dp = drift_phase + drift_inc * static_cast<float>(offset);
                    if (dp >= 2.0f * static_cast<float>(M_PI)) dp = std::fmod(dp, 2.0f * static_cast<float>(M_PI));
                    phase_inc *= dsp::cents_to_ratio(std::sin(dp) * drift_scale);
                }

                const float t = static_cast<float>(offset + 1) * inv_block;
                const float smooth_pos = vb.pos_from[vi] + (vb.pos_to[vi] - vb.pos_from[vi]) * t;
                const float smooth_warp = vb.warp_from[vi] + (vb.warp_to[vi] - vb.warp_from[vi]) * t;
                const float frame_pos = smooth_pos * frame_count_m1;
                uint32_t f0 = std::min(static_cast<uint32_t>(frame_pos), pwt.frame_count - 1);
                uint32_t f1 = std::min(f0 + 1, pwt.frame_count - 1);
                float frame_blend = frame_pos - static_cast<float>(f0);
                float lookup_phase = phase - std::floor(phase);
                if constexpr (HasWarp) {
                    lookup_phase = std::clamp(dsp::warp_phase(lookup_phase, params.warp_mode, smooth_warp, 0.0f),
                                              0.0f, 0.999999f);
                }
                float sample = lookup_guarded_scalar(pwt, mip, f0, f1, frame_blend, lookup_phase);
                float voice_gain = has_voice_gain ? vb.voice_gain_audio[vi][s] : 1.0f;
                const int rem = declick_start - static_cast<int>(offset);
                float declick = rem > 0
                    ? static_cast<float>(kDeClickSamples - rem + 1) / static_cast<float>(kDeClickSamples)
                    : 1.0f;
                float scaled = sample * gain * voice_gain * declick;
                out_l[s] += scaled * pan_l;
                out_r[s] += scaled * pan_r;
                phase = wrap_phase_scalar(phase + phase_inc);
            }

            ru.phase[slot] = phase;
            if constexpr (HasDrift) {
                drift_phase += drift_inc * static_cast<float>(block_len);
                if (drift_phase >= 2.0f * static_cast<float>(M_PI)) {
                    drift_phase = std::fmod(drift_phase, 2.0f * static_cast<float>(M_PI));
                }
                ru.drift_phase[slot] = drift_phase;
            }
        }

        for (int v = 0; v < vb.voice_count; ++v) {
            vb.declick_remaining[v] = std::max(0, vb.declick_remaining[v] - static_cast<int>(block_len));
        }
    }
}

} // namespace

void RenderBlockSimd(
    float* stereo_out,
    uint32_t frames,
    float sample_rate,
    RenderUnit& ru,
    VoiceBlock& vb,
    const PreparedWavetable& pwt,
    const RenderParams& params
) {
    const WarpFn warp_fn = select_warp_fn(params.warp_mode);
    if (params.warp_mode == dsp::WARP_NONE) {
        if (params.drift_enabled) {
            RenderBlockSimdSampleAxisImpl<false, true>(stereo_out, frames, sample_rate, ru, vb, pwt, params, warp_fn);
        } else {
            RenderBlockSimdSampleAxisImpl<false, false>(stereo_out, frames, sample_rate, ru, vb, pwt, params, warp_fn);
        }
    } else if (params.drift_enabled) {
        RenderBlockSimdSampleAxisImpl<true, true>(stereo_out, frames, sample_rate, ru, vb, pwt, params, warp_fn);
    } else {
        RenderBlockSimdSampleAxisImpl<true, false>(stereo_out, frames, sample_rate, ru, vb, pwt, params, warp_fn);
    }
}

}  // namespace vivid_wavetable::layer::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace vivid_wavetable::layer {

HWY_EXPORT(RenderBlockSimd);

void render_block_simd(
    float* stereo_out,
    uint32_t frames,
    float sample_rate,
    RenderUnit& ru,
    VoiceBlock& vb,
    const PreparedWavetable& pwt,
    const RenderParams& params
) {
    HWY_DYNAMIC_DISPATCH(RenderBlockSimd)(stereo_out, frames, sample_rate, ru, vb, pwt, params);
}

}  // namespace vivid_wavetable::layer
#endif  // HWY_ONCE

#endif  // VIVID_HAS_HIGHWAY
