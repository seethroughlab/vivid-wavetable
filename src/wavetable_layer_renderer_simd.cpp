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
            RenderBlockSimdImpl<false, true>(stereo_out, frames, sample_rate, ru, vb, pwt, params, warp_fn);
        } else {
            RenderBlockSimdImpl<false, false>(stereo_out, frames, sample_rate, ru, vb, pwt, params, warp_fn);
        }
    } else if (params.drift_enabled) {
        RenderBlockSimdImpl<true, true>(stereo_out, frames, sample_rate, ru, vb, pwt, params, warp_fn);
    } else {
        RenderBlockSimdImpl<true, false>(stereo_out, frames, sample_rate, ru, vb, pwt, params, warp_fn);
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
