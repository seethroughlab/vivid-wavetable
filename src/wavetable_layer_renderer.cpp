#include "wavetable_layer_renderer.h"
#include "wavetable_voice_utils.h"
#include "wavetable_interp.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace vivid_wavetable::layer {

using namespace vivid_wavetable::voice;
using vivid_wavetable::dsp::WarpMode;
using vivid_wavetable::dsp::cents_to_ratio;

// ---------------------------------------------------------------------------
// PreparedWavetable
// ---------------------------------------------------------------------------

void PreparedWavetable::prepare_from(const bank::Wavetable& src) {
    frame_count = src.frame_count;
    if (frame_count == 0) {
        flat_data.clear();
        return;
    }

    // Compute total size: kNumMipLevels levels × frame_count × kGuardedFrameSize
    uint32_t frames_per_level = frame_count * kGuardedFrameSize;
    uint32_t total = bank::kNumMipLevels * frames_per_level;
    flat_data.resize(total);

    for (int level = 0; level < bank::kNumMipLevels; ++level) {
        level_offset[level] = level * frames_per_level;
        const float* src_level = src.level_data(level);

        for (uint32_t fr = 0; fr < frame_count; ++fr) {
            const float* src_frame = src_level + fr * bank::kSamplesPerFrame;
            float* dst_frame = flat_data.data() + level_offset[level] + fr * kGuardedFrameSize;

            // Copy 2048 samples
            std::memcpy(dst_frame, src_frame, bank::kSamplesPerFrame * sizeof(float));
            // Append guard sample (wrap: sample[0])
            dst_frame[bank::kSamplesPerFrame] = src_frame[0];
        }
    }
}

// ---------------------------------------------------------------------------
// Linear table lookup with guard samples (no modulo)
// ---------------------------------------------------------------------------

static inline float lookup_linear(const float* frame_data, float phase) {
    float sp = phase * static_cast<float>(bank::kSamplesPerFrame);
    int i = static_cast<int>(sp);
    i = std::clamp(i, 0, static_cast<int>(bank::kSamplesPerFrame) - 1);
    float frac = sp - static_cast<float>(i);
    return frame_data[i] + (frame_data[i + 1] - frame_data[i]) * frac;
}

static inline float sample_guarded(const PreparedWavetable& pwt, int level, uint32_t f0,
                                   uint32_t f1, float frame_blend, float phase) {
    const float* d0 = pwt.frame_data(level, f0);
    float s0 = lookup_linear(d0, phase);
    if (f0 == f1) return s0;
    const float* d1 = pwt.frame_data(level, f1);
    float s1 = lookup_linear(d1, phase);
    return s0 + (s1 - s0) * frame_blend;
}

// ---------------------------------------------------------------------------
// render_block_scalar
// ---------------------------------------------------------------------------

void render_block_scalar(
    float* stereo_out,
    uint32_t frames,
    float sample_rate,
    RenderUnit& ru,
    VoiceBlock& vb,
    const PreparedWavetable& pwt,
    const RenderParams& params
) {
    if (ru.active_count == 0 || pwt.frame_count == 0) return;

    float* out_l = stereo_out;
    float* out_r = stereo_out + frames;

    const float drift_scale = params.drift_amount * 8.0f; // max cents of drift
    const float frame_count_m1 = static_cast<float>(std::max(pwt.frame_count, 1u) - 1);

    for (uint32_t sb = 0; sb < frames; sb += kControlSubBlock) {
        uint32_t block_len = std::min(static_cast<uint32_t>(kControlSubBlock), frames - sb);
        float inv_block = 1.0f / static_cast<float>(block_len);

        // Update per-voice sub-block smoothing targets
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

        for (uint32_t offset = 0; offset < block_len; ++offset) {
            uint32_t s = sb + offset;
            float t = static_cast<float>(offset + 1) * inv_block;

            for (int slot = 0; slot < ru.active_count; ++slot) {
                int vi = ru.voice_idx[slot];

                // Audio-rate pitch modulation
                float phase_inc = ru.phase_inc[slot];
                float pitch_offset = vb.pitch_lane_base[vi];
                if (vb.pitch_mod_audio[vi]) {
                    pitch_offset += vb.pitch_mod_audio[vi][s];
                }
                if (std::abs(pitch_offset) > 1e-6f) {
                    phase_inc *= std::pow(2.0f, pitch_offset / 12.0f);
                }

                // Drift
                if (params.drift_enabled) {
                    float drift_cents = std::sin(ru.drift_phase[slot]) * drift_scale;
                    phase_inc *= cents_to_ratio(drift_cents);
                    ru.drift_phase[slot] += ru.drift_phase_inc[slot];
                    if (ru.drift_phase[slot] >= 2.0f * static_cast<float>(M_PI))
                        ru.drift_phase[slot] -= 2.0f * static_cast<float>(M_PI);
                }

                // Interpolated position and warp
                float smooth_pos = vb.pos_from[vi] + (vb.pos_to[vi] - vb.pos_from[vi]) * t;
                float smooth_warp = vb.warp_from[vi] + (vb.warp_to[vi] - vb.warp_from[vi]) * t;

                // Frame plan from smooth_pos
                float frame_pos = smooth_pos * frame_count_m1;
                uint32_t f0 = static_cast<uint32_t>(frame_pos);
                f0 = std::min(f0, pwt.frame_count - 1);
                uint32_t f1 = std::min(f0 + 1, pwt.frame_count - 1);
                float frame_blend = frame_pos - static_cast<float>(f0);

                // Warp phase
                float phase = ru.phase[slot];
                phase = phase - std::floor(phase); // ensure [0,1)
                if (params.warp_mode != 0) {
                    phase = vivid_wavetable::dsp::warp_phase(phase, params.warp_mode, smooth_warp, 0.0f);
                    phase = std::clamp(phase, 0.0f, 0.999999f);
                }

                // Table lookup
                int mip = ru.mip_level[slot];
                float sample = sample_guarded(pwt, mip, f0, f1, frame_blend, phase);

                // Voice gain (audio-rate envelope)
                float voice_gain = 1.0f;
                if (vb.voice_gain_audio[vi]) {
                    voice_gain = vb.voice_gain_audio[vi][s];
                }

                // Declick
                float declick = 1.0f;
                if (vb.declick_remaining[vi] > 0) {
                    declick = static_cast<float>(kDeClickSamples - vb.declick_remaining[vi] + 1)
                              / static_cast<float>(kDeClickSamples);
                }

                // Accumulate to stereo
                float scaled = sample * ru.gain[slot] * voice_gain * declick;
                out_l[s] += scaled * ru.pan_l[slot];
                out_r[s] += scaled * ru.pan_r[slot];

                // Advance phase
                ru.phase[slot] += phase_inc;
                if (ru.phase[slot] >= 1.0f) ru.phase[slot] -= 1.0f;
                if (ru.phase[slot] < 0.0f) ru.phase[slot] += 1.0f;
            }

            // Advance declick counters (once per sample, per voice)
            for (int v = 0; v < vb.voice_count; ++v) {
                if (vb.declick_remaining[v] > 0) --vb.declick_remaining[v];
            }
        }
    }
}

} // namespace vivid_wavetable::layer
