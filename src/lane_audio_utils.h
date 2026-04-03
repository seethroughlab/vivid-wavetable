#pragma once

#include "operator_api/types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace vivid_wavetable::lane_audio {

inline float clamp01(float x) {
    return std::clamp(x, 0.0f, 1.0f);
}

inline float read_lane(const VividLanePort* lane, int slot, float fallback = 0.0f) {
    if (lane && lane->data && slot >= 0 && static_cast<uint32_t>(slot) < lane->length) {
        return lane->data[slot];
    }
    return fallback;
}

inline float read_lane(const VividLanePort* lane, uint32_t slot, float fallback) {
    return read_lane(lane, static_cast<int>(slot), fallback);
}

inline float* resolve_mod_channel(float* buf, uint32_t ch_count, uint32_t voice, uint32_t frames) {
    if (!buf || ch_count == 0) return nullptr;
    uint32_t ch = (voice < ch_count) ? voice : ch_count - 1;
    return buf + ch * frames;
}

inline uint32_t resolve_lane_id(const VividLanePort* lane_id_lane, uint32_t voice_index) {
    if (lane_id_lane && lane_id_lane->data && voice_index < lane_id_lane->length) {
        return static_cast<uint32_t>(lane_id_lane->data[voice_index]);
    }
    return voice_index;
}

inline float one_pole_coeff(float sample_rate,
                            float cutoff_hz,
                            float min_cutoff_hz = 10.0f,
                            float max_nyquist_scale = 0.45f) {
    cutoff_hz = std::clamp(cutoff_hz, min_cutoff_hz, sample_rate * max_nyquist_scale);
    return 1.0f - std::exp(-2.0f * 3.14159265358979323846f * cutoff_hz / sample_rate);
}

} // namespace vivid_wavetable::lane_audio
