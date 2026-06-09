#pragma once

#include "operator_api/types.h"
#include "operator_api/value_view.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace vivid_wavetable::lane_audio {

inline float clamp01(float x) {
    return std::clamp(x, 0.0f, 1.0f);
}

// read_lane now reads a many-value input view (the value-model successor to the
// removed VividLaneView). Pass &ctx->values[port].
inline float read_lane(const VividValueView* v, int slot, float fallback = 0.0f) {
    const float* d = vivid_value_floats(v);
    uint32_t n = vivid_value_count(v);
    if (d && slot >= 0 && static_cast<uint32_t>(slot) < n) {
        return d[slot];
    }
    return fallback;
}

inline float read_lane(const VividValueView* v, uint32_t slot, float fallback) {
    return read_lane(v, static_cast<int>(slot), fallback);
}

inline float* resolve_mod_channel(float* buf, uint32_t ch_count, uint32_t voice, uint32_t frames) {
    if (!buf || ch_count == 0) return nullptr;
    uint32_t ch = (voice < ch_count) ? voice : ch_count - 1;
    return buf + ch * frames;
}

inline uint32_t resolve_lane_id(const VividValueView* lane_id_view, uint32_t voice_index) {
    const float* d = vivid_value_floats(lane_id_view);
    uint32_t n = vivid_value_count(lane_id_view);
    if (d && voice_index < n) {
        return static_cast<uint32_t>(d[voice_index]);
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
