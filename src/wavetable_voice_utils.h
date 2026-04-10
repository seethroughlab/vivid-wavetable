#pragma once

#include <cmath>
#include <cstdint>

namespace vivid_wavetable::voice {

inline float wrap_phase(double phase) {
    phase -= std::floor(phase);
    return static_cast<float>(phase);
}

inline float smoothing_coeff(float sample_rate, float smooth_ms) {
    if (smooth_ms <= 0.0f || sample_rate <= 0.0f) return 1.0f;
    float samples = smooth_ms * 0.001f * sample_rate;
    if (samples <= 1.0f) return 1.0f;
    return 1.0f - std::exp(-1.0f / samples);
}

inline uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

inline float hash01(uint32_t seed) {
    return static_cast<float>(hash_u32(seed) & 0x00ffffffU) / static_cast<float>(0x01000000U);
}

inline float normalized_unison_position(int index, int count) {
    if (count <= 1) return 0.0f;
    return (static_cast<float>(index) / static_cast<float>(count - 1)) * 2.0f - 1.0f;
}

inline float unison_detune_offset(int index, int count, float spread_cents,
                                  int spread_mode, uint32_t lane_seed) {
    float linear = normalized_unison_position(index, count);
    switch (spread_mode) {
        case 1:
            linear = std::copysign(linear * linear, linear);
            break;
        case 2: {
            float mag = 0.35f + 0.65f * hash01(lane_seed + static_cast<uint32_t>(index * 17));
            linear *= mag;
            break;
        }
        default:
            break;
    }
    return linear * spread_cents;
}

inline float unison_pan_position(int index, int count, float stereo_depth) {
    return normalized_unison_position(index, count) * stereo_depth;
}

inline float base_phase_offset(int index, int count, bool stereo_pairs,
                               float stereo_phase, uint32_t lane_seed) {
    float offset = 0.0f;
    if (stereo_pairs && count > 1 && stereo_phase > 0.0f) {
        offset += normalized_unison_position(index, count) * stereo_phase * 0.18f;
    }
    if (!stereo_pairs && count > 1 && stereo_phase > 0.0f) {
        offset += (hash01(lane_seed + static_cast<uint32_t>(index * 97)) - 0.5f) * stereo_phase * 0.12f;
    }
    return offset;
}

inline float stereo_pair_phase_shift(int index, int count, float stereo_phase,
                                     uint32_t lane_seed) {
    if (count <= 1 || stereo_phase <= 0.0f) return 0.0f;
    float contour = 0.6f + 0.4f * std::abs(normalized_unison_position(index, count));
    float seeded = 0.04f + 0.08f * hash01(lane_seed + static_cast<uint32_t>(index * 173));
    return stereo_phase * contour * seeded;
}

// phase_reset_mode: 0=FreeRun, 1=Reset, 2=Randomized
inline float gate_on_phase(int phase_reset_mode, float start_phase_value,
                           float phase_random_amount, float base_offset,
                           int index, uint32_t lane_seed) {
    float phase = start_phase_value + base_offset;
    if (phase_reset_mode == 2) { // Randomized
        float r = hash01(lane_seed + static_cast<uint32_t>(index * 131));
        phase += (r - 0.5f) * phase_random_amount;
    }
    phase -= std::floor(phase);
    return phase;
}

} // namespace vivid_wavetable::voice
