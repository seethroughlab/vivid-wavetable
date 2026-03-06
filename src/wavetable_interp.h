#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace vivid_wavetable {
namespace interp {

inline float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

inline float smoothstep01(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

inline float catmull_rom(float p0, float p1, float p2, float p3, float t) {
    float a = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
    float b = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
    float c = -0.5f * p0 + 0.5f * p2;
    float d = p1;
    return ((a * t + b) * t + c) * t + d;
}

inline float sample_periodic_catmull(const float* data, uint32_t sample_count, float phase) {
    if (!data || sample_count == 0) return 0.0f;
    if (sample_count == 1) return data[0];

    phase = phase - std::floor(phase);
    float sp = phase * static_cast<float>(sample_count);
    uint32_t i1 = static_cast<uint32_t>(sp) % sample_count;
    float t = sp - std::floor(sp);

    uint32_t i0 = (i1 + sample_count - 1) % sample_count;
    uint32_t i2 = (i1 + 1) % sample_count;
    uint32_t i3 = (i1 + 2) % sample_count;
    return catmull_rom(data[i0], data[i1], data[i2], data[i3], t);
}

} // namespace interp
} // namespace vivid_wavetable
