#include "wavetable_dsp.h"

#include <algorithm>
#include <cmath>

namespace vivid_wavetable::dsp {

float warp_phase(float phase, int mode, float amount, float last_sample) {
    if (amount <= 0.0f || mode == WARP_NONE) return phase;
    phase = phase - std::floor(phase);

    switch (mode) {
        case WARP_SYNC: {
            float r = 1.0f + amount * 7.0f;
            float sp = phase * r;
            return sp - std::floor(sp);
        }
        case WARP_BEND_PLUS:
            return std::pow(phase, 1.0f + amount * 3.0f);
        case WARP_BEND_MINUS:
            return std::pow(phase, 1.0f / (1.0f + amount * 3.0f));
        case WARP_MIRROR: {
            float mid = 0.5f - amount * 0.3f;
            if (phase > mid) return mid - (phase - mid);
            return phase / mid * 0.5f;
        }
        case WARP_ASYM: {
            float stretch = 0.5f + amount * 0.3f;
            if (phase < 0.5f) return (phase / 0.5f) * stretch;
            return stretch + ((phase - 0.5f) / 0.5f) * (1.0f - stretch);
        }
        case WARP_QUANTIZE: {
            int steps = std::max(4, static_cast<int>(256.0f - amount * 252.0f));
            return std::floor(phase * static_cast<float>(steps)) / static_cast<float>(steps);
        }
        case WARP_FM: {
            float mp = phase + last_sample * amount * 0.5f;
            return mp - std::floor(mp);
        }
        case WARP_FLIP:
            if (phase >= 0.5f) {
                float flipped = 1.0f - phase;
                return phase * (1.0f - amount) + flipped * amount;
            }
            return phase;
        default:
            return phase;
    }
}

float MotionSmoother::process(float target, float coefficient) {
    coefficient = std::clamp(coefficient, 0.0f, 1.0f);
    if (!initialized) {
        reset(target);
        return value;
    }
    value += (target - value) * coefficient;
    return value;
}

} // namespace vivid_wavetable::dsp
