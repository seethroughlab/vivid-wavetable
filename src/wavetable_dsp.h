#pragma once

namespace vivid_wavetable::dsp {

enum WarpMode {
    WARP_NONE,
    WARP_SYNC,
    WARP_BEND_PLUS,
    WARP_BEND_MINUS,
    WARP_MIRROR,
    WARP_ASYM,
    WARP_QUANTIZE,
    WARP_FM,
    WARP_FLIP
};

float warp_phase(float phase, int mode, float amount, float last_sample);

struct MotionSmoother {
    float value = 0.0f;
    bool initialized = false;

    void reset(float next) {
        value = next;
        initialized = true;
    }

    float process(float target, float coefficient);
};

} // namespace vivid_wavetable::dsp
