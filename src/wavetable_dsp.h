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

struct DCBlocker {
    float last_input = 0.0f;
    float last_output = 0.0f;

    void reset() {
        last_input = 0.0f;
        last_output = 0.0f;
    }

    float process(float input, float coefficient = 0.995f);
};

float interaction_depth_curve(float depth);
float interaction_tracking_frequency(float base_frequency, float tracking);
float condition_interaction_input(float input, float gain, DCBlocker& dc_blocker);

} // namespace vivid_wavetable::dsp
