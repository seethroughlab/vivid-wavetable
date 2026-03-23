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

enum FilterType {
    FILTER_LP12,
    FILTER_LP24,
    FILTER_HP12,
    FILTER_BP,
    FILTER_NOTCH,
    FILTER_COMB,
    FILTER_LADDER,
    FILTER_FORMANT
};

float warp_phase(float phase, int mode, float amount, float last_sample);

struct CombFilterState {
    static constexpr int kMaxDelay = 2048;
    float buffer[kMaxDelay] = {};
    int write_pos = 0;

    void reset();
    float process(float input, float delay_samples, float feedback);
};

struct LadderFilterState {
    float stage[4] = {};

    void reset();
    float process(float input, float cutoff_hz, float reso, float sample_rate);
};

struct FormantFilterState {
    float z1[3] = {};
    float z2[3] = {};

    void reset();
    float process(float input, float morph, float reso, float sample_rate);
};

} // namespace vivid_wavetable::dsp
