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
    FILTER_LP12,       // 0
    FILTER_LP24,       // 1
    FILTER_HP12,       // 2
    FILTER_BP,         // 3
    FILTER_NOTCH,      // 4
    FILTER_COMB,       // 5
    FILTER_LADDER,     // 6
    FILTER_FORMANT,    // 7
    // --- new types (indices 8+) ---
    FILTER_HP24,       // 8  - 4-pole highpass
    FILTER_PEAK,       // 9  - peaking/bell EQ
    FILTER_ALLPASS,    // 10 - phase shift, no amplitude change
    FILTER_BP24,       // 11 - tight bandpass (LP24+HP12 in series)
    FILTER_DIODE,      // 12 - diode ladder (aggressive saturation)
    FILTER_MS20,       // 13 - Korg MS-20 style (Sallen-Key, self-oscillating)
    FILTER_COUNT
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

struct DiodeLadderState {
    float stage[4] = {};
    float feedback = 0;

    void reset();
    float process(float input, float cutoff_hz, float reso, float sample_rate);
};

struct MS20FilterState {
    float hp = 0, bp = 0, lp = 0;
    float s1 = 0, s2 = 0;

    void reset();
    float process(float input, float cutoff_hz, float reso, float sample_rate);
};

} // namespace vivid_wavetable::dsp
