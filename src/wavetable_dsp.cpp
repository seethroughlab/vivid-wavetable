#include "wavetable_dsp.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace vivid_wavetable::dsp {
namespace {

constexpr float kPi = static_cast<float>(M_PI);
constexpr float kTwoPi = 2.0f * kPi;

} // namespace

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

void CombFilterState::reset() {
    std::memset(buffer, 0, sizeof(buffer));
    write_pos = 0;
}

float CombFilterState::process(float input, float delay_samples, float feedback) {
    delay_samples = std::clamp(delay_samples, 1.0f, static_cast<float>(kMaxDelay - 1));
    feedback = std::clamp(feedback, -0.98f, 0.98f);

    int d_int = static_cast<int>(delay_samples);
    float d_frac = delay_samples - static_cast<float>(d_int);

    int read0 = (write_pos - d_int + kMaxDelay) % kMaxDelay;
    int read1 = (read0 - 1 + kMaxDelay) % kMaxDelay;

    float delayed = buffer[read0] * (1.0f - d_frac) + buffer[read1] * d_frac;

    float out = input + delayed * feedback;
    buffer[write_pos] = out;
    write_pos = (write_pos + 1) % kMaxDelay;
    return out;
}

void LadderFilterState::reset() {
    stage[0] = stage[1] = stage[2] = stage[3] = 0.0f;
}

float LadderFilterState::process(float input, float cutoff_hz, float reso, float sample_rate) {
    cutoff_hz = std::clamp(cutoff_hz, 20.0f, sample_rate * 0.45f);
    float g = std::tan(kPi * cutoff_hz / sample_rate);
    float fb = reso * 4.0f;
    float x = std::tanh(input - fb * stage[3]);

    for (int i = 0; i < 4; ++i) {
        float v = (x - stage[i]) * g / (1.0f + g);
        float y = v + stage[i];
        stage[i] = y + v;
        x = y;
    }
    return x;
}

void FormantFilterState::reset() {
    std::memset(z1, 0, sizeof(z1));
    std::memset(z2, 0, sizeof(z2));
}

float FormantFilterState::process(float input, float morph, float reso, float sample_rate) {
    static constexpr float FORMANTS[5][3] = {
        {800.0f, 1150.0f, 2900.0f},
        {350.0f, 2000.0f, 2800.0f},
        {270.0f, 2300.0f, 3000.0f},
        {450.0f, 800.0f, 2830.0f},
        {325.0f, 700.0f, 2530.0f},
    };
    static constexpr float GAINS[3] = {1.0f, 0.5f, 0.25f};

    float pos = morph * 4.0f;
    int idx = std::min(static_cast<int>(pos), 3);
    float frac = pos - static_cast<float>(idx);

    float Q = 1.0f + reso * 19.0f;
    float out = 0.0f;

    for (int b = 0; b < 3; ++b) {
        float freq = FORMANTS[idx][b] * (1.0f - frac) + FORMANTS[idx + 1][b] * frac;
        freq = std::min(freq, sample_rate * 0.45f);

        float omega = kTwoPi * freq / sample_rate;
        float sin_w = std::sin(omega);
        float cos_w = std::cos(omega);
        float alpha = sin_w / (2.0f * Q);

        float b0 = sin_w * 0.5f;
        float b1 = 0.0f;
        float b2 = -sin_w * 0.5f;
        float a0 = 1.0f + alpha;
        float a1 = -2.0f * cos_w;
        float a2 = 1.0f - alpha;

        float inv_a0 = 1.0f / a0;
        b0 *= inv_a0;
        b1 *= inv_a0;
        b2 *= inv_a0;
        a1 *= inv_a0;
        a2 *= inv_a0;

        float y = b0 * input + z1[b];
        z1[b] = b1 * input - a1 * y + z2[b];
        z2[b] = b2 * input - a2 * y;

        out += y * GAINS[b];
    }

    return out;
}

void DiodeLadderState::reset() {
    stage[0] = stage[1] = stage[2] = stage[3] = 0.0f;
    feedback = 0.0f;
}

float DiodeLadderState::process(float input, float cutoff_hz, float reso, float sample_rate) {
    // Diode ladder: asymmetric clipping per stage + higher feedback saturation
    cutoff_hz = std::clamp(cutoff_hz, 20.0f, sample_rate * 0.45f);
    float g = std::tan(kPi * cutoff_hz / sample_rate);
    float fb = reso * 4.5f;  // slightly higher feedback range than standard ladder

    // Asymmetric soft clip: positive side saturates faster
    auto diode_clip = [](float x) -> float {
        if (x > 0.0f) return std::tanh(x * 1.5f);
        return std::tanh(x * 0.8f);
    };

    float x = diode_clip(input - fb * feedback);

    for (int i = 0; i < 4; ++i) {
        float v = (x - stage[i]) * g / (1.0f + g);
        float y = v + stage[i];
        stage[i] = y + v;
        x = diode_clip(y);
    }

    feedback = x;
    return x;
}

void MS20FilterState::reset() {
    hp = bp = lp = s1 = s2 = 0.0f;
}

float MS20FilterState::process(float input, float cutoff_hz, float reso, float sample_rate) {
    // Korg MS-20 style: Sallen-Key topology with high-feedback self-oscillation
    cutoff_hz = std::clamp(cutoff_hz, 20.0f, sample_rate * 0.45f);
    float f = 2.0f * std::sin(kPi * cutoff_hz / sample_rate);
    f = std::clamp(f, 0.0f, 1.0f);
    float k = reso * 2.0f;  // resonance drives self-oscillation

    // Saturating feedback path (MS-20 character)
    float fb = std::tanh(k * bp);

    hp = input - lp - fb;
    bp = hp * f + s1;
    lp = bp * f + s2;

    s1 = bp;
    s2 = lp;

    // Output lowpass (classic MS-20 LP mode)
    return lp;
}

} // namespace vivid_wavetable::dsp
