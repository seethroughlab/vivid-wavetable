#include "wavetable_bank.h"

#include "miniaudio.h"
#include "wavetable_interp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace vivid_wavetable::bank {
namespace {

constexpr float kPi = static_cast<float>(M_PI);
constexpr float kTwoPi = 2.0f * kPi;

// Radix-2 Cooley-Tukey FFT/IFFT, in-place, N must be power-of-2.
static void fft_inplace(float* real, float* imag, int N, bool inverse) {
    for (int i = 1, j = 0; i < N; ++i) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }

    for (int len = 2; len <= N; len <<= 1) {
        float ang = kTwoPi / static_cast<float>(len) * (inverse ? -1.0f : 1.0f);
        float w_re = std::cos(ang);
        float w_im = std::sin(ang);
        for (int i = 0; i < N; i += len) {
            float cur_re = 1.0f;
            float cur_im = 0.0f;
            for (int j = 0; j < len / 2; ++j) {
                int u = i + j;
                int v = u + len / 2;
                float t_re = cur_re * real[v] - cur_im * imag[v];
                float t_im = cur_re * imag[v] + cur_im * real[v];
                real[v] = real[u] - t_re;
                imag[v] = imag[u] - t_im;
                real[u] += t_re;
                imag[u] += t_im;
                float next_re = cur_re * w_re - cur_im * w_im;
                cur_im = cur_re * w_im + cur_im * w_re;
                cur_re = next_re;
            }
        }
    }

    if (inverse) {
        float inv_n = 1.0f / static_cast<float>(N);
        for (int i = 0; i < N; ++i) {
            real[i] *= inv_n;
            imag[i] *= inv_n;
        }
    }
}

static float clamp01(float x) {
    return std::clamp(x, 0.0f, 1.0f);
}

static float shape_triangle(float phase) {
    return 4.0f * std::abs(phase - 0.5f) - 1.0f;
}

static float shape_saw(float phase) {
    return 2.0f * phase - 1.0f;
}

static float shape_square(float phase, float width) {
    return phase < width ? 1.0f : -1.0f;
}

static float hash01(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return static_cast<float>(x & 0x00ffffffU) / static_cast<float>(0x01000000U);
}

static void normalize_frame(float* frame) {
    float peak = 0.0f;
    for (uint32_t i = 0; i < kSamplesPerFrame; ++i)
        peak = std::max(peak, std::abs(frame[i]));
    if (peak > 0.00001f) {
        float inv_peak = 1.0f / peak;
        for (uint32_t i = 0; i < kSamplesPerFrame; ++i)
            frame[i] *= inv_peak;
    }
}

static void lowpass_wrap(float* frame, int passes, float center_weight) {
    center_weight = std::clamp(center_weight, 0.0f, 1.0f);
    float neighbor_weight = (1.0f - center_weight) * 0.5f;
    for (int pass = 0; pass < passes; ++pass) {
        float prev = frame[kSamplesPerFrame - 1];
        for (uint32_t i = 0; i < kSamplesPerFrame; ++i) {
            float cur = frame[i];
            float next = frame[(i + 1) % kSamplesPerFrame];
            frame[i] = cur * center_weight + (prev + next) * neighbor_weight;
            prev = cur;
        }
    }
}

static float phase_distort(float phase, float amount) {
    amount = std::clamp(amount, -0.95f, 0.95f);
    if (amount >= 0.0f) {
        float split = 0.5f - amount * 0.35f;
        if (phase < split) return 0.5f * (phase / std::max(split, 0.001f));
        return 0.5f + 0.5f * ((phase - split) / std::max(1.0f - split, 0.001f));
    }
    float bend = -amount;
    float split = 0.5f + bend * 0.35f;
    if (phase < split) return 0.5f * (phase / std::max(split, 0.001f));
    return 0.5f + 0.5f * ((phase - split) / std::max(1.0f - split, 0.001f));
}

static void generate_analog_family(Wavetable& wt, int member) {
    wt.allocate(48);
    for (uint32_t fr = 0; fr < wt.frame_count; ++fr) {
        float* d = wt.frame_ptr(fr);
        float t = static_cast<float>(fr) / static_cast<float>(wt.frame_count - 1);
        float pulse_width = 0.18f + 0.64f * clamp01(0.4f * t + 0.1f * static_cast<float>(member));
        for (uint32_t i = 0; i < kSamplesPerFrame; ++i) {
            float p = static_cast<float>(i) / static_cast<float>(kSamplesPerFrame);
            float sine = std::sin(p * kTwoPi);
            float tri = shape_triangle(p);
            float saw = shape_saw(p);
            float square = shape_square(p, pulse_width);
            float sample = 0.0f;

            switch (member) {
                case MEMBER_CORE:
                    sample = sine * (1.0f - t) + saw * t;
                    break;
                case MEMBER_SOFT:
                    sample = tri * (0.7f + 0.3f * (1.0f - t)) + 0.25f * sine;
                    sample = std::tanh(sample * 0.85f);
                    break;
                case MEMBER_RICH: {
                    sample = 0.55f * saw + 0.25f * square + 0.2f * tri;
                    for (int h = 2; h <= 10; ++h) {
                        float drift = 0.002f * std::sin((t + static_cast<float>(member)) * 7.0f * static_cast<float>(h));
                        sample += 0.15f / static_cast<float>(h) *
                                  std::sin((p + drift) * kTwoPi * static_cast<float>(h));
                    }
                    sample = std::tanh(sample * 0.9f);
                    break;
                }
                case MEMBER_HOLLOW:
                    sample = 0.8f * tri - 0.35f * std::sin(p * kTwoPi * 2.0f) + 0.2f * std::sin(p * kTwoPi * 5.0f);
                    sample *= 0.7f + 0.3f * (1.0f - t);
                    break;
                case MEMBER_SWEEP:
                    sample = shape_square(p, pulse_width) * (0.6f + 0.4f * t) + saw * (0.4f - 0.2f * t);
                    lowpass_wrap(d, 0, 0.75f);
                    break;
                case MEMBER_GLASS:
                    sample = 0.65f * tri + 0.25f * std::sin(p * kTwoPi * 3.0f) + 0.1f * std::sin(p * kTwoPi * 7.0f);
                    sample = std::tanh(sample * (0.9f + 0.2f * t));
                    break;
                case MEMBER_EDGE:
                    sample = 0.7f * saw + 0.45f * shape_square(p, 0.48f - 0.18f * t);
                    sample = std::tanh(sample * 1.1f);
                    break;
                case MEMBER_AIR:
                    sample = 0.55f * sine + 0.25f * tri + 0.12f * std::sin(p * kTwoPi * 6.0f) + 0.08f * std::sin(p * kTwoPi * 11.0f);
                    sample *= 0.8f + 0.2f * t;
                    break;
            }

            d[i] = sample;
        }
        if (member == MEMBER_SWEEP)
            lowpass_wrap(d, 2, 0.74f);
        else if (member == MEMBER_SOFT || member == MEMBER_AIR)
            lowpass_wrap(d, 1, 0.82f);
        normalize_frame(d);
    }
}

static void generate_digital_family(Wavetable& wt, int member) {
    wt.allocate(48);
    for (uint32_t fr = 0; fr < wt.frame_count; ++fr) {
        float* d = wt.frame_ptr(fr);
        float t = static_cast<float>(fr) / static_cast<float>(wt.frame_count - 1);
        for (uint32_t i = 0; i < kSamplesPerFrame; ++i) {
            float p = static_cast<float>(i) / static_cast<float>(kSamplesPerFrame);
            float sample = 0.0f;
            switch (member) {
                case MEMBER_CORE: {
                    float mod = std::sin(p * kTwoPi * (1.0f + std::floor(t * 4.0f)));
                    sample = std::sin(p * kTwoPi + mod * (2.5f + t * 4.0f));
                    break;
                }
                case MEMBER_SOFT: {
                    float pd = phase_distort(p, 0.25f + 0.35f * t);
                    sample = 0.8f * std::sin(pd * kTwoPi) + 0.2f * std::sin(pd * kTwoPi * 2.0f);
                    break;
                }
                case MEMBER_RICH: {
                    float mod = std::sin(p * kTwoPi * (2.0f + std::floor(t * 5.0f)));
                    float carrier = std::sin(p * kTwoPi * (1.0f + 0.5f * t) + mod * (4.0f + 6.0f * t));
                    sample = 0.7f * carrier + 0.3f * std::sin(p * kTwoPi * 3.0f);
                    break;
                }
                case MEMBER_HOLLOW: {
                    float pd = phase_distort(p, -0.55f + 0.2f * t);
                    sample = std::sin(pd * kTwoPi) - 0.4f * std::sin(pd * kTwoPi * 2.0f);
                    break;
                }
                case MEMBER_SWEEP: {
                    float ratio = 1.0f + 6.0f * t;
                    sample = std::sin((p + 0.12f * std::sin(p * kTwoPi * ratio)) * kTwoPi);
                    break;
                }
                case MEMBER_GLASS: {
                    float ratio = 1.6f + 3.2f * t;
                    float shimmer = std::sin(p * kTwoPi * ratio) * (0.7f + 1.3f * t);
                    sample = 0.78f * std::sin(p * kTwoPi + shimmer);
                    sample += 0.14f * std::sin(p * kTwoPi * 3.0f + t * 0.35f * kPi);
                    sample += 0.10f * std::sin(p * kTwoPi * 5.0f);
                    break;
                }
                case MEMBER_EDGE: {
                    float q = std::round((std::sin(p * kTwoPi * (1.0f + 3.0f * t)) * 0.5f + 0.5f) * 6.0f) / 3.0f - 1.0f;
                    sample = 0.65f * std::sin(p * kTwoPi) + 0.6f * q;
                    break;
                }
                case MEMBER_AIR: {
                    float fold = std::sin(p * kTwoPi) + 0.4f * std::sin(p * kTwoPi * 8.0f + t * kPi);
                    sample = std::tanh(fold * (0.8f + 0.5f * t));
                    break;
                }
            }
            d[i] = sample;
        }
        if (member == MEMBER_SOFT)
            lowpass_wrap(d, 2, 0.84f);
        else if (member == MEMBER_GLASS)
            lowpass_wrap(d, 1, 0.86f);
        normalize_frame(d);
    }
}

static void generate_vocal_family(Wavetable& wt, int member) {
    wt.allocate(64);
    const float formants[6][4] = {
        {800.0f, 1150.0f, 2800.0f, 3400.0f},
        {400.0f, 2000.0f, 2550.0f, 3300.0f},
        {350.0f, 2700.0f, 2900.0f, 3500.0f},
        {450.0f, 800.0f,  2830.0f, 3400.0f},
        {325.0f, 700.0f,  2530.0f, 3300.0f},
        {650.0f, 1300.0f, 2350.0f, 3200.0f}
    };
    const float amps[8][4] = {
        {1.0f, 0.65f, 0.24f, 0.10f},
        {1.0f, 0.48f, 0.32f, 0.11f},
        {1.0f, 0.22f, 0.34f, 0.10f},
        {1.0f, 0.85f, 0.14f, 0.06f},
        {1.0f, 0.70f, 0.20f, 0.07f},
        {1.0f, 0.60f, 0.30f, 0.10f},
        {1.0f, 0.55f, 0.26f, 0.12f},
        {1.0f, 0.42f, 0.22f, 0.14f}
    };
    for (uint32_t fr = 0; fr < wt.frame_count; ++fr) {
        float* d = wt.frame_ptr(fr);
        float t = static_cast<float>(fr) / static_cast<float>(wt.frame_count - 1);
        float path = t * 5.0f;
        int v0 = static_cast<int>(path);
        int v1 = std::min(v0 + 1, 5);
        float blend = path - static_cast<float>(v0);
        float brightness = 0.8f + 0.08f * static_cast<float>(member);
        float emphasis = 1.0f + 0.12f * static_cast<float>(member == MEMBER_GLASS || member == MEMBER_EDGE);

        for (uint32_t i = 0; i < kSamplesPerFrame; ++i) {
            float p = static_cast<float>(i) / static_cast<float>(kSamplesPerFrame);
            float fundamental = 100.0f + 20.0f * t;
            float sample = 0.0f;
            for (int h = 1; h <= 56; ++h) {
                float freq = fundamental * static_cast<float>(h);
                float amp_sum = 0.0f;
                for (int f = 0; f < 4; ++f) {
                    float center = formants[v0][f] * (1.0f - blend) + formants[v1][f] * blend;
                    float weight = amps[member][f] * emphasis;
                    float bw = 80.0f + 35.0f * static_cast<float>(f) + 15.0f * static_cast<float>(member == MEMBER_SOFT);
                    float dist = (freq - center) / bw;
                    amp_sum += weight * std::exp(-0.5f * dist * dist);
                }
                float harmonic_tilt = 1.0f / std::pow(static_cast<float>(h), brightness);
                sample += amp_sum * harmonic_tilt * std::sin(p * kTwoPi * static_cast<float>(h));
            }
            d[i] = std::tanh(sample * 0.42f);
        }
        if (member == MEMBER_SOFT || member == MEMBER_AIR)
            lowpass_wrap(d, 1, 0.84f);
        normalize_frame(d);
    }
}

static void generate_metallic_family(Wavetable& wt, int member) {
    wt.allocate(48);
    const float ratio_sets[8][8] = {
        {1.0f, 1.5f, 2.3f, 3.1f, 4.7f, 6.2f, 0.0f, 0.0f},
        {1.0f, 1.25f, 1.82f, 2.4f, 3.2f, 4.6f, 0.0f, 0.0f},
        {1.0f, 1.34f, 1.87f, 2.58f, 3.24f, 3.81f, 4.53f, 0.0f},
        {1.0f, 1.19f, 1.56f, 2.0f, 2.44f, 2.83f, 3.15f, 3.74f},
        {1.0f, 1.42f, 1.78f, 2.15f, 2.76f, 3.42f, 4.20f, 5.10f},
        {1.0f, 2.0f, 3.0f, 4.2f, 5.4f, 6.8f, 0.0f, 0.0f},
        {1.0f, 1.09f, 1.33f, 1.71f, 2.27f, 2.88f, 3.62f, 4.58f},
        {1.0f, 1.62f, 2.51f, 3.73f, 4.95f, 6.40f, 0.0f, 0.0f}
    };
    const float amp_sets[8][8] = {
        {1.0f, 0.62f, 0.38f, 0.28f, 0.18f, 0.12f, 0.0f, 0.0f},
        {1.0f, 0.54f, 0.36f, 0.22f, 0.15f, 0.10f, 0.0f, 0.0f},
        {1.0f, 0.72f, 0.52f, 0.40f, 0.28f, 0.22f, 0.16f, 0.0f},
        {1.0f, 0.70f, 0.48f, 0.36f, 0.28f, 0.18f, 0.13f, 0.10f},
        {1.0f, 0.66f, 0.50f, 0.40f, 0.30f, 0.24f, 0.18f, 0.13f},
        {1.0f, 0.58f, 0.34f, 0.25f, 0.18f, 0.12f, 0.0f, 0.0f},
        {1.0f, 0.86f, 0.62f, 0.44f, 0.28f, 0.18f, 0.14f, 0.11f},
        {1.0f, 0.50f, 0.32f, 0.22f, 0.16f, 0.12f, 0.0f, 0.0f}
    };

    for (uint32_t fr = 0; fr < wt.frame_count; ++fr) {
        float* d = wt.frame_ptr(fr);
        float t = static_cast<float>(fr) / static_cast<float>(wt.frame_count - 1);
        for (uint32_t i = 0; i < kSamplesPerFrame; ++i) {
            float p = static_cast<float>(i) / static_cast<float>(kSamplesPerFrame);
            float sample = 0.0f;
            for (int h = 0; h < 8; ++h) {
                float ratio = ratio_sets[member][h];
                float amp = amp_sets[member][h];
                if (ratio <= 0.0f || amp <= 0.0f) continue;
                float motion = 1.0f + 0.02f * std::sin((t * 3.0f + static_cast<float>(h)) * kPi);
                sample += amp * std::sin(p * kTwoPi * ratio * motion + t * static_cast<float>(h) * 0.6f);
            }
            if (member == MEMBER_EDGE || member == MEMBER_GLASS)
                sample = std::tanh(sample * 1.1f);
            d[i] = sample;
        }
        normalize_frame(d);
    }
}

static void generate_harmonic_family(Wavetable& wt, int member) {
    wt.allocate(64);
    for (uint32_t fr = 0; fr < wt.frame_count; ++fr) {
        float* d = wt.frame_ptr(fr);
        float t = static_cast<float>(fr) / static_cast<float>(wt.frame_count - 1);
        for (uint32_t i = 0; i < kSamplesPerFrame; ++i) {
            float p = static_cast<float>(i) / static_cast<float>(kSamplesPerFrame);
            float sample = 0.0f;
            for (int h = 1; h <= 64; ++h) {
                float amp = 1.0f;
                switch (member) {
                    case MEMBER_CORE:
                        amp = 1.0f / std::pow(static_cast<float>(h), 1.2f + 0.8f * t);
                        break;
                    case MEMBER_SOFT:
                        amp = 1.0f / std::pow(static_cast<float>(h), 1.8f + 0.7f * t);
                        break;
                    case MEMBER_RICH:
                        amp = 1.0f / std::pow(static_cast<float>(h), 0.95f + 0.45f * t);
                        break;
                    case MEMBER_HOLLOW:
                        amp = (h % 2 == 1) ? (1.0f / std::pow(static_cast<float>(h), 1.15f)) : (0.18f / static_cast<float>(h));
                        break;
                    case MEMBER_SWEEP:
                        amp = 1.0f / std::pow(static_cast<float>(h), 1.4f + std::sin(t * kPi) * 0.5f);
                        break;
                    case MEMBER_GLASS:
                        amp = 1.0f / std::pow(static_cast<float>(h), 1.0f + 0.2f * t);
                        amp *= (h % 3 == 0) ? 1.25f : 0.82f;
                        break;
                    case MEMBER_EDGE:
                        amp = 1.0f / std::pow(static_cast<float>(h), 0.85f + 0.15f * t);
                        amp *= (h < 8) ? 0.8f : 1.15f;
                        break;
                    case MEMBER_AIR:
                        amp = 1.0f / std::pow(static_cast<float>(h), 1.6f);
                        amp *= 0.8f + 0.4f * std::sin(static_cast<float>(h) * 0.4f + t * kPi);
                        break;
                }
                sample += amp * std::sin(p * kTwoPi * static_cast<float>(h));
            }
            d[i] = sample;
        }
        normalize_frame(d);
    }
}

static void generate_texture_family(Wavetable& wt, int member) {
    wt.allocate(48);
    uint32_t seed = 0x1234abcdu + static_cast<uint32_t>(member * 7919);
    for (uint32_t fr = 0; fr < wt.frame_count; ++fr) {
        float* d = wt.frame_ptr(fr);
        float t = static_cast<float>(fr) / static_cast<float>(wt.frame_count - 1);
        for (uint32_t i = 0; i < kSamplesPerFrame; ++i) {
            float p = static_cast<float>(i) / static_cast<float>(kSamplesPerFrame);
            seed = seed * 1664525u + 1013904223u;
            float noise = hash01(seed) * 2.0f - 1.0f;
            float sample = 0.0f;
            switch (member) {
                case MEMBER_CORE:
                    sample = (1.0f - t) * (0.7f * std::sin(p * kTwoPi) + 0.2f * std::sin(p * kTwoPi * 3.0f)) + t * noise * 0.6f;
                    break;
                case MEMBER_SOFT:
                    sample = 0.65f * std::sin(p * kTwoPi) + 0.22f * std::sin(p * kTwoPi * 2.0f) + noise * 0.18f * t;
                    break;
                case MEMBER_RICH:
                    sample = 0.45f * std::sin(p * kTwoPi) + 0.3f * std::sin(p * kTwoPi * 2.0f + t * 3.0f) +
                             0.2f * std::sin(p * kTwoPi * 5.0f) + noise * 0.24f;
                    break;
                case MEMBER_HOLLOW:
                    sample = 0.75f * shape_triangle(p) - 0.25f * std::sin(p * kTwoPi * 2.0f) + noise * 0.12f;
                    break;
                case MEMBER_SWEEP:
                    sample = 0.6f * std::sin((p + 0.12f * std::sin(t * kTwoPi)) * kTwoPi) + noise * (0.2f + 0.25f * t);
                    break;
                case MEMBER_GLASS:
                    sample = 0.55f * std::sin(p * kTwoPi) + 0.16f * std::sin(p * kTwoPi * 8.0f) + noise * 0.22f;
                    break;
                case MEMBER_EDGE:
                    sample = 0.38f * shape_saw(p) + 0.28f * std::sin(p * kTwoPi * 7.0f) + noise * 0.32f;
                    break;
                case MEMBER_AIR:
                    sample = 0.48f * std::sin(p * kTwoPi) + 0.10f * std::sin(p * kTwoPi * 11.0f) + noise * (0.12f + 0.18f * t);
                    break;
            }
            d[i] = sample;
        }
        if (member == MEMBER_SOFT || member == MEMBER_AIR)
            lowpass_wrap(d, 3, 0.86f);
        else
            lowpass_wrap(d, 2, 0.78f);
        normalize_frame(d);
    }
}

} // namespace

void Wavetable::allocate(uint32_t frames) {
    frame_count = std::min(frames, kMaxFrames);
    uint32_t needed = frame_count * kSamplesPerFrame;
    if (data.size() < needed) {
        data.assign(needed, 0.0f);
    } else {
        std::fill_n(data.data(), needed, 0.0f);
    }

    for (int level = 0; level < kNumMipLevels - 1; ++level) {
        if (mip[level].size() < needed) {
            mip[level].assign(needed, 0.0f);
        } else {
            std::fill_n(mip[level].data(), needed, 0.0f);
        }
    }
}

float* Wavetable::frame_ptr(uint32_t frame_index) {
    return data.data() + frame_index * kSamplesPerFrame;
}

void Wavetable::build_mipmaps() {
    const int N = static_cast<int>(kSamplesPerFrame);
    std::vector<float> tmp_re(N), tmp_im(N);
    std::vector<float> freq_re(N), freq_im(N);

    for (uint32_t fr = 0; fr < frame_count; ++fr) {
        const float* src = data.data() + fr * kSamplesPerFrame;
        std::copy(src, src + N, freq_re.data());
        std::fill(freq_im.begin(), freq_im.end(), 0.0f);
        fft_inplace(freq_re.data(), freq_im.data(), N, false);

        for (int level = 1; level < kNumMipLevels; ++level) {
            int max_bin = N / 2 >> level;
            std::copy(freq_re.begin(), freq_re.end(), tmp_re.data());
            std::copy(freq_im.begin(), freq_im.end(), tmp_im.data());

            for (int bin = max_bin + 1; bin <= N / 2; ++bin) {
                tmp_re[bin] = tmp_im[bin] = 0.0f;
                if (bin < N) {
                    tmp_re[N - bin] = tmp_im[N - bin] = 0.0f;
                }
            }

            fft_inplace(tmp_re.data(), tmp_im.data(), N, true);
            float* dst = mip[level - 1].data() + fr * kSamplesPerFrame;
            std::copy(tmp_re.data(), tmp_re.data() + N, dst);
        }
    }
}

const float* Wavetable::level_data(int level) const {
    level = std::clamp(level, 0, kNumMipLevels - 1);
    return (level == 0) ? data.data() : mip[level - 1].data();
}

PreparedMipPlan Wavetable::prepare_mip_plan(float freq_hz, float sample_rate, bool quantize_fast) const {
    PreparedMipPlan plan{};
    if (data.empty() || frame_count == 0) return plan;
    if (!std::isfinite(freq_hz) || freq_hz <= 0.0f) freq_hz = 1.0f;

    float max_h = sample_rate / (2.0f * freq_hz);
    float level_f = std::log2(static_cast<float>(kSamplesPerFrame / 2) / std::max(max_h, 1.0f));
    if (!(level_f >= 0.0f)) level_f = 0.0f;
    if (!(level_f <= static_cast<float>(kNumMipLevels - 1))) {
        level_f = static_cast<float>(kNumMipLevels - 1);
    }

    if (quantize_fast) {
        int nearest = static_cast<int>(std::round(level_f));
        nearest = std::clamp(nearest, 0, kNumMipLevels - 1);
        plan.lo = nearest;
        plan.hi = nearest;
        plan.blend = 0.0f;
        plan.single_level = true;
        return plan;
    }

    plan.lo = static_cast<int>(level_f);
    plan.hi = std::min(plan.lo + 1, kNumMipLevels - 1);
    float frac = level_f - static_cast<float>(plan.lo);
    plan.blend = vivid_wavetable::interp::smoothstep01(frac);
    plan.single_level = (plan.lo == plan.hi) || (frac < 0.001f);
    return plan;
}

PreparedFramePlan Wavetable::prepare_frame_plan(float position) const {
    PreparedFramePlan plan{};
    if (data.empty() || frame_count == 0) return plan;
    position = std::clamp(position, 0.0f, 1.0f);
    float frame_pos = position * static_cast<float>(frame_count - 1);
    plan.f0 = static_cast<uint32_t>(frame_pos);
    plan.f1 = std::min(plan.f0 + 1, frame_count - 1);
    float frac = frame_pos - static_cast<float>(plan.f0);
    plan.blend = vivid_wavetable::interp::smoothstep01(frac);
    plan.single_frame = (plan.f0 == plan.f1) || (frac < 0.001f);
    return plan;
}

namespace {

inline float sample_level_prepared(const float* buf,
                                   const PreparedFramePlan& frame_plan,
                                   float phase) {
    if (!buf) return 0.0f;
    const float* d0 = buf + frame_plan.f0 * kSamplesPerFrame;
    float a = vivid_wavetable::interp::sample_periodic_catmull(d0, kSamplesPerFrame, phase);
    if (frame_plan.single_frame) return a;

    const float* d1 = buf + frame_plan.f1 * kSamplesPerFrame;
    float b = vivid_wavetable::interp::sample_periodic_catmull(d1, kSamplesPerFrame, phase);
    return vivid_wavetable::interp::lerp(a, b, frame_plan.blend);
}

} // namespace

float Wavetable::sample_level(float phase, float position, int level) const {
    PreparedFramePlan frame_plan = prepare_frame_plan(position);
    return sample_level_prepared(level_data(level), frame_plan, phase);
}

float Wavetable::sample_prepared(float phase,
                                 const PreparedFramePlan& frame_plan,
                                 const PreparedMipPlan& mip_plan) const {
    if (data.empty() || frame_count == 0) return 0.0f;

    float s_lo = sample_level_prepared(level_data(mip_plan.lo), frame_plan, phase);
    if (mip_plan.single_level) return s_lo;

    float s_hi = sample_level_prepared(level_data(mip_plan.hi), frame_plan, phase);
    return vivid_wavetable::interp::lerp(s_lo, s_hi, mip_plan.blend);
}

float Wavetable::sample(float phase, float position, float freq_hz, float sample_rate) const {
    PreparedFramePlan frame_plan = prepare_frame_plan(position);
    PreparedMipPlan mip_plan = prepare_mip_plan(freq_hz, sample_rate, false);
    return sample_prepared(phase, frame_plan, mip_plan);
}

Wavetable* load_wavetable_from_wav(const std::string& path) {
    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 1, 0);
    ma_decoder decoder;

    ma_result result = ma_decoder_init_file(path.c_str(), &config, &decoder);
    if (result != MA_SUCCESS) {
        std::fprintf(stderr, "[wavetable] Failed to decode wav: %s (error %d)\n",
                     path.c_str(), result);
        return nullptr;
    }

    ma_uint64 total_frames = 0;
    ma_decoder_get_length_in_pcm_frames(&decoder, &total_frames);
    if (total_frames == 0) {
        std::fprintf(stderr, "[wavetable] Empty or unreadable: %s\n", path.c_str());
        ma_decoder_uninit(&decoder);
        return nullptr;
    }

    std::vector<float> samples(static_cast<std::size_t>(total_frames));
    ma_uint64 frames_read = 0;
    ma_decoder_read_pcm_frames(&decoder, samples.data(), total_frames, &frames_read);
    ma_decoder_uninit(&decoder);

    uint32_t frame_count = static_cast<uint32_t>(frames_read) / kSamplesPerFrame;
    if (frame_count == 0) {
        std::fprintf(stderr, "[wavetable] Wav too short for even one frame (%llu samples): %s\n",
                     static_cast<unsigned long long>(frames_read), path.c_str());
        return nullptr;
    }
    if (frame_count > kMaxFrames) frame_count = kMaxFrames;

    float peak = 0.0f;
    uint32_t total_samples = frame_count * kSamplesPerFrame;
    for (uint32_t i = 0; i < total_samples; ++i) {
        float a = std::abs(samples[i]);
        if (a > peak) peak = a;
    }

    auto* wt = new Wavetable();
    wt->allocate(frame_count);

    float scale = (peak > 0.0001f) ? (1.0f / peak) : 1.0f;
    for (uint32_t i = 0; i < total_samples; ++i)
        wt->data[i] = samples[i] * scale;

    wt->build_mipmaps();

    std::fprintf(stderr, "[wavetable] Loaded custom wav: %s (%u frames, %llu samples)\n",
                 path.c_str(), frame_count, static_cast<unsigned long long>(frames_read));
    return wt;
}

void build_builtin_wavetables(Wavetable* tables, std::size_t count) {
    if (!tables || count < kBuiltinWavetableCount) return;

    for (int member = 0; member < kBuiltinMembersPerFamily; ++member) {
        generate_analog_family(tables[builtin_index(FAMILY_ANALOG_WARM, member)], member);
        generate_digital_family(tables[builtin_index(FAMILY_BRIGHT_DIGITAL, member)], member);
        generate_vocal_family(tables[builtin_index(FAMILY_VOCAL_FORMANT, member)], member);
        generate_metallic_family(tables[builtin_index(FAMILY_METALLIC, member)], member);
        generate_harmonic_family(tables[builtin_index(FAMILY_HARMONIC_SPECTRAL, member)], member);
        generate_texture_family(tables[builtin_index(FAMILY_TEXTURE_MOTION, member)], member);
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(kBuiltinWavetableCount); ++i)
        tables[i].build_mipmaps();
}

const Wavetable* resolve_builtin_wavetable(const Wavetable* tables, int family, int member) {
    if (!tables) return nullptr;
    family = std::clamp(family, 0, kBuiltinFamilyCount - 1);
    member = std::clamp(member, 0, kBuiltinMembersPerFamily - 1);
    return &tables[builtin_index(family, member)];
}

} // namespace vivid_wavetable::bank
