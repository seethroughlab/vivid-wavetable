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

static void generate_basic(Wavetable& wt) {
    wt.allocate(32);
    for (uint32_t fr = 0; fr < wt.frame_count; ++fr) {
        float* d = wt.frame_ptr(fr);
        float t = static_cast<float>(fr) / 31.0f;
        for (uint32_t i = 0; i < kSamplesPerFrame; ++i) {
            float p = static_cast<float>(i) / static_cast<float>(kSamplesPerFrame);
            float sine     = std::sin(p * kTwoPi);
            float triangle = 4.0f * std::abs(p - 0.5f) - 1.0f;
            float saw      = 2.0f * p - 1.0f;
            float square   = p < 0.5f ? 1.0f : -1.0f;
            float s;
            if (t < 0.333f) {
                float b = t / 0.333f;
                s = sine * (1.0f - b) + triangle * b;
            } else if (t < 0.666f) {
                float b = (t - 0.333f) / 0.333f;
                s = triangle * (1.0f - b) + saw * b;
            } else {
                float b = (t - 0.666f) / 0.334f;
                s = saw * (1.0f - b) + square * b;
            }
            d[i] = s;
        }
    }
}

static void generate_analog(Wavetable& wt) {
    wt.allocate(32);
    for (uint32_t fr = 0; fr < wt.frame_count; ++fr) {
        float* d = wt.frame_ptr(fr);
        float t = static_cast<float>(fr) / 31.0f;
        for (uint32_t i = 0; i < kSamplesPerFrame; ++i) {
            float p = static_cast<float>(i) / static_cast<float>(kSamplesPerFrame);
            float sample = 0.0f;
            int nh = 3 + static_cast<int>(t * 12);
            for (int h = 1; h <= nh; ++h) {
                float amp = 1.0f / static_cast<float>(h);
                if (h % 2 == 1) amp *= 1.2f;
                float drift = std::sin(static_cast<float>(fr * h) * 0.1f) * 0.02f;
                sample += amp * std::sin((p + drift) * kTwoPi * static_cast<float>(h));
            }
            d[i] = std::tanh(sample * 0.8f);
        }
    }
}

static void generate_digital(Wavetable& wt) {
    wt.allocate(32);
    for (uint32_t fr = 0; fr < wt.frame_count; ++fr) {
        float* d = wt.frame_ptr(fr);
        float t = static_cast<float>(fr) / 31.0f;
        float mod_index = t * 8.0f;
        float ratio = 1.0f + std::floor(t * 4.0f);
        for (uint32_t i = 0; i < kSamplesPerFrame; ++i) {
            float p = static_cast<float>(i) / static_cast<float>(kSamplesPerFrame);
            float mod = std::sin(p * kTwoPi * ratio);
            d[i] = std::sin(p * kTwoPi + mod * mod_index);
        }
    }
}

static void generate_vocal(Wavetable& wt) {
    wt.allocate(32);
    const float formants[5][3] = {
        {800.0f, 1150.0f, 2800.0f},
        {400.0f, 2000.0f, 2550.0f},
        {350.0f, 2700.0f, 2900.0f},
        {450.0f, 800.0f,  2830.0f},
        {325.0f, 700.0f,  2530.0f}
    };
    const float amps[5][3] = {
        {1.0f, 0.6f, 0.2f},
        {1.0f, 0.4f, 0.3f},
        {1.0f, 0.2f, 0.3f},
        {1.0f, 0.8f, 0.1f},
        {1.0f, 0.8f, 0.1f}
    };
    for (uint32_t fr = 0; fr < wt.frame_count; ++fr) {
        float* d = wt.frame_ptr(fr);
        float t = static_cast<float>(fr) / 31.0f;
        float vowel_pos = t * 4.0f;
        int v0 = static_cast<int>(vowel_pos);
        int v1 = std::min(v0 + 1, 4);
        float blend = vowel_pos - static_cast<float>(v0);
        v0 = std::min(v0, 4);

        float blended_formants[3];
        float blended_amps[3];
        for (int f = 0; f < 3; ++f) {
            blended_formants[f] = formants[v0][f] * (1.0f - blend) + formants[v1][f] * blend;
            blended_amps[f]     = amps[v0][f]     * (1.0f - blend) + amps[v1][f]     * blend;
        }

        for (uint32_t i = 0; i < kSamplesPerFrame; ++i) {
            float p = static_cast<float>(i) / static_cast<float>(kSamplesPerFrame);
            float sample = 0.0f;
            float fundamental = 120.0f;
            for (int h = 1; h <= 40; ++h) {
                float freq = fundamental * static_cast<float>(h);
                float amp = 0.0f;
                for (int f = 0; f < 3; ++f) {
                    float bw = 80.0f + static_cast<float>(f) * 40.0f;
                    float dist = (freq - blended_formants[f]) / bw;
                    amp += blended_amps[f] * std::exp(-dist * dist * 0.5f);
                }
                sample += amp * std::sin(p * kTwoPi * static_cast<float>(h));
            }
            d[i] = std::tanh(sample * 0.3f);
        }
    }
}

static void generate_texture(Wavetable& wt) {
    wt.allocate(32);
    uint32_t seed = 12345;
    auto rand_f = [&seed]() -> float {
        seed = seed * 1103515245 + 12345;
        return (static_cast<float>(seed & 0x7FFFFFFF) /
                static_cast<float>(0x7FFFFFFF)) * 2.0f - 1.0f;
    };
    for (uint32_t fr = 0; fr < wt.frame_count; ++fr) {
        float* d = wt.frame_ptr(fr);
        float t = static_cast<float>(fr) / 31.0f;
        for (uint32_t i = 0; i < kSamplesPerFrame; ++i) {
            float p = static_cast<float>(i) / static_cast<float>(kSamplesPerFrame);
            float harm = std::sin(p * kTwoPi)
                       + 0.5f * std::sin(p * kTwoPi * 2.0f)
                       + 0.25f * std::sin(p * kTwoPi * 3.0f);
            harm *= 0.5f;
            d[i] = harm * (1.0f - t) + rand_f() * t;
        }
        for (int pass = 0; pass < 3; ++pass) {
            for (uint32_t i = 1; i < kSamplesPerFrame - 1; ++i)
                d[i] = d[i] * 0.5f + (d[i - 1] + d[i + 1]) * 0.25f;
        }
    }
}

static void generate_pwm(Wavetable& wt) {
    wt.allocate(32);
    for (uint32_t fr = 0; fr < wt.frame_count; ++fr) {
        float* d = wt.frame_ptr(fr);
        float t = static_cast<float>(fr) / 31.0f;
        float pw = 0.1f + t * 0.8f;
        for (uint32_t i = 0; i < kSamplesPerFrame; ++i) {
            float p = static_cast<float>(i) / static_cast<float>(kSamplesPerFrame);
            d[i] = p < pw ? 1.0f : -1.0f;
        }
        for (int pass = 0; pass < 2; ++pass) {
            float prev = d[kSamplesPerFrame - 1];
            for (uint32_t i = 0; i < kSamplesPerFrame; ++i) {
                float next = d[(i + 1) % kSamplesPerFrame];
                float smoothed = d[i] * 0.7f + (prev + next) * 0.15f;
                prev = d[i];
                d[i] = smoothed;
            }
        }
    }
}

static void generate_formant(Wavetable& wt) {
    wt.allocate(64);
    const float formants[8][4] = {
        { 730.0f, 1090.0f, 2440.0f, 3400.0f},
        { 660.0f, 1720.0f, 2410.0f, 3400.0f},
        { 270.0f, 2290.0f, 3010.0f, 3400.0f},
        { 570.0f,  840.0f, 2410.0f, 3400.0f},
        { 300.0f,  870.0f, 2240.0f, 3400.0f},
        { 860.0f, 1550.0f, 2500.0f, 3400.0f},
        { 450.0f, 1500.0f, 2500.0f, 3400.0f},
        { 480.0f, 1270.0f, 2130.0f, 3320.0f}
    };
    const float form_amps[8][4] = {
        {1.0f, 0.6f, 0.2f, 0.1f},
        {1.0f, 0.4f, 0.3f, 0.1f},
        {1.0f, 0.2f, 0.3f, 0.1f},
        {1.0f, 0.8f, 0.1f, 0.05f},
        {1.0f, 0.8f, 0.1f, 0.05f},
        {1.0f, 0.5f, 0.25f, 0.1f},
        {1.0f, 0.6f, 0.2f, 0.08f},
        {1.0f, 0.5f, 0.3f, 0.12f}
    };
    for (uint32_t fr = 0; fr < wt.frame_count; ++fr) {
        float* d = wt.frame_ptr(fr);
        float t = static_cast<float>(fr) / 63.0f;
        float vowel_pos = t * 8.0f;
        int v0 = static_cast<int>(vowel_pos) % 8;
        int v1 = (v0 + 1) % 8;
        float blend = vowel_pos - std::floor(vowel_pos);

        float bf[4], ba[4];
        for (int f = 0; f < 4; ++f) {
            bf[f] = formants[v0][f] * (1.0f - blend) + formants[v1][f] * blend;
            ba[f] = form_amps[v0][f] * (1.0f - blend) + form_amps[v1][f] * blend;
        }

        for (uint32_t i = 0; i < kSamplesPerFrame; ++i) {
            float p = static_cast<float>(i) / static_cast<float>(kSamplesPerFrame);
            float sample = 0.0f;
            float fundamental = 120.0f;
            for (int h = 1; h <= 64; ++h) {
                float freq = fundamental * static_cast<float>(h);
                float amp = 0.0f;
                for (int f = 0; f < 4; ++f) {
                    float bw = 80.0f + static_cast<float>(f) * 40.0f;
                    float dist = (freq - bf[f]) / bw;
                    amp += ba[f] * std::exp(-dist * dist * 0.5f);
                }
                sample += amp * std::sin(p * kTwoPi * static_cast<float>(h));
            }
            d[i] = std::tanh(sample * 0.3f);
        }
    }
}

static void generate_harmonic(Wavetable& wt) {
    wt.allocate(64);
    for (uint32_t fr = 0; fr < wt.frame_count; ++fr) {
        float* d = wt.frame_ptr(fr);
        for (uint32_t i = 0; i < kSamplesPerFrame; ++i) {
            float p = static_cast<float>(i) / static_cast<float>(kSamplesPerFrame);
            float sample = 0.0f;

            if (fr < 16) {
                float t_local = static_cast<float>(fr) / 15.0f;
                int num_partials = 1 + static_cast<int>(t_local * 63.0f);
                for (int h = 1; h <= num_partials; ++h) {
                    float amp = 1.0f / static_cast<float>(h);
                    sample += amp * std::sin(p * kTwoPi * static_cast<float>(h));
                }
            } else if (fr < 32) {
                float t_local = static_cast<float>(fr - 16) / 15.0f;
                float even_weight = 1.0f - t_local;
                for (int h = 1; h <= 64; ++h) {
                    float amp = 1.0f / static_cast<float>(h);
                    if (h % 2 == 0) amp *= even_weight;
                    sample += amp * std::sin(p * kTwoPi * static_cast<float>(h));
                }
            } else if (fr < 48) {
                float t_local = static_cast<float>(fr - 32) / 15.0f;
                float even_weight = t_local;
                float odd_weight = 1.0f - 0.5f * t_local;
                for (int h = 1; h <= 64; ++h) {
                    float amp = 1.0f / static_cast<float>(h);
                    if (h % 2 == 0)
                        amp *= even_weight;
                    else
                        amp *= odd_weight;
                    sample += amp * std::sin(p * kTwoPi * static_cast<float>(h));
                }
            } else {
                float t_local = static_cast<float>(fr - 48) / 15.0f;
                float tilt = 2.0f - t_local * 1.7f;
                for (int h = 1; h <= 64; ++h) {
                    float amp = 1.0f / std::pow(static_cast<float>(h), tilt);
                    sample += amp * std::sin(p * kTwoPi * static_cast<float>(h));
                }
            }

            d[i] = sample;
        }

        float peak = 0.0f;
        for (uint32_t i = 0; i < kSamplesPerFrame; ++i)
            peak = std::max(peak, std::abs(d[i]));
        if (peak > 0.0f) {
            float inv = 1.0f / peak;
            for (uint32_t i = 0; i < kSamplesPerFrame; ++i)
                d[i] *= inv;
        }
    }
}

static void generate_metallic(Wavetable& wt) {
    wt.allocate(32);

    const int region_counts[4] = {5, 6, 7, 8};
    const float region_ratios[4][8] = {
        {1.0f, 2.0f,  3.0f,  4.2f,  5.4f,  0.0f,  0.0f,  0.0f},
        {1.0f, 1.5f,  2.3f,  3.1f,  4.7f,  6.2f,  0.0f,  0.0f},
        {1.0f, 1.19f, 1.56f, 2.0f,  2.44f, 2.83f, 3.15f, 0.0f},
        {1.0f, 1.34f, 1.87f, 2.15f, 2.58f, 3.24f, 3.81f, 4.53f}
    };
    const float region_amps[4][8] = {
        {1.0f, 0.5f,  0.3f,  0.25f, 0.2f,  0.0f,  0.0f,  0.0f},
        {1.0f, 0.6f,  0.4f,  0.3f,  0.2f,  0.15f, 0.0f,  0.0f},
        {1.0f, 0.7f,  0.5f,  0.4f,  0.3f,  0.2f,  0.15f, 0.0f},
        {1.0f, 0.8f,  0.6f,  0.5f,  0.4f,  0.3f,  0.25f, 0.2f}
    };

    for (uint32_t fr = 0; fr < wt.frame_count; ++fr) {
        float* d = wt.frame_ptr(fr);
        int region = static_cast<int>(fr / 8);
        if (region > 3) region = 3;
        int next_region = std::min(region + 1, 3);
        float t_local = static_cast<float>(fr % 8) / 7.0f;
        int max_partials = std::max(region_counts[region], region_counts[next_region]);

        for (uint32_t i = 0; i < kSamplesPerFrame; ++i) {
            float p = static_cast<float>(i) / static_cast<float>(kSamplesPerFrame);
            float sample = 0.0f;
            for (int h = 0; h < max_partials; ++h) {
                float r0 = (h < region_counts[region])      ? region_ratios[region][h]      : 0.0f;
                float r1 = (h < region_counts[next_region]) ? region_ratios[next_region][h] : 0.0f;
                float a0 = (h < region_counts[region])      ? region_amps[region][h]        : 0.0f;
                float a1 = (h < region_counts[next_region]) ? region_amps[next_region][h]   : 0.0f;

                float ratio = r0 * (1.0f - t_local) + r1 * t_local;
                float amp = a0 * (1.0f - t_local) + a1 * t_local;
                if (amp > 0.0f && ratio > 0.0f)
                    sample += amp * std::sin(p * kTwoPi * ratio);
            }
            d[i] = sample;
        }

        float peak = 0.0f;
        for (uint32_t i = 0; i < kSamplesPerFrame; ++i)
            peak = std::max(peak, std::abs(d[i]));
        if (peak > 0.0f) {
            float inv = 1.0f / peak;
            for (uint32_t i = 0; i < kSamplesPerFrame; ++i)
                d[i] *= inv;
        }
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

float Wavetable::sample_level(float phase, float position, int level) const {
    level = std::clamp(level, 0, kNumMipLevels - 1);
    const float* buf = (level == 0) ? data.data() : mip[level - 1].data();
    if (!buf) return 0.0f;

    position = std::clamp(position, 0.0f, 1.0f);
    float frame_pos = position * static_cast<float>(frame_count - 1);
    uint32_t f0 = static_cast<uint32_t>(frame_pos);
    uint32_t f1 = std::min(f0 + 1, frame_count - 1);
    float ff = frame_pos - static_cast<float>(f0);

    const float* d0 = buf + f0 * kSamplesPerFrame;
    const float* d1 = buf + f1 * kSamplesPerFrame;

    float a = vivid_wavetable::interp::sample_periodic_catmull(d0, kSamplesPerFrame, phase);
    float b = vivid_wavetable::interp::sample_periodic_catmull(d1, kSamplesPerFrame, phase);
    float frame_blend = vivid_wavetable::interp::smoothstep01(ff);
    return vivid_wavetable::interp::lerp(a, b, frame_blend);
}

float Wavetable::sample(float phase, float position, float freq_hz, float sample_rate) const {
    if (data.empty() || frame_count == 0) return 0.0f;
    if (!std::isfinite(freq_hz) || freq_hz <= 0.0f) freq_hz = 1.0f;

    float max_h = sample_rate / (2.0f * freq_hz);
    float level_f = std::log2(static_cast<float>(kSamplesPerFrame / 2) / std::max(max_h, 1.0f));
    if (!(level_f >= 0.0f)) level_f = 0.0f;
    if (!(level_f <= static_cast<float>(kNumMipLevels - 1)))
        level_f = static_cast<float>(kNumMipLevels - 1);

    int lo = static_cast<int>(level_f);
    int hi = std::min(lo + 1, kNumMipLevels - 1);
    float frac = level_f - static_cast<float>(lo);

    float s_lo = sample_level(phase, position, lo);
    if (frac < 0.001f) return s_lo;

    float s_hi = sample_level(phase, position, hi);
    float mip_blend = vivid_wavetable::interp::smoothstep01(frac);
    return vivid_wavetable::interp::lerp(s_lo, s_hi, mip_blend);
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

    generate_basic(tables[0]);
    generate_analog(tables[1]);
    generate_digital(tables[2]);
    generate_vocal(tables[3]);
    generate_texture(tables[4]);
    generate_pwm(tables[5]);
    generate_formant(tables[6]);
    generate_harmonic(tables[7]);
    generate_metallic(tables[8]);

    for (std::size_t i = 0; i < kBuiltinWavetableCount; ++i)
        tables[i].build_mipmaps();
}

} // namespace vivid_wavetable::bank
