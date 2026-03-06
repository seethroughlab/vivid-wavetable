#include "../../src/wavetable_interp.h"
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
constexpr float kPi = 3.14159265358979323846f;

bool approx(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

float sample_linear_periodic(const float* data, uint32_t count, float phase) {
    phase = phase - std::floor(phase);
    float sp = phase * static_cast<float>(count);
    uint32_t i0 = static_cast<uint32_t>(sp) % count;
    uint32_t i1 = (i0 + 1) % count;
    float t = sp - std::floor(sp);
    return data[i0] + (data[i1] - data[i0]) * t;
}
} // namespace

int main() {
    using namespace vivid_wavetable::interp;

    if (!approx(smoothstep01(0.0f), 0.0f) || !approx(smoothstep01(1.0f), 1.0f)) {
        std::fprintf(stderr, "smoothstep endpoints failed\n");
        return 1;
    }
    if (!approx(smoothstep01(0.5f), 0.5f, 1e-6f)) {
        std::fprintf(stderr, "smoothstep midpoint failed\n");
        return 1;
    }

    if (!approx(catmull_rom(0.0f, 1.0f, 2.0f, 3.0f, 0.25f), 1.25f, 1e-6f)) {
        std::fprintf(stderr, "catmull linearity failed\n");
        return 1;
    }

    std::vector<float> sine(256);
    for (size_t i = 0; i < sine.size(); ++i) {
        float ph = static_cast<float>(i) / static_cast<float>(sine.size());
        sine[i] = std::sin(2.0f * kPi * ph);
    }

    // Periodic wrap consistency: phase 0 and 1 must match.
    float at0 = sample_periodic_catmull(sine.data(), static_cast<uint32_t>(sine.size()), 0.0f);
    float at1 = sample_periodic_catmull(sine.data(), static_cast<uint32_t>(sine.size()), 1.0f);
    if (!approx(at0, at1, 1e-6f)) {
        std::fprintf(stderr, "periodic wrap consistency failed\n");
        return 1;
    }

    // Quality check: Catmull should better approximate a smooth curve than linear interpolation.
    double err_linear = 0.0;
    double err_catmull = 0.0;
    const int samples = 2048;
    for (int i = 0; i < samples; ++i) {
        float phase = (static_cast<float>(i) + 0.37f) / static_cast<float>(samples);
        float truth = std::sin(2.0f * kPi * phase);
        float a = sample_linear_periodic(sine.data(), static_cast<uint32_t>(sine.size()), phase);
        float b = sample_periodic_catmull(sine.data(), static_cast<uint32_t>(sine.size()), phase);
        double da = static_cast<double>(a - truth);
        double db = static_cast<double>(b - truth);
        err_linear += da * da;
        err_catmull += db * db;
    }
    if (!(err_catmull < err_linear)) {
        std::fprintf(stderr, "catmull quality regression (catmull mse >= linear mse)\n");
        return 1;
    }

    std::printf("wavetable interpolation tests passed\n");
    return 0;
}
