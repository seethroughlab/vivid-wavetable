#include "wavetable_dsp.h"

#include <cstdio>
#include <cmath>

static int failures = 0;

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        failures++;
    } else {
        std::fprintf(stderr, "PASS: %s\n", msg);
    }
}

int main() {
    vivid_wavetable::dsp::MotionSmoother smoother;
    float coeff = 0.2f;
    float prev = smoother.process(0.0f, coeff);

    for (int i = 0; i < 16; ++i) {
        float current = smoother.process(1.0f, coeff);
        check(current >= prev, "smoother response is monotonic");
        check(current <= 1.0f, "smoother stays bounded by the target");
        prev = current;
    }

    check(std::fabs(prev - 1.0f) < 0.1f, "smoother converges toward the target");

    for (int i = 0; i < 16; ++i) {
        float current = smoother.process(0.0f, coeff);
        check(current <= prev, "smoother decays monotonically");
        check(current >= 0.0f, "smoother stays above the lower bound");
        prev = current;
    }

    return failures == 0 ? 0 : 1;
}
