// Exports selected builtin wavetables to mono 16-bit WAV files.
// Build:  clang++ -std=c++17 -I../src scripts/export_factory_wavetables.cpp ../src/wavetable_bank.cpp -o export_factory_wavetables
// Run:    ./export_factory_wavetables

#include "wavetable_bank.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace vivid_wavetable::bank;

struct WavExportEntry {
    const char* filename;
    int family;
    int member;
};

static const WavExportEntry kExports[] = {
    {"assets/wavetables/warm-keys-core.wav",    FAMILY_ANALOG_WARM,        MEMBER_CORE},
    {"assets/wavetables/vocal-pad-sweep.wav",   FAMILY_VOCAL_FORMANT,      MEMBER_SWEEP},
    {"assets/wavetables/glass-motion.wav",       FAMILY_METALLIC,           MEMBER_GLASS},
    {"assets/wavetables/rooted-bass-edge.wav",   FAMILY_ANALOG_WARM,        MEMBER_EDGE},
    {"assets/wavetables/texture-tide.wav",       FAMILY_TEXTURE_MOTION,     MEMBER_CORE},
    {"assets/wavetables/bright-pluck-edge.wav",  FAMILY_BRIGHT_DIGITAL,     MEMBER_EDGE},
};

static bool write_wav(const char* path, const float* samples, uint32_t total_samples) {
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::fprintf(stderr, "Failed to create %s\n", path);
        return false;
    }

    uint16_t channels = 1;
    uint32_t sample_rate = 48000;
    uint16_t bits = 16;
    uint16_t block_align = channels * (bits / 8);
    uint32_t byte_rate = sample_rate * block_align;
    uint32_t data_size = total_samples * block_align;
    uint32_t riff_size = 36 + data_size;

    // RIFF header
    std::fwrite("RIFF", 1, 4, f);
    std::fwrite(&riff_size, 4, 1, f);
    std::fwrite("WAVE", 1, 4, f);

    // fmt chunk
    std::fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    uint16_t audio_format = 1; // PCM
    std::fwrite(&fmt_size, 4, 1, f);
    std::fwrite(&audio_format, 2, 1, f);
    std::fwrite(&channels, 2, 1, f);
    std::fwrite(&sample_rate, 4, 1, f);
    std::fwrite(&byte_rate, 4, 1, f);
    std::fwrite(&block_align, 2, 1, f);
    std::fwrite(&bits, 2, 1, f);

    // data chunk
    std::fwrite("data", 1, 4, f);
    std::fwrite(&data_size, 4, 1, f);

    // Convert float -> int16
    for (uint32_t i = 0; i < total_samples; ++i) {
        float s = std::clamp(samples[i], -1.0f, 1.0f);
        int16_t pcm = static_cast<int16_t>(s * 32767.0f);
        std::fwrite(&pcm, 2, 1, f);
    }

    std::fclose(f);
    return true;
}

int main() {
    Wavetable tables[kBuiltinWavetableCount];
    build_builtin_wavetables(tables, kBuiltinWavetableCount);

    int failures = 0;

    for (const auto& entry : kExports) {
        const Wavetable* wt = resolve_builtin_wavetable(tables, entry.family, entry.member);
        if (!wt || wt->frame_count == 0) {
            std::fprintf(stderr, "Failed to resolve builtin family=%d member=%d\n",
                         entry.family, entry.member);
            ++failures;
            continue;
        }

        uint32_t total_samples = wt->frame_count * kSamplesPerFrame;
        if (!write_wav(entry.filename, wt->data.data(), total_samples)) {
            ++failures;
            continue;
        }

        std::fprintf(stderr, "Exported %s (%u frames, %u samples)\n",
                     entry.filename, wt->frame_count, total_samples);
    }

    if (failures > 0) {
        std::fprintf(stderr, "FAILED: %d exports failed\n", failures);
        return 1;
    }

    std::fprintf(stderr, "All %zu factory wavetables exported successfully.\n",
                 sizeof(kExports) / sizeof(kExports[0]));
    return 0;
}
