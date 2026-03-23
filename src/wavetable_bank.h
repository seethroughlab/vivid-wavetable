#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vivid_wavetable::bank {

inline constexpr uint32_t kSamplesPerFrame = 2048;
inline constexpr uint32_t kMaxFrames = 256;
inline constexpr int kNumMipLevels = 11;
inline constexpr int kBuiltinWavetableCount = 9;

struct Wavetable {
    std::vector<float> data;
    std::vector<float> mip[kNumMipLevels - 1];
    uint32_t frame_count = 0;

    void allocate(uint32_t frames);
    float* frame_ptr(uint32_t frame_index);
    void build_mipmaps();
    float sample_level(float phase, float position, int level) const;
    float sample(float phase, float position, float freq_hz, float sample_rate) const;
};

Wavetable* load_wavetable_from_wav(const std::string& path);
void build_builtin_wavetables(Wavetable* tables, std::size_t count);

} // namespace vivid_wavetable::bank
