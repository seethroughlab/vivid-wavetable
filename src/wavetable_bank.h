#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vivid_wavetable::bank {

inline constexpr uint32_t kSamplesPerFrame = 2048;
inline constexpr uint32_t kMaxFrames = 256;
inline constexpr int kNumMipLevels = 11;
inline constexpr int kBuiltinFamilyCount = 6;
inline constexpr int kBuiltinMembersPerFamily = 8;
inline constexpr int kBuiltinWavetableCount = kBuiltinFamilyCount * kBuiltinMembersPerFamily;

enum BuiltinFamily {
    FAMILY_ANALOG_WARM = 0,
    FAMILY_BRIGHT_DIGITAL = 1,
    FAMILY_VOCAL_FORMANT = 2,
    FAMILY_METALLIC = 3,
    FAMILY_HARMONIC_SPECTRAL = 4,
    FAMILY_TEXTURE_MOTION = 5,
};

enum BuiltinMember {
    MEMBER_CORE = 0,
    MEMBER_SOFT = 1,
    MEMBER_RICH = 2,
    MEMBER_HOLLOW = 3,
    MEMBER_SWEEP = 4,
    MEMBER_GLASS = 5,
    MEMBER_EDGE = 6,
    MEMBER_AIR = 7,
};

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

constexpr int builtin_index(int family, int member) {
    return family * kBuiltinMembersPerFamily + member;
}

Wavetable* load_wavetable_from_wav(const std::string& path);
void build_builtin_wavetables(Wavetable* tables, std::size_t count);
const Wavetable* resolve_builtin_wavetable(const Wavetable* tables, int family, int member);

} // namespace vivid_wavetable::bank
