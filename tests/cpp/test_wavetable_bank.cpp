#include "wavetable_bank.h"

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

static int failures = 0;

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        failures++;
    } else {
        std::fprintf(stderr, "PASS: %s\n", msg);
    }
}

static void append_u16(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xffu));
}

static void append_u32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xffu));
}

static std::filesystem::path write_test_wav() {
    namespace fs = std::filesystem;
    fs::path path = fs::temp_directory_path() / "vivid_wt_bank_test.wav";
    constexpr uint32_t sample_rate = 48000;
    constexpr uint16_t channels = 1;
    constexpr uint16_t bits_per_sample = 16;
    constexpr uint32_t sample_count =
        vivid_wavetable::bank::kSamplesPerFrame * 2;

    std::vector<int16_t> pcm(sample_count);
    for (uint32_t i = 0; i < sample_count; ++i) {
        float phase = static_cast<float>(i) / static_cast<float>(sample_rate);
        float sample = std::sin(phase * 440.0f * 6.28318530718f);
        pcm[i] = static_cast<int16_t>(std::round(sample * 32767.0f));
    }

    std::vector<uint8_t> wav;
    uint32_t data_bytes = static_cast<uint32_t>(pcm.size() * sizeof(int16_t));
    uint32_t riff_bytes = 36 + data_bytes;

    wav.insert(wav.end(), {'R', 'I', 'F', 'F'});
    append_u32(wav, riff_bytes);
    wav.insert(wav.end(), {'W', 'A', 'V', 'E'});
    wav.insert(wav.end(), {'f', 'm', 't', ' '});
    append_u32(wav, 16);
    append_u16(wav, 1);
    append_u16(wav, channels);
    append_u32(wav, sample_rate);
    append_u32(wav, sample_rate * channels * bits_per_sample / 8);
    append_u16(wav, channels * bits_per_sample / 8);
    append_u16(wav, bits_per_sample);
    wav.insert(wav.end(), {'d', 'a', 't', 'a'});
    append_u32(wav, data_bytes);

    const uint8_t* pcm_bytes = reinterpret_cast<const uint8_t*>(pcm.data());
    wav.insert(wav.end(), pcm_bytes, pcm_bytes + data_bytes);

    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(wav.data()), static_cast<std::streamsize>(wav.size()));
    file.close();
    return path;
}

int main() {
    using namespace vivid_wavetable::bank;

    std::vector<Wavetable> tables(kBuiltinWavetableCount);
    build_builtin_wavetables(tables.data(), tables.size());

    for (int family = 0; family < kBuiltinFamilyCount; ++family) {
        for (int member = 0; member < kBuiltinMembersPerFamily; ++member) {
            const Wavetable* wt = resolve_builtin_wavetable(tables.data(), family, member);
            check(wt != nullptr, "builtin lookup returns a table");
            if (!wt) continue;
            check(wt->frame_count > 0, "builtin table has frames");
            check(!wt->data.empty(), "builtin table has sample data");
            float a = wt->sample(0.15f, 0.2f, 110.0f, 48000.0f);
            float b = wt->sample(0.62f, 0.8f, 880.0f, 48000.0f);
            check(std::isfinite(a) && std::isfinite(b), "builtin samples stay finite");
        }
    }

    {
        const Wavetable* analog = resolve_builtin_wavetable(tables.data(), FAMILY_ANALOG_WARM, MEMBER_CORE);
        const Wavetable* metallic = resolve_builtin_wavetable(tables.data(), FAMILY_METALLIC, MEMBER_CORE);
        float a = analog ? analog->sample(0.31f, 0.5f, 220.0f, 48000.0f) : 0.0f;
        float b = metallic ? metallic->sample(0.31f, 0.5f, 220.0f, 48000.0f) : 0.0f;
        check(std::fabs(a - b) > 0.01f, "different families yield different sample values");
    }

    {
        std::filesystem::path wav_path = write_test_wav();
        Wavetable* custom = load_wavetable_from_wav(wav_path.string());
        check(custom != nullptr, "custom wav import succeeds");
        if (custom) {
            check(custom->frame_count >= 1, "custom wav import has frames");
            float s = custom->sample(0.25f, 0.0f, 220.0f, 48000.0f);
            check(std::isfinite(s), "custom wav sample stays finite");
            delete custom;
        }
        std::filesystem::remove(wav_path);
    }

    return failures == 0 ? 0 : 1;
}
