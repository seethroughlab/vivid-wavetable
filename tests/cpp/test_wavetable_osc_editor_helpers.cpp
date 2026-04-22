// Pure-logic tests for wavetable_osc_editor_shared: family/member
// naming, waveform sampling, unison layout math, family cell rects
// and hit-testing. These helpers power the editor's three regions
// (browser grid, preview polyline, unison scatter).

#include "wavetable_osc_editor_shared.h"
#include "wavetable_bank.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace ed = ::vivid_wavetable::editor;
namespace bank = ::vivid_wavetable::bank;

static int failures = 0;

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "  FAIL: %s\n", msg);
        failures++;
    } else {
        std::fprintf(stderr, "  PASS: %s\n", msg);
    }
}

int main() {
    std::fprintf(stderr, "=== Test: WavetableOsc editor helpers ===\n\n");

    // --- Family / member naming: canonical labels + out-of-range guard ---
    {
        check(std::strcmp(ed::family_short_name(0), "Analog") == 0,
              "family_short_name(0) == Analog");
        check(std::strcmp(ed::family_long_name(0), "AnalogWarm") == 0,
              "family_long_name(0) == AnalogWarm");
        check(std::strcmp(ed::family_long_name(5), "TextureMotion") == 0,
              "family_long_name(5) == TextureMotion");
        check(std::strcmp(ed::family_short_name(-1), "?") == 0,
              "family_short_name(-1) guards to ?");
        check(std::strcmp(ed::family_short_name(99), "?") == 0,
              "family_short_name(99) guards to ?");
        check(std::strcmp(ed::member_name(0), "Core") == 0,
              "member_name(0) == Core");
        check(std::strcmp(ed::member_name(7), "Air") == 0,
              "member_name(7) == Air");
        check(std::strcmp(ed::member_name(99), "?") == 0,
              "member_name(99) guards to ?");
    }

    // --- Param / output index sentinels match collect_params() order ---
    {
        check(ed::kWavetableSourceIndex  == 0,
              "kWavetableSourceIndex  == 0");
        check(ed::kWavetableFamilyIndex  == 1,
              "kWavetableFamilyIndex  == 1");
        check(ed::kWavetableMemberIndex  == 2,
              "kWavetableMemberIndex  == 2");
        check(ed::kPositionIndex         == 4,
              "kPositionIndex         == 4");
        check(ed::kAmplitudeIndex        == 5,
              "kAmplitudeIndex        == 5");
        check(ed::kWarpModeIndex         == 6,
              "kWarpModeIndex         == 6");
        check(ed::kUnisonVoicesIndex     == 16,
              "kUnisonVoicesIndex     == 16");
        check(ed::kUnisonSpreadModeIndex == 19,
              "kUnisonSpreadModeIndex == 19");
    }

    // --- Waveform sampling: null / empty table → zeros, no crash ---
    {
        float samples[64] = {};
        for (int i = 0; i < 64; ++i) samples[i] = 12.5f;
        ed::sample_waveform_polyline(nullptr, 0.5f, samples, 64);
        bool all_zero = true;
        for (int i = 0; i < 64; ++i) if (samples[i] != 0.0f) all_zero = false;
        check(all_zero, "null table fills with zeros");

        bank::Wavetable empty{};  // frame_count = 0
        empty.frame_count = 0;
        for (int i = 0; i < 64; ++i) samples[i] = 7.0f;
        ed::sample_waveform_polyline(&empty, 0.5f, samples, 64);
        all_zero = true;
        for (int i = 0; i < 64; ++i) if (samples[i] != 0.0f) all_zero = false;
        check(all_zero, "frame_count=0 table fills with zeros");
    }

    // --- Waveform sampling: real builtin produces non-trivial output ---
    {
        static std::array<bank::Wavetable, bank::kBuiltinWavetableCount> tables{};
        bank::build_builtin_wavetables(tables.data(), tables.size());
        const bank::Wavetable* t = &tables[0];
        check(t->frame_count > 0, "builtin table has frames");

        float samples[128] = {};
        ed::sample_waveform_polyline(t, 0.0f, samples, 128);
        // Most builtins are zero-DC waveforms — verify the samples
        // cover both signs (a meaningful polyline, not just zeros).
        bool has_pos = false, has_neg = false;
        for (int i = 0; i < 128; ++i) {
            if (samples[i] > 1e-3f)  has_pos = true;
            if (samples[i] < -1e-3f) has_neg = true;
        }
        check(has_pos && has_neg,
              "builtin polyline spans both signs");
    }

    // --- Unison layout: single voice is centered ---
    {
        ed::VoicePoint v[16] = {};
        ed::compute_unison_layout(1, 20.0f, 1.0f, 0, v);
        check(std::fabs(v[0].detune_cents) < 1e-4f,
              "unison=1: detune is 0");
        check(std::fabs(v[0].pan) < 1e-4f,
              "unison=1: pan is centered");
    }

    // --- Unison layout: symmetric spread for an even voice count ---
    {
        ed::VoicePoint v[16] = {};
        ed::compute_unison_layout(4, 50.0f, 1.0f, 0, v);
        // Linear mode → voices span [-spread, +spread].
        check(v[0].detune_cents < 0.0f && v[3].detune_cents > 0.0f,
              "unison=4: endpoints are on opposite signs");
        check(std::fabs(v[0].detune_cents + v[3].detune_cents) < 1e-3f,
              "unison=4: linear spread is symmetric about 0");
        check(std::fabs(v[1].detune_cents + v[2].detune_cents) < 1e-3f,
              "unison=4: middle voices are symmetric about 0");
    }

    // --- Unison layout: voice_count=0 is clamped, no out-of-bounds ---
    {
        ed::VoicePoint v[16] = {};
        for (int i = 0; i < 16; ++i) v[i] = {999.0f, 999.0f};
        ed::compute_unison_layout(0, 50.0f, 1.0f, 0, v);
        // clamp(0, 1, 16) = 1 → only slot 0 is written.
        check(std::fabs(v[0].detune_cents) < 1e-4f,
              "unison=0 clamps to 1 and writes slot 0");
    }

    // --- Cell rect layout: grid is laminar (columns add up correctly) ---
    {
        auto r00 = ed::family_cell_rect(10.0f, 20.0f, 30.0f, 20.0f, 0, 0);
        auto r21 = ed::family_cell_rect(10.0f, 20.0f, 30.0f, 20.0f, 2, 1);
        check(r00.x == 10.0f && r00.y == 20.0f,
              "family_cell_rect(0,0) origin");
        check(r00.w == 30.0f && r00.h == 20.0f,
              "family_cell_rect(0,0) size");
        check(r21.x == 10.0f + 2 * 30.0f,
              "family_cell_rect(2,1) x offset");
        check(r21.y == 20.0f + 1 * 20.0f,
              "family_cell_rect(2,1) y offset");
    }

    // --- Cell hit-test: reverse of family_cell_rect for legal points ---
    {
        // Grid: origin (10, 20), cell 30×20.
        auto h = ed::family_cell_from_point(10.0f, 20.0f, 30.0f, 20.0f,
                                            /*mx=*/25.0f, /*my=*/30.0f);
        check(h.family == 0 && h.member == 0,
              "hit-test inside (0,0) cell");

        auto h2 = ed::family_cell_from_point(10.0f, 20.0f, 30.0f, 20.0f,
                                             /*mx=*/75.0f, /*my=*/50.0f);
        // x=75 → col = (75-10)/30 = 2; y=50 → row = (50-20)/20 = 1.
        check(h2.family == 2 && h2.member == 1,
              "hit-test inside (2,1) cell");

        // Out of range on either axis returns {-1,-1}.
        auto h3 = ed::family_cell_from_point(10.0f, 20.0f, 30.0f, 20.0f,
                                             /*mx=*/5.0f, /*my=*/30.0f);
        check(h3.family == -1 && h3.member == -1,
              "hit-test to the left of the grid → no hit");
        auto h4 = ed::family_cell_from_point(10.0f, 20.0f, 30.0f, 20.0f,
                                             /*mx=*/25.0f, /*my=*/10.0f);
        check(h4.family == -1 && h4.member == -1,
              "hit-test above the grid → no hit");
        auto h5 = ed::family_cell_from_point(10.0f, 20.0f, 30.0f, 20.0f,
                                             /*mx=*/9999.0f, /*my=*/30.0f);
        check(h5.family == -1 && h5.member == -1,
              "hit-test far right of the grid → no hit (family clamps)");
    }

    // --- Cell hit-test with pathological cell size returns {-1,-1} ---
    {
        auto h = ed::family_cell_from_point(10.0f, 20.0f, 0.0f, 20.0f,
                                            25.0f, 30.0f);
        check(h.family == -1 && h.member == -1,
              "cell_w=0 → no hit");
        auto h2 = ed::family_cell_from_point(10.0f, 20.0f, 30.0f, 0.0f,
                                             25.0f, 30.0f);
        check(h2.family == -1 && h2.member == -1,
              "cell_h=0 → no hit");
    }

    std::fprintf(stderr, "\n=== %d failures ===\n", failures);
    return failures == 0 ? 0 : 1;
}
