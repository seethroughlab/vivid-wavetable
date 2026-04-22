#include "wavetable_osc_editor_shared.h"
#include "wavetable_voice_utils.h"

#include <algorithm>
#include <cmath>

namespace vivid_wavetable::editor {

const char* family_short_name(int family) {
    static const char* names[] = {
        "Analog", "Bright", "Vocal", "Metal", "Harmonic", "Texture",
    };
    if (family < 0 || family >= static_cast<int>(sizeof(names) / sizeof(names[0])))
        return "?";
    return names[family];
}

const char* family_long_name(int family) {
    static const char* names[] = {
        "AnalogWarm", "BrightDigital", "VocalFormant",
        "Metallic",   "HarmonicSpectral", "TextureMotion",
    };
    if (family < 0 || family >= static_cast<int>(sizeof(names) / sizeof(names[0])))
        return "?";
    return names[family];
}

const char* member_name(int member) {
    static const char* names[] = {
        "Core", "Soft", "Rich", "Hollow", "Sweep", "Glass", "Edge", "Air",
    };
    if (member < 0 || member >= static_cast<int>(sizeof(names) / sizeof(names[0])))
        return "?";
    return names[member];
}

void sample_waveform_polyline(const bank::Wavetable* table,
                              float position,
                              float* out,
                              int samples) {
    if (!out || samples <= 0) return;
    if (!table || table->frame_count == 0) {
        for (int i = 0; i < samples; ++i) out[i] = 0.0f;
        return;
    }
    position = std::clamp(position, 0.0f, 1.0f);
    for (int i = 0; i < samples; ++i) {
        const float phase = static_cast<float>(i) /
                            static_cast<float>(std::max(1, samples));
        out[i] = table->sample_level(phase, position, 0);
    }
}

void compute_unison_layout(int voice_count,
                           float spread_cents,
                           float stereo_depth,
                           int spread_mode,
                           VoicePoint* out) {
    if (!out) return;
    voice_count = std::clamp(voice_count, 1, 16);
    for (int i = 0; i < voice_count; ++i) {
        out[i].detune_cents = voice::unison_detune_offset(
            i, voice_count, spread_cents, spread_mode, /*lane_seed=*/0);
        out[i].pan = voice::unison_pan_position(
            i, voice_count, stereo_depth);
    }
}

CellRect family_cell_rect(float origin_x, float origin_y,
                          float cell_w, float cell_h,
                          int family, int member) {
    CellRect r{};
    r.x = origin_x + static_cast<float>(family) * cell_w;
    r.y = origin_y + static_cast<float>(member) * cell_h;
    r.w = cell_w;
    r.h = cell_h;
    return r;
}

FamilyHit family_cell_from_point(float origin_x, float origin_y,
                                 float cell_w, float cell_h,
                                 float mx, float my) {
    FamilyHit h{};
    if (cell_w <= 0.0f || cell_h <= 0.0f) return h;
    if (mx < origin_x || my < origin_y) return h;
    const int fam = static_cast<int>((mx - origin_x) / cell_w);
    const int mem = static_cast<int>((my - origin_y) / cell_h);
    if (fam < 0 || fam >= bank::kBuiltinFamilyCount) return h;
    if (mem < 0 || mem >= bank::kBuiltinMembersPerFamily) return h;
    h.family = fam;
    h.member = mem;
    return h;
}

} // namespace vivid_wavetable::editor
