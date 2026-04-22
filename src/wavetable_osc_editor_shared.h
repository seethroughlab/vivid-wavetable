#pragma once
// Pure-logic helpers for the WavetableOsc editor. Pulls the waveform
// sampling, unison-scatter voice layout, and family/member decoding
// into one test-friendly surface so the editor.cpp stays thin.

#include "wavetable_bank.h"

#include <cstddef>
#include <cstdint>

namespace vivid_wavetable::editor {

// --- Param / output indices -----------------------------------------------
//
// Must stay in sync with WavetableOsc::collect_params(). Declaration
// order drives indices; changes there must update these here.
inline constexpr int kWavetableSourceIndex   = 0;
inline constexpr int kWavetableFamilyIndex   = 1;
inline constexpr int kWavetableMemberIndex   = 2;
// wav_file (FilePath) at index 3 — string param.
inline constexpr int kPositionIndex          = 4;
inline constexpr int kAmplitudeIndex         = 5;
inline constexpr int kWarpModeIndex          = 6;
inline constexpr int kWarpAmountIndex        = 7;
inline constexpr int kUnisonVoicesIndex      = 16;
inline constexpr int kUnisonSpreadIndex      = 17;
inline constexpr int kUnisonStereoIndex      = 18;
inline constexpr int kUnisonSpreadModeIndex  = 19;

// --- Family / member decoding ---------------------------------------------

// Short and long display names. Indexed by enum BuiltinFamily / BuiltinMember.
const char* family_short_name(int family);   // "Analog", "Bright", "Vocal", ...
const char* family_long_name(int family);    // "AnalogWarm", "BrightDigital", ...
const char* member_name(int member);         // "Core", "Soft", "Rich", ...

// --- Waveform sampling ----------------------------------------------------

// Sample the wavetable at N phase points [0..1) at the given position,
// filling out[0..n-1] with amplitude values. Uses the highest-fidelity
// mip level (level 0). Handles null table and frame_count == 0 gracefully
// by zeroing the buffer. Caller sizes `out` to at least `samples`.
//
// This is what the preview polyline renders from.
void sample_waveform_polyline(const ::vivid_wavetable::bank::Wavetable* table,
                              float position,
                              float* out,
                              int samples);

// --- Unison voice layout --------------------------------------------------

// One voice's computed position on the detune × pan plane.
struct VoicePoint {
    float detune_cents;  // signed, spread away from 0
    float pan;           // -1 (hard left) .. +1 (hard right)
};

// Compute the positions of all `voice_count` unison voices given the
// spread/stereo/mode params. Mirrors the core's
// WavetableOsc::unison_detune_offset / unison_pan_position logic
// (seed = 0 for deterministic visualization). Fills out[0..voice_count-1].
void compute_unison_layout(int voice_count,
                           float spread_cents,
                           float stereo_depth,
                           int spread_mode,           // 0 Linear, 1 Exponential, 2 Random
                           VoicePoint* out);

// --- Family / member grid hit-test ----------------------------------------

// Compute a (family, member) cell rect inside a grid laid out as:
//   6 family columns across the top
//   8 member rows down the left
// cell_w / cell_h are the per-cell size; origin_x/origin_y is the
// top-left of the cell area (after the row-label gutter).
struct CellRect {
    float x, y, w, h;
};
CellRect family_cell_rect(float origin_x, float origin_y,
                          float cell_w, float cell_h,
                          int family, int member);

// Reverse of family_cell_rect. Returns {-1, -1} when the point isn't
// on a cell. `mx`, `my` are in the same coordinate space as origin_x/y.
struct FamilyHit { int family = -1; int member = -1; };
FamilyHit family_cell_from_point(float origin_x, float origin_y,
                                 float cell_w, float cell_h,
                                 float mx, float my);

} // namespace vivid_wavetable::editor
