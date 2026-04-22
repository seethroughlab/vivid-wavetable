// Dedicated editor window for WavetableOsc. Three-region layout:
//   - Left (240px): family × member grid, 6 families × 8 members
//   - Center top: waveform preview polyline, drag to scrub position
//   - Center bottom: unison scatter (detune × pan plane)
//   - Right (220px): cursor readout + amplitude / warp sliders
//
// Uses the shared editor toolkit (editor_ui.h, editor_keys.h,
// draw_ui_helpers.h). No dependencies on operators/shared/editor_ui
// (which this sibling repo can't reach) — all geometry helpers live
// in wavetable_osc_editor_shared.{h,cpp}.

#include "wavetable_osc_internal.h"
#include "wavetable_osc_editor_shared.h"
#include "operator_api/draw_ui_helpers.h"
#include "operator_api/editor_keys.h"
#include "operator_api/thumbnail.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

namespace ed = ::vivid_wavetable::editor;
namespace bank = ::vivid_wavetable::bank;

constexpr float kInset      = 8.0f;
constexpr float kTopBarH    = 28.0f;
constexpr float kBrowserW   = 240.0f;
constexpr float kSidePanelW = 220.0f;

constexpr int   kPreviewSamples = 128;
constexpr int   kStackFrames    = 20;  // visible frames in the stacked preview

// Accent palette for the scatter-voice dots.
constexpr float kVoiceColor[3] = {0.88f, 0.65f, 0.25f};

} // namespace


VividEditorMetadata WavetableOsc::editor_metadata() {
    VividEditorMetadata m{};
    m.default_width  = 1200;
    m.default_height = 700;
    m.min_width      = 900;
    m.min_height     = 520;
    m.title_suffix   = "WavetableOsc Editor";
    return m;
}

void WavetableOsc::draw_editor(VividEditorContext* ctx) {
    if (!ctx) return;
    namespace ek = ::vivid::editor_keys;

    auto& d = ctx->draw;
    void* o = d.opaque;
    const auto& th = ctx->theme;

    auto get_param = [&](int idx, float fallback) -> float {
        if (idx < 0) return fallback;
        if (static_cast<uint32_t>(idx) >= ctx->param_count) return fallback;
        return ctx->param_values[idx];
    };
    auto set_named = [&](const char* name, float v) {
        if (ctx->commands.set_param)
            ctx->commands.set_param(ctx->commands.opaque, name, v);
    };

    // ---- Live state ----
    const int family = std::clamp(
        static_cast<int>(std::lround(get_param(ed::kWavetableFamilyIndex, 0.0f))),
        0, bank::kBuiltinFamilyCount - 1);
    const int member = std::clamp(
        static_cast<int>(std::lround(get_param(ed::kWavetableMemberIndex, 0.0f))),
        0, bank::kBuiltinMembersPerFamily - 1);
    const float position = std::clamp(
        get_param(ed::kPositionIndex, 0.0f), 0.0f, 1.0f);
    const float amplitude = std::clamp(
        get_param(ed::kAmplitudeIndex, 0.3f), 0.0f, 1.0f);
    const int warp_mode = std::clamp(
        static_cast<int>(std::lround(get_param(ed::kWarpModeIndex, 0.0f))), 0, 8);
    const float warp_amount = std::clamp(
        get_param(ed::kWarpAmountIndex, 0.0f), 0.0f, 1.0f);
    const int unison_voices = std::clamp(
        static_cast<int>(std::lround(get_param(ed::kUnisonVoicesIndex, 1.0f))), 1, 16);
    const float unison_spread = std::clamp(
        get_param(ed::kUnisonSpreadIndex, 20.0f), 0.0f, 100.0f);
    const float unison_stereo = std::clamp(
        get_param(ed::kUnisonStereoIndex, 1.0f), 0.0f, 1.0f);
    const int unison_spread_mode = std::clamp(
        static_cast<int>(std::lround(
            get_param(ed::kUnisonSpreadModeIndex, 0.0f))), 0, 2);

    // ---- Layout ----
    const float surf_w = ctx->surface_width;
    const float surf_h = ctx->surface_height;

    const float top_y = kInset;
    const float top_h = kTopBarH;

    const float body_y = top_y + top_h + kInset;
    const float body_h = std::max(0.0f, surf_h - body_y - kInset);

    const float browser_x = kInset;
    const float browser_w = kBrowserW;

    const float side_x = surf_w - kInset - kSidePanelW;
    const float side_w = kSidePanelW;

    const float preview_x = browser_x + browser_w + kInset;
    const float preview_w = std::max(0.0f, side_x - preview_x - kInset);
    const float preview_h = body_h * 0.6f;  // top 60% is waveform preview
    const float preview_y = body_y;

    const float scatter_x = preview_x;
    const float scatter_y = preview_y + preview_h + kInset;
    const float scatter_w = preview_w;
    const float scatter_h = std::max(0.0f, body_h - preview_h - kInset);

    // ---- Keyboard ----
    ctx->wants_keyboard = 1;
    for (uint32_t ei = 0; ei < ctx->event_count; ++ei) {
        const auto& e = ctx->events[ei];
        if (e.type != VIVID_EDITOR_EVENT_KEY) continue;
        if (e.action != ek::kPress && e.action != ek::kRepeat) continue;

        // [ / ]: cycle family
        if (e.key == ek::kLeftBracket) {
            const int nf = (family - 1 + bank::kBuiltinFamilyCount)
                           % bank::kBuiltinFamilyCount;
            set_named("wavetable_family", static_cast<float>(nf));
            continue;
        }
        if (e.key == ek::kRightBracket) {
            const int nf = (family + 1) % bank::kBuiltinFamilyCount;
            set_named("wavetable_family", static_cast<float>(nf));
            continue;
        }
        // - / =: cycle member
        if (e.key == ek::kMinus) {
            const int nm = (member - 1 + bank::kBuiltinMembersPerFamily)
                           % bank::kBuiltinMembersPerFamily;
            set_named("wavetable_member", static_cast<float>(nm));
            continue;
        }
        if (e.key == ek::kEqual) {
            const int nm = (member + 1) % bank::kBuiltinMembersPerFamily;
            set_named("wavetable_member", static_cast<float>(nm));
            continue;
        }
        // Left / Right: nudge position
        if (e.key == ek::kLeft) {
            set_named("position",
                      std::clamp(position - 0.01f, 0.0f, 1.0f));
            continue;
        }
        if (e.key == ek::kRight) {
            set_named("position",
                      std::clamp(position + 0.01f, 0.0f, 1.0f));
            continue;
        }
        // Up / Down: nudge unison voices
        if (e.key == ek::kUp) {
            set_named("unison_voices",
                      static_cast<float>(std::min(16, unison_voices + 1)));
            continue;
        }
        if (e.key == ek::kDown) {
            set_named("unison_voices",
                      static_cast<float>(std::max(1, unison_voices - 1)));
            continue;
        }
    }

    // ---- Mouse: handle region-specific interactions ----
    const auto& mouse = ctx->mouse;

    // Family/member grid click.
    const float grid_origin_x = browser_x + 4.0f;
    const float grid_origin_y = body_y + 28.0f;  // 28 = family-header row height
    const float grid_cell_w = (browser_w - 8.0f) /
                              static_cast<float>(bank::kBuiltinFamilyCount);
    const float grid_cell_h = (body_h - 28.0f - 8.0f) /
                              static_cast<float>(bank::kBuiltinMembersPerFamily);
    if (mouse.left_clicked) {
        auto hit = ed::family_cell_from_point(
            grid_origin_x, grid_origin_y,
            grid_cell_w, grid_cell_h, mouse.x, mouse.y);
        if (hit.family >= 0 && hit.member >= 0) {
            set_named("wavetable_family", static_cast<float>(hit.family));
            set_named("wavetable_member", static_cast<float>(hit.member));
        }
    }

    // Preview drag: set position from mouse y fraction.
    // Stacked-frames view has y=bottom → position 0 (front of stack) and
    // y=top → position 1 (back of stack). Drag-y picks a frame.
    const bool mouse_in_preview = (mouse.x >= preview_x &&
                                   mouse.x <  preview_x + preview_w &&
                                   mouse.y >= preview_y &&
                                   mouse.y <  preview_y + preview_h);
    if (mouse.left_clicked && mouse_in_preview) {
        editor_drag_position_ = true;
    }
    if (!mouse.left_down) editor_drag_position_ = false;
    if (editor_drag_position_ && preview_h > 0.0f) {
        const float frac = std::clamp(
            1.0f - (mouse.y - preview_y) / preview_h, 0.0f, 1.0f);
        set_named("position", frac);
    }

    // Preview scroll: nudge position.
    for (uint32_t ei = 0; ei < ctx->event_count; ++ei) {
        const auto& e = ctx->events[ei];
        if (e.type != VIVID_EDITOR_EVENT_MOUSE_SCROLL) continue;
        if (!mouse_in_preview) continue;
        const float step = (e.modifiers & ek::kModShift) ? 0.05f : 0.01f;
        const float dir = (e.scroll_dy > 0) ? +1.0f : -1.0f;
        set_named("position",
                  std::clamp(position + dir * step, 0.0f, 1.0f));
    }

    // Scatter: drag a voice → adjust unison_spread + unison_stereo.
    // Voice positions on the scatter: x = detune_cents / 100,
    // y = -pan (invert so +y is up). Hit radius = ~8px.
    const float scatter_cx = scatter_x + scatter_w * 0.5f;
    const float scatter_cy = scatter_y + scatter_h * 0.5f;
    // Units: 100 cents per half-width, 1.0 pan per half-height.
    const float scatter_half_w = scatter_w * 0.5f;
    const float scatter_half_h = scatter_h * 0.5f;
    const bool mouse_in_scatter = (mouse.x >= scatter_x &&
                                   mouse.x <  scatter_x + scatter_w &&
                                   mouse.y >= scatter_y &&
                                   mouse.y <  scatter_y + scatter_h);

    ed::VoicePoint voices[16];
    ed::compute_unison_layout(unison_voices, unison_spread, unison_stereo,
                              unison_spread_mode, voices);

    if (mouse.left_clicked && mouse_in_scatter) {
        // Find the nearest voice within 10px.
        int best = -1;
        float best_dist_sq = 100.0f;
        for (int i = 0; i < unison_voices; ++i) {
            const float vx = scatter_cx +
                             (voices[i].detune_cents / 100.0f) * scatter_half_w;
            const float vy = scatter_cy -
                             voices[i].pan * scatter_half_h;
            const float dx = mouse.x - vx;
            const float dy = mouse.y - vy;
            const float ds = dx * dx + dy * dy;
            if (ds < best_dist_sq) {
                best_dist_sq = ds;
                best = i;
            }
        }
        if (best >= 0) {
            editor_drag_unison_ = true;
            editor_drag_voice_idx_ = best;
        }
    }
    if (!mouse.left_down) {
        editor_drag_unison_ = false;
        editor_drag_voice_idx_ = -1;
    }
    if (editor_drag_unison_ && editor_drag_voice_idx_ >= 0) {
        // Delta from center = proportional adjustment of spread / stereo.
        // Keep the drag symmetric for simplicity (adjust global spread/
        // stereo rather than per-voice — per-voice is a deferred feature).
        const float dx = (mouse.x - scatter_cx) / std::max(1.0f, scatter_half_w);
        const float dy = (scatter_cy - mouse.y) / std::max(1.0f, scatter_half_h);
        // Scale: voice at index i normally sits at unison_detune_offset(i, n, spread),
        // which scales linearly with spread. So we infer spread from the drag.
        // Voice's normalized position in [-1, +1] range (Linear mode) =
        //   (2i / (n-1)) - 1 for n > 1; 0 for n == 1.
        float norm_pos = 0.0f;
        if (unison_voices > 1) {
            norm_pos = (2.0f * editor_drag_voice_idx_ /
                        static_cast<float>(unison_voices - 1)) - 1.0f;
        }
        if (std::abs(norm_pos) > 1e-3f) {
            const float implied_spread_100 = dx / norm_pos;  // as fraction of 100 cents
            const float new_spread = std::clamp(
                implied_spread_100 * 100.0f, 0.0f, 100.0f);
            set_named("unison_spread", new_spread);

            const float implied_stereo = dy / norm_pos;
            const float new_stereo = std::clamp(implied_stereo, 0.0f, 1.0f);
            set_named("unison_stereo", new_stereo);
        }
    }

    // Scatter scroll: adjust unison_voices.
    for (uint32_t ei = 0; ei < ctx->event_count; ++ei) {
        const auto& e = ctx->events[ei];
        if (e.type != VIVID_EDITOR_EVENT_MOUSE_SCROLL) continue;
        if (!mouse_in_scatter) continue;
        const int tick = (e.scroll_dy > 0) ? +1 : -1;
        set_named("unison_voices",
                  static_cast<float>(std::clamp(unison_voices + tick, 1, 16)));
    }

    // ---- Drawing ----

    // Top bar.
    if (d.draw_text) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%s · %s  ·  position %.2f",
            ed::family_long_name(family),
            ed::member_name(member),
            position);
        d.draw_text(o, kInset, top_y + 6.0f, buf,
            {th.bright_text.r, th.bright_text.g,
             th.bright_text.b, 0.95f}, 1.0f);

        const char* hints =
            "click cell=select  ·  [/] family  ·  -/= member  ·  "
            "drag preview=position  ·  scroll unison  ·  drag voice";
        const float scale = 0.7f;
        const float hints_w = d.text_width
            ? d.text_width(o, hints, scale) : 620.0f;
        d.draw_text(o, surf_w - kInset - hints_w, top_y + 8.0f, hints,
            {th.dim_text.r, th.dim_text.g, th.dim_text.b, 0.7f}, scale);
    }

    // --- Browser region ---
    vivid::draw_ui::draw_panel(d, o, browser_x, body_y, browser_w, body_h,
        {th.dark_bg.r, th.dark_bg.g, th.dark_bg.b, 0.9f},
        {th.separator.r, th.separator.g, th.separator.b, 0.6f}, 4.0f, 1.0f);

    // Family header row.
    if (d.draw_text) {
        for (int f = 0; f < bank::kBuiltinFamilyCount; ++f) {
            const float hx = grid_origin_x + f * grid_cell_w + 2.0f;
            const float hy = body_y + 6.0f;
            const bool active_col = (f == family);
            d.draw_text(o, hx, hy, ed::family_short_name(f),
                active_col
                    ? VividColor{th.bright_text.r, th.bright_text.g,
                                 th.bright_text.b, 0.95f}
                    : VividColor{th.dim_text.r, th.dim_text.g,
                                 th.dim_text.b, 0.8f}, 0.7f);
        }
    }

    // Member grid.
    for (int m = 0; m < bank::kBuiltinMembersPerFamily; ++m) {
        for (int f = 0; f < bank::kBuiltinFamilyCount; ++f) {
            const auto cell = ed::family_cell_rect(
                grid_origin_x, grid_origin_y,
                grid_cell_w, grid_cell_h, f, m);
            const bool is_active = (f == family && m == member);
            const bool is_row    = (m == member);
            const bool is_col    = (f == family);

            VividColor fill = is_active
                ? VividColor{th.accent.r, th.accent.g, th.accent.b, 0.65f}
                : (is_row || is_col)
                    ? VividColor{th.accent.r, th.accent.g, th.accent.b, 0.12f}
                    : VividColor{0.14f, 0.15f, 0.17f, 0.85f};
            vivid::draw_ui::draw_panel(d, o,
                cell.x + 1.0f, cell.y + 1.0f,
                cell.w - 2.0f, cell.h - 2.0f,
                fill, {0, 0, 0, 0}, 2.0f);

            // First column of each row: show member name as a text label.
            if (f == 0 && d.draw_text) {
                d.draw_text(o, browser_x + 4.0f,
                            cell.y + cell.h * 0.5f - 5.0f,
                            ed::member_name(m),
                            is_active
                                ? VividColor{th.bright_text.r, th.bright_text.g,
                                             th.bright_text.b, 0.9f}
                                : VividColor{th.dim_text.r, th.dim_text.g,
                                             th.dim_text.b, 0.75f}, 0.6f);
            }
        }
    }

    // --- Preview region: stacked-frames view ---
    //
    // Bottom of the region is position 0 (front of the stack); top is
    // position 1 (back). We render kStackFrames polylines sampled from
    // the wavetable at evenly-spaced positions, offset upward and slightly
    // right to suggest depth, with the frame nearest the current position
    // highlighted in the accent color. A horizontal cursor line across the
    // stack marks where `position` currently points.
    vivid::draw_ui::draw_panel(d, o, preview_x, preview_y, preview_w, preview_h,
        {th.dark_bg.r, th.dark_bg.g, th.dark_bg.b, 0.92f},
        {th.separator.r, th.separator.g, th.separator.b, 0.6f}, 4.0f, 1.0f);

    const bank::Wavetable* table = resolve_table();

    const float inset_px   = 8.0f;
    const float depth_x    = preview_w * 0.14f;       // right-shift per-frame
    const float depth_y    = preview_h - 2.0f * inset_px;
    const float line_w     = preview_w - depth_x - 2.0f * inset_px;
    const float per_frame_amp = std::max(8.0f,
        (depth_y / static_cast<float>(kStackFrames)) * 1.8f);

    // Live effective position (base + position_mod + smoothing) for the
    // active voice — updated every audio buffer by process_audio.
    const float effective_position = std::clamp(
        editor_effective_position_.load(std::memory_order_relaxed),
        0.0f, 1.0f);

    // Highlighted stack frame follows the *live* effective position so
    // per-note modulation sweeps are visible; the drag/scroll still sets
    // the base `position` param.
    const int current_stack_idx = std::clamp(
        static_cast<int>(std::lround(effective_position * (kStackFrames - 1))),
        0, kStackFrames - 1);

    auto frame_origin = [&](int fi, float& ox, float& oy) {
        const float frac = (kStackFrames > 1)
            ? static_cast<float>(fi) / (kStackFrames - 1) : 0.0f;
        ox = preview_x + inset_px + frac * depth_x;
        // fi=0 sits at the bottom, fi=N-1 at the top.
        oy = preview_y + preview_h - inset_px - frac * depth_y;
    };

    auto draw_stack_frame = [&](int fi, bool highlight) {
        if (!d.draw_line || line_w <= 0.0f) return;
        const float frac = (kStackFrames > 1)
            ? static_cast<float>(fi) / (kStackFrames - 1) : 0.0f;
        float samples[kPreviewSamples] = {};
        ed::sample_waveform_polyline(table, frac, samples, kPreviewSamples);

        float ox, oy;
        frame_origin(fi, ox, oy);

        const float amp_scale = per_frame_amp * 0.5f;
        const float dist = std::fabs(effective_position - frac);  // 0..1
        // Back frames slightly dimmer; current frame vivid.
        const float alpha = highlight
            ? 0.95f
            : std::max(0.14f, 0.55f - dist * 0.9f);
        const VividColor col = highlight
            ? VividColor{th.accent.r, th.accent.g, th.accent.b, alpha}
            : VividColor{th.bright_text.r * 0.6f + th.accent.r * 0.2f,
                         th.bright_text.g * 0.6f + th.accent.g * 0.2f,
                         th.bright_text.b * 0.6f + th.accent.b * 0.2f,
                         alpha};
        const float thickness = highlight ? 1.6f : 0.8f;

        float prev_x = ox;
        float prev_y = oy - samples[0] * amp_scale;
        for (int i = 1; i < kPreviewSamples; ++i) {
            const float t  = static_cast<float>(i) /
                             static_cast<float>(kPreviewSamples - 1);
            const float cx = ox + t * line_w;
            const float cy = oy - samples[i] * amp_scale;
            d.draw_line(o, prev_x, prev_y, cx, cy, thickness, col);
            prev_x = cx; prev_y = cy;
        }
    };

    // Paint non-current frames first, then the highlighted current frame
    // on top so it's never obscured.
    for (int fi = 0; fi < kStackFrames; ++fi) {
        if (fi == current_stack_idx) continue;
        draw_stack_frame(fi, /*highlight=*/false);
    }
    draw_stack_frame(current_stack_idx, /*highlight=*/true);

    // Base-position cursor (dim) — reflects the `position` param the
    // user sets by dragging; useful drag feedback. Drawn behind the
    // live cursor so modulation visibility wins when both overlap.
    if (d.draw_rect && preview_h > 0.0f && std::fabs(position - effective_position) > 1e-3f) {
        const int base_stack_idx = std::clamp(
            static_cast<int>(std::lround(position * (kStackFrames - 1))),
            0, kStackFrames - 1);
        float ox, oy;
        frame_origin(base_stack_idx, ox, oy);
        d.draw_rect(o, preview_x + 2.0f, oy - 0.5f,
                    preview_w - 4.0f, 1.0f,
                    {th.dim_text.r, th.dim_text.g, th.dim_text.b, 0.35f});
    }

    // Live-position cursor (bright accent) — tracks the effective
    // position that the oscillator is actually playing right now.
    if (d.draw_rect && preview_h > 0.0f) {
        float ox, oy;
        frame_origin(current_stack_idx, ox, oy);
        d.draw_rect(o, preview_x + 2.0f, oy - 0.5f,
                    preview_w - 4.0f, 1.0f,
                    {th.accent.r, th.accent.g, th.accent.b, 0.7f});
    }

    // --- Scatter region ---
    vivid::draw_ui::draw_panel(d, o, scatter_x, scatter_y, scatter_w, scatter_h,
        {th.dark_bg.r, th.dark_bg.g, th.dark_bg.b, 0.92f},
        {th.separator.r, th.separator.g, th.separator.b, 0.6f}, 4.0f, 1.0f);

    // Crosshair axes.
    if (d.draw_rect) {
        d.draw_rect(o, scatter_x, scatter_cy - 0.5f, scatter_w, 1.0f,
            {th.dim_text.r, th.dim_text.g, th.dim_text.b, 0.3f});
        d.draw_rect(o, scatter_cx - 0.5f, scatter_y, 1.0f, scatter_h,
            {th.dim_text.r, th.dim_text.g, th.dim_text.b, 0.3f});
    }

    // Axis labels.
    if (d.draw_text) {
        d.draw_text(o, scatter_x + 4.0f, scatter_y + 4.0f,
            "unison scatter  (x = detune cents, y = pan)",
            {th.dim_text.r, th.dim_text.g, th.dim_text.b, 0.7f}, 0.7f);
        char lb[48];
        std::snprintf(lb, sizeof(lb), "voices: %d  ·  spread %.0f¢  ·  stereo %.2f",
            unison_voices, unison_spread, unison_stereo);
        d.draw_text(o, scatter_x + 4.0f, scatter_y + scatter_h - 16.0f, lb,
            {th.dim_text.r, th.dim_text.g, th.dim_text.b, 0.7f}, 0.7f);
    }

    // Voice dots.
    for (int i = 0; i < unison_voices; ++i) {
        const float vx = scatter_cx +
            (voices[i].detune_cents / 100.0f) * scatter_half_w;
        const float vy = scatter_cy -
            voices[i].pan * scatter_half_h;
        const float r = (i == editor_drag_voice_idx_) ? 6.0f : 4.5f;
        if (d.draw_rounded_rect) {
            d.draw_rounded_rect(o, vx - r, vy - r, r * 2.0f, r * 2.0f, r,
                {kVoiceColor[0], kVoiceColor[1], kVoiceColor[2],
                 i == editor_drag_voice_idx_ ? 1.0f : 0.8f});
        } else if (d.draw_rect) {
            d.draw_rect(o, vx - r, vy - r, r * 2.0f, r * 2.0f,
                {kVoiceColor[0], kVoiceColor[1], kVoiceColor[2], 0.8f});
        }
    }

    // --- Side panel ---
    const float side_y = body_y;
    const float side_h = body_h;
    vivid::draw_ui::draw_panel(d, o, side_x, side_y, side_w, side_h,
        {th.dark_bg.r, th.dark_bg.g, th.dark_bg.b, 0.85f},
        {th.separator.r, th.separator.g, th.separator.b, 0.8f}, 4.0f, 1.0f);

    if (d.draw_text) {
        constexpr float kSpPad = 10.0f;
        char line[96];

        d.draw_text(o, side_x + kSpPad, side_y + kSpPad,
            ed::family_long_name(family),
            {th.bright_text.r, th.bright_text.g,
             th.bright_text.b, 0.95f}, 1.0f);
        d.draw_text(o, side_x + kSpPad, side_y + kSpPad + 18.0f,
            ed::member_name(member),
            {th.dim_text.r, th.dim_text.g, th.dim_text.b, 0.85f}, 0.9f);

        std::snprintf(line, sizeof(line), "position %.3f", position);
        d.draw_text(o, side_x + kSpPad, side_y + kSpPad + 44.0f, line,
            {th.dim_text.r, th.dim_text.g, th.dim_text.b, 0.9f}, 0.9f);

        std::snprintf(line, sizeof(line), "amplitude %.2f", amplitude);
        d.draw_text(o, side_x + kSpPad, side_y + kSpPad + 64.0f, line,
            {th.dim_text.r, th.dim_text.g, th.dim_text.b, 0.9f}, 0.9f);

        static const char* kWarpNames[] = {
            "None", "Sync", "BendPlus", "BendMinus", "Mirror",
            "Asym", "Quantize", "FM", "Flip"};
        std::snprintf(line, sizeof(line), "warp: %s (%.2f)",
            kWarpNames[std::clamp(warp_mode, 0, 8)], warp_amount);
        d.draw_text(o, side_x + kSpPad, side_y + kSpPad + 84.0f, line,
            {th.dim_text.r, th.dim_text.g, th.dim_text.b, 0.9f}, 0.9f);

        // Amplitude slider.
        const float slider_x = side_x + kSpPad;
        const float slider_w = side_w - 2.0f * kSpPad;
        const float amp_y = side_y + kSpPad + 112.0f;
        if (d.draw_rect) {
            d.draw_rect(o, slider_x, amp_y, slider_w, 10.0f,
                {0.13f, 0.14f, 0.16f, 0.9f});
            d.draw_rect(o, slider_x, amp_y, slider_w * amplitude, 10.0f,
                {th.accent.r, th.accent.g, th.accent.b, 0.85f});
        }
        d.draw_text(o, slider_x, amp_y - 14.0f, "Amplitude",
            {th.dim_text.r, th.dim_text.g, th.dim_text.b, 0.75f}, 0.7f);

        // Warp amount slider.
        const float warp_y = amp_y + 40.0f;
        if (d.draw_rect) {
            d.draw_rect(o, slider_x, warp_y, slider_w, 10.0f,
                {0.13f, 0.14f, 0.16f, 0.9f});
            d.draw_rect(o, slider_x, warp_y, slider_w * warp_amount, 10.0f,
                {th.accent.r, th.accent.g, th.accent.b, 0.85f});
        }
        d.draw_text(o, slider_x, warp_y - 14.0f, "Warp amount",
            {th.dim_text.r, th.dim_text.g, th.dim_text.b, 0.75f}, 0.7f);

        // Slider click handlers — drag to set.
        auto slider_hit = [&](float slider_y) {
            return (mouse.x >= slider_x && mouse.x < slider_x + slider_w &&
                    mouse.y >= slider_y && mouse.y < slider_y + 10.0f);
        };
        if (mouse.left_down && slider_hit(amp_y)) {
            const float v = std::clamp(
                (mouse.x - slider_x) / slider_w, 0.0f, 1.0f);
            set_named("amplitude", v);
        }
        if (mouse.left_down && slider_hit(warp_y)) {
            const float v = std::clamp(
                (mouse.x - slider_x) / slider_w, 0.0f, 1.0f);
            set_named("warp_amount", v);
        }
    }
}
