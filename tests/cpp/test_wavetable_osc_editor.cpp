// End-to-end tests for WavetableOsc::draw_editor(). Synthesised
// VividEditorContext drives keyboard + mouse flows; captured set_param
// writes verify the editor routes user intent to the right params.

#include "wavetable_osc_internal.h"
#include "wavetable_osc_editor_shared.h"
#include "operator_api/editor_keys.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace ek = ::vivid::editor_keys;
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

namespace {

struct CapturedSet {
    std::string name;
    float value = 0.0f;
};
struct CaptureCtx {
    std::vector<CapturedSet> calls;
};

void capture_set_param(void* opaque, const char* name, float v) {
    auto* c = static_cast<CaptureCtx*>(opaque);
    if (c) c->calls.push_back({std::string(name ? name : ""), v});
}
void capture_set_string_param(void*, const char*, const char*) {}

void noop_draw_rect(void*, float, float, float, float, VividColor) {}
void noop_draw_rounded_rect(void*, float, float, float, float, float, VividColor) {}
void noop_draw_text(void*, float, float, const char*, VividColor, float) {}
void noop_draw_line(void*, float, float, float, float, float, VividColor) {}
float fake_text_width(void*, const char* text, float scale) {
    const std::size_t len = text ? std::strlen(text) : 0;
    return static_cast<float>(len) * 7.0f * scale;
}
float fake_line_height(void*) { return 14.0f; }
void noop_push_clip(void*, float, float, float, float) {}
void noop_pop_clip(void*) {}

VividDrawAPI make_draw_api() {
    VividDrawAPI d{};
    d.draw_rect         = noop_draw_rect;
    d.draw_rounded_rect = noop_draw_rounded_rect;
    d.draw_text         = noop_draw_text;
    d.draw_line         = noop_draw_line;
    d.text_width        = fake_text_width;
    d.line_height       = fake_line_height;
    d.push_clip_rect    = noop_push_clip;
    d.pop_clip_rect     = noop_pop_clip;
    return d;
}

// Param slot count — must match WavetableOsc::collect_params() length.
// 27 params in order: wavetable_source, wavetable_family, wavetable_member,
// wav_file (string; param_values still has a slot at this index — set 0),
// position, amplitude, warp_mode, warp_amount, position_smooth_ms,
// warp_smooth_ms, phase_reset_mode, start_phase, phase_random,
// stereo_phase_offset, drift_amount, drift_rate_hz, unison_voices,
// unison_spread, unison_stereo, unison_spread_mode, unison_output_mode,
// detune, portamento, interaction_mode, interaction_depth,
// interaction_input_gain, interaction_tracking.
constexpr int kParamCount = 27;

struct EditorHarness {
    WavetableOsc core;
    CaptureCtx capture;
    std::vector<float> params;
    std::vector<float> outputs;
    std::vector<VividEditorEvent> events;
    VividEditorContext ctx{};

    EditorHarness() : params(kParamCount, 0.0f), outputs(1, 0.0f) {
        // Defaults aligned with constructor-set values.
        params[ed::kPositionIndex]      = 0.0f;
        params[ed::kAmplitudeIndex]     = 0.3f;
        params[ed::kWarpModeIndex]      = 0.0f;
        params[ed::kWarpAmountIndex]    = 0.0f;
        params[ed::kUnisonVoicesIndex]  = 1.0f;
        params[ed::kUnisonSpreadIndex]  = 20.0f;
        params[ed::kUnisonStereoIndex]  = 1.0f;

        ctx.surface_width  = 1200.0f;
        ctx.surface_height = 700.0f;
        ctx.dpi_scale      = 1.0f;
        ctx.draw           = make_draw_api();
        ctx.commands.opaque           = &capture;
        ctx.commands.set_param        = capture_set_param;
        ctx.commands.set_string_param = capture_set_string_param;
        ctx.param_values  = params.data();
        ctx.param_count   = static_cast<uint32_t>(params.size());
        ctx.output_values = outputs.data();
        ctx.output_count  = static_cast<uint32_t>(outputs.size());
        ctx.mouse         = {};
        ctx.time          = 0.0;
        refresh_events();
    }
    void refresh_events() {
        ctx.events      = events.empty() ? nullptr : events.data();
        ctx.event_count = static_cast<uint32_t>(events.size());
    }
    void clear_input() {
        events.clear(); refresh_events();
        ctx.mouse = {};
    }
    void clear_capture() { capture.calls.clear(); }
    void draw() {
        refresh_events();
        ctx.wants_keyboard = 0;
        core.draw_editor(&ctx);
    }
};

VividEditorEvent key_ev(int k, int mods = 0) {
    VividEditorEvent e{};
    e.type = VIVID_EDITOR_EVENT_KEY;
    e.key = k;
    e.action = ek::kPress;
    e.modifiers = mods;
    return e;
}
VividEditorEvent scroll_ev(float dy, int mods = 0) {
    VividEditorEvent e{};
    e.type = VIVID_EDITOR_EVENT_MOUSE_SCROLL;
    e.scroll_dy = dy;
    e.modifiers = mods;
    return e;
}

bool captured(const CaptureCtx& c, const char* name, float v,
              float tol = 1e-3f) {
    for (const auto& call : c.calls)
        if (call.name == name && std::fabs(call.value - v) < tol) return true;
    return false;
}
bool captured_name(const CaptureCtx& c, const char* name) {
    for (const auto& call : c.calls)
        if (call.name == name) return true;
    return false;
}
float captured_last(const CaptureCtx& c, const char* name) {
    for (auto it = c.calls.rbegin(); it != c.calls.rend(); ++it)
        if (it->name == name) return it->value;
    return std::nanf("");
}

} // namespace

int main() {
    std::fprintf(stderr, "=== Test: WavetableOsc draw_editor ===\n\n");

    // --- Editor metadata ---
    {
        auto m = WavetableOsc::editor_metadata();
        check(m.default_width  == 1200, "metadata: default_width = 1200");
        check(m.default_height == 700,  "metadata: default_height = 700");
        check(m.min_width      == 900,  "metadata: min_width = 900");
        check(m.title_suffix != nullptr &&
              std::strcmp(m.title_suffix, "WavetableOsc Editor") == 0,
              "metadata: title_suffix = WavetableOsc Editor");
    }

    // --- wants_keyboard set every frame ---
    {
        EditorHarness h;
        h.draw();
        check(h.ctx.wants_keyboard == 1, "draw_editor sets wants_keyboard");
    }

    // --- [ / ] cycle family, - / = cycle member ---
    {
        EditorHarness h;
        h.events = {key_ev(ek::kRightBracket)};
        h.draw();
        check(captured(h.capture, "wavetable_family", 1.0f),
              "] advances family to 1");

        h.clear_input(); h.clear_capture();
        h.params[ed::kWavetableFamilyIndex] = 0.0f;
        h.events = {key_ev(ek::kLeftBracket)};
        h.draw();
        check(captured(h.capture, "wavetable_family",
                       static_cast<float>(bank::kBuiltinFamilyCount - 1)),
              "[ from family=0 wraps to last family");

        h.clear_input(); h.clear_capture();
        h.params[ed::kWavetableMemberIndex] = 2.0f;
        h.events = {key_ev(ek::kEqual)};
        h.draw();
        check(captured(h.capture, "wavetable_member", 3.0f),
              "= advances member to 3");

        h.clear_input(); h.clear_capture();
        h.params[ed::kWavetableMemberIndex] = 0.0f;
        h.events = {key_ev(ek::kMinus)};
        h.draw();
        check(captured(h.capture, "wavetable_member",
                       static_cast<float>(bank::kBuiltinMembersPerFamily - 1)),
              "- from member=0 wraps to last member");
    }

    // --- Left / Right nudge position, Up / Down nudge unison_voices ---
    {
        EditorHarness h;
        h.params[ed::kPositionIndex] = 0.5f;
        h.events = {key_ev(ek::kRight)};
        h.draw();
        check(captured(h.capture, "position", 0.51f, 1e-3f),
              "Right arrow nudges position +0.01");

        h.clear_input(); h.clear_capture();
        h.params[ed::kPositionIndex] = 0.5f;
        h.events = {key_ev(ek::kLeft)};
        h.draw();
        check(captured(h.capture, "position", 0.49f, 1e-3f),
              "Left arrow nudges position -0.01");

        h.clear_input(); h.clear_capture();
        h.params[ed::kUnisonVoicesIndex] = 3.0f;
        h.events = {key_ev(ek::kUp)};
        h.draw();
        check(captured(h.capture, "unison_voices", 4.0f),
              "Up arrow nudges unison_voices +1");

        h.clear_input(); h.clear_capture();
        h.params[ed::kUnisonVoicesIndex] = 3.0f;
        h.events = {key_ev(ek::kDown)};
        h.draw();
        check(captured(h.capture, "unison_voices", 2.0f),
              "Down arrow nudges unison_voices -1");
    }

    // --- Clamps at extrema ---
    {
        EditorHarness h;
        h.params[ed::kPositionIndex] = 0.0f;
        h.events = {key_ev(ek::kLeft)};
        h.draw();
        check(captured(h.capture, "position", 0.0f),
              "Left at position=0 clamps to 0 (no negative)");

        h.clear_input(); h.clear_capture();
        h.params[ed::kUnisonVoicesIndex] = 16.0f;
        h.events = {key_ev(ek::kUp)};
        h.draw();
        check(captured(h.capture, "unison_voices", 16.0f),
              "Up at unison_voices=16 clamps to 16");

        h.clear_input(); h.clear_capture();
        h.params[ed::kUnisonVoicesIndex] = 1.0f;
        h.events = {key_ev(ek::kDown)};
        h.draw();
        check(captured(h.capture, "unison_voices", 1.0f),
              "Down at unison_voices=1 clamps to 1");
    }

    // --- Grid click → selects (family, member) ---
    {
        EditorHarness h;
        // Layout geometry (must match draw_editor):
        //   body_y = kInset + kTopBarH + kInset = 8 + 28 + 8 = 44
        //   browser_x = 8, browser_w = 240
        //   grid_origin_x = 8 + 4 = 12
        //   grid_origin_y = 44 + 28 = 72
        //   grid_cell_w   = (240 - 8) / 6 = 38.666...
        //   body_h = 700 - 44 - 8 = 648
        //   grid_cell_h   = (648 - 28 - 8) / 8 = 76.5
        // Target cell (family=2, member=3):
        //   center = (12 + 2.5 * 38.666, 72 + 3.5 * 76.5) = (108.666, 339.75)
        h.ctx.mouse.x = 108.666f;
        h.ctx.mouse.y = 339.75f;
        h.ctx.mouse.left_clicked = 1;
        h.ctx.mouse.left_down = 1;
        h.draw();
        check(captured(h.capture, "wavetable_family", 2.0f),
              "grid click selects family");
        check(captured(h.capture, "wavetable_member", 3.0f),
              "grid click selects member");
    }

    // --- Scroll wheel over preview nudges position ---
    {
        EditorHarness h;
        h.params[ed::kPositionIndex] = 0.3f;
        // Preview region: x in [252, 972), y in [44, 44 + 0.6*648) = [44, 432.8).
        // Put mouse well inside.
        h.ctx.mouse.x = 500.0f;
        h.ctx.mouse.y = 200.0f;
        h.events = {scroll_ev(+1.0f)};
        h.draw();
        check(captured(h.capture, "position", 0.31f, 1e-3f),
              "scroll up over preview nudges position +0.01");

        h.clear_input(); h.clear_capture();
        h.params[ed::kPositionIndex] = 0.3f;
        h.ctx.mouse.x = 500.0f;
        h.ctx.mouse.y = 200.0f;
        h.events = {scroll_ev(+1.0f, ek::kModShift)};
        h.draw();
        check(captured(h.capture, "position", 0.35f, 1e-3f),
              "Shift+scroll uses coarse position step (+0.05)");
    }

    // --- Preview drag sets position directly (y-axis; stacked view) ---
    {
        EditorHarness h;
        // Start a drag inside the preview. Preview region:
        //   x in [252, 972), y in [44, 432.8).
        // mouse.y near the top of preview → position near 1;
        // mouse.y near the bottom → position near 0.
        h.ctx.mouse.x = 500.0f;
        h.ctx.mouse.y = 100.0f;  // near top → position close to 1
        h.ctx.mouse.left_clicked = 1;
        h.ctx.mouse.left_down = 1;
        h.draw();
        check(captured_name(h.capture, "position"),
              "preview click emits position");
        const float v_top = captured_last(h.capture, "position");
        check(v_top > 0.75f,
              "preview click near top emits high position (stacked back)");

        h.clear_input(); h.clear_capture();
        h.ctx.mouse.x = 500.0f;
        h.ctx.mouse.y = 420.0f;  // near bottom → position close to 0
        h.ctx.mouse.left_clicked = 1;
        h.ctx.mouse.left_down = 1;
        h.draw();
        const float v_bot = captured_last(h.capture, "position");
        check(v_bot < 0.15f,
              "preview click near bottom emits low position (stacked front)");
    }

    // --- Scatter scroll nudges unison_voices ---
    {
        EditorHarness h;
        h.params[ed::kUnisonVoicesIndex] = 3.0f;
        // Scatter region: y starts at preview_y + preview_h + kInset
        //   = 44 + 0.6*648 + 8 ≈ 440.8 up to ~692.
        h.ctx.mouse.x = 500.0f;
        h.ctx.mouse.y = 550.0f;
        h.events = {scroll_ev(+1.0f)};
        h.draw();
        check(captured(h.capture, "unison_voices", 4.0f),
              "scatter scroll up → unison_voices +1");

        h.clear_input(); h.clear_capture();
        h.params[ed::kUnisonVoicesIndex] = 3.0f;
        h.ctx.mouse.x = 500.0f;
        h.ctx.mouse.y = 550.0f;
        h.events = {scroll_ev(-1.0f)};
        h.draw();
        check(captured(h.capture, "unison_voices", 2.0f),
              "scatter scroll down → unison_voices -1");
    }

    // --- Null ctx is safe ---
    {
        WavetableOsc core;
        core.draw_editor(nullptr);
        // If we're here, no crash.
        check(true, "draw_editor(nullptr) does not crash");
    }

    // --- Release of mouse clears drag flags ---
    {
        EditorHarness h;
        // Simulate an ongoing drag, then a release.
        h.core.editor_drag_position_ = true;
        h.core.editor_drag_unison_   = true;
        h.core.editor_drag_voice_idx_ = 2;
        h.ctx.mouse.left_down = 0;
        h.draw();
        check(h.core.editor_drag_position_ == false,
              "mouse release clears editor_drag_position_");
        check(h.core.editor_drag_unison_ == false,
              "mouse release clears editor_drag_unison_");
        check(h.core.editor_drag_voice_idx_ == -1,
              "mouse release clears editor_drag_voice_idx_");
    }

    std::fprintf(stderr, "\n=== %d failures ===\n", failures);
    return failures == 0 ? 0 : 1;
}
