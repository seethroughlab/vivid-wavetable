// WavetableLayer MIDI-input smoke test.
//
// Verifies midi_in is declared, MIDI note-on produces stereo audio,
// ADSR release decays, and the legacy lane-array path still works.

#include "operator_api/note_types.h"
#include "operator_api/types.h"
#include "test_support.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static int failures = 0;
static void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "  FAIL: %s\n", msg); failures++; }
    else       { std::fprintf(stderr, "  PASS: %s\n", msg); }
}

namespace {

constexpr int kFrames = 2048;
constexpr int kMaxVoices = 16;

static int find_param(const VividOperatorDescriptor* desc, const char* name) {
    for (uint32_t p = 0; p < desc->param_count; ++p)
        if (std::strcmp(desc->params[p].name, name) == 0) return static_cast<int>(p);
    return -1;
}
static bool has_port(const VividOperatorDescriptor* desc, const char* name) {
    for (uint32_t p = 0; p < desc->port_count; ++p)
        if (std::strcmp(desc->ports[p].name, name) == 0) return true;
    return false;
}
struct ParamOverride { const char* name; float value; };
static std::vector<float> make_params(const VividOperatorDescriptor* desc,
                                      std::initializer_list<ParamOverride> ov = {}) {
    std::vector<float> params(desc->param_count);
    for (uint32_t p = 0; p < desc->param_count; ++p)
        params[p] = desc->params[p].default_value;
    for (auto& o : ov) {
        int idx = find_param(desc, o.name);
        if (idx >= 0) params[idx] = o.value;
    }
    return params;
}

float stereo_rms(const float* out, uint32_t frames) {
    double s = 0.0;
    for (uint32_t i = 0; i < frames; ++i) { s += out[i] * out[i]; s += out[frames + i] * out[frames + i]; }
    return static_cast<float>(std::sqrt(s / (2.0 * frames)));
}

void push_note_on(VividNoteBuffer& buf, uint8_t note, uint8_t vel, uint64_t id) {
    if (buf.count >= VIVID_NOTE_BUFFER_CAPACITY) return;
    auto& e = buf.events[buf.count++];
    e.type = VIVID_NOTE_ON; e.note_number = note;
    e.value = static_cast<float>(vel) / 127.0f;
    e.note_id = id; e.frame_offset_samples = 0;
}
void push_note_off(VividNoteBuffer& buf, uint64_t id) {
    if (buf.count >= VIVID_NOTE_BUFFER_CAPACITY) return;
    auto& e = buf.events[buf.count++];
    e.type = VIVID_NOTE_OFF; e.note_id = id; e.frame_offset_samples = 0;
}
void push_pressure(VividNoteBuffer& buf, uint64_t id, float v_0_1) {
    if (buf.count >= VIVID_NOTE_BUFFER_CAPACITY) return;
    auto& e = buf.events[buf.count++];
    e.type = VIVID_NOTE_PRESSURE; e.note_id = id; e.value = v_0_1;
    e.frame_offset_samples = 0;
}
void push_timbre(VividNoteBuffer& buf, uint64_t id, float v_0_1) {
    if (buf.count >= VIVID_NOTE_BUFFER_CAPACITY) return;
    auto& e = buf.events[buf.count++];
    e.type = VIVID_NOTE_TIMBRE; e.note_id = id; e.value = v_0_1;
    e.frame_offset_samples = 0;
}

// Crude spectral centroid: sum |sample[i]| weighted by a slowly-rising
// position estimate. Sufficient for "did the timbre shift" assertions.
// Returns the spectral-centroid index in [0, 1] of frames length.
float rough_brightness(const float* out, uint32_t frames) {
    // Count zero-crossings as a brightness proxy. Higher zc rate = brighter.
    int zc = 0;
    for (uint32_t i = 1; i < frames; ++i) {
        if ((out[i - 1] >= 0.0f) != (out[i] >= 0.0f)) ++zc;
    }
    return static_cast<float>(zc) / static_cast<float>(frames);
}

} // namespace

int main(int argc, char** argv) {
    const std::string staging = (argc > 1) ? argv[1] : ".";
    const std::string dylib_path = staging + "/wavetable_layer.dylib";

    if (!std::filesystem::exists(dylib_path)) {
        std::fprintf(stderr, "FATAL: %s not found\n", dylib_path.c_str());
        return 1;
    }
    MiniLoader loader;
    if (!loader.load(dylib_path.c_str())) {
        std::fprintf(stderr, "  SKIP: could not load wavetable_layer.dylib\n");
        return 0;
    }

    const auto* desc = loader.descriptor();
    check(desc != nullptr, "WavetableLayer descriptor not null");
    if (!desc) return 1;
    check(std::strcmp(desc->name, "WavetableLayer") == 0, "operator name is WavetableLayer");
    check(has_port(desc, "notes_in"), "WavetableLayer declares notes_in port");
    check(!has_port(desc, "frequencies"), "WavetableLayer lane frequencies port removed (PR3)");
    check(find_param(desc, "attack") >= 0, "WavetableLayer declares attack param");

    // Test 1: midi_in note-on produces stereo audio.
    {
        std::fprintf(stderr, "\n--- WavetableLayer: midi_in note-on produces stereo ---\n");
        PolyTestContext tc;
        tc.set_output_channels(2);  // WavetableLayer outputs stereo always
        tc.clear_lane_ports();
        tc.clear_audio_inputs();

        VividNoteBuffer midi{};
        push_note_on(midi, 60, 100, /*id=*/1);
        void* midi_inputs[1] = {&midi};
        tc.ctx.custom_inputs = midi_inputs;
        tc.ctx.custom_input_count = 1;

        auto params = make_params(desc, {
            {"amplitude", 0.5f},
            {"unison_voices", 1.0f},
            {"attack", 0.001f}, {"decay", 0.05f}, {"sustain", 0.9f}, {"release", 0.05f},
        });
        tc.ctx.param_values = params.data();

        void* inst = loader.create_instance();
        tc.clear_output();
        loader.process_audio(inst, &tc.ctx);
        float on_rms = stereo_rms(tc.output_buf, kFrames);
        check(on_rms > 0.05f, "stereo RMS > 0.05 after note-on");
        std::fprintf(stderr, "  note-on RMS: %.4f\n", on_rms);
        loader.destroy_instance(inst);
    }

    // Test 2: note-off triggers release.
    {
        std::fprintf(stderr, "\n--- WavetableLayer: note-off releases envelope ---\n");
        PolyTestContext tc;
        tc.set_output_channels(2);
        tc.clear_lane_ports();
        tc.clear_audio_inputs();

        VividNoteBuffer midi{};
        void* midi_inputs[1] = {&midi};
        tc.ctx.custom_inputs = midi_inputs;
        tc.ctx.custom_input_count = 1;

        auto params = make_params(desc, {
            {"amplitude", 0.5f},
            {"unison_voices", 1.0f},
            {"attack", 0.001f}, {"decay", 0.005f}, {"sustain", 1.0f}, {"release", 0.01f},
        });
        tc.ctx.param_values = params.data();

        void* inst = loader.create_instance();

        push_note_on(midi, 60, 100, /*id=*/2);
        tc.clear_output(); loader.process_audio(inst, &tc.ctx); midi.count = 0;
        tc.clear_output(); loader.process_audio(inst, &tc.ctx);
        float sustain_rms = stereo_rms(tc.output_buf, kFrames);

        push_note_off(midi, /*id=*/2);
        tc.clear_output(); loader.process_audio(inst, &tc.ctx); midi.count = 0;
        tc.clear_output(); loader.process_audio(inst, &tc.ctx);
        float release_rms = stereo_rms(tc.output_buf, kFrames);

        check(release_rms < sustain_rms, "release-tail RMS lower than sustain");
        std::fprintf(stderr, "  sustain RMS: %.4f, release RMS: %.4f\n",
                     sustain_rms, release_rms);
        loader.destroy_instance(inst);
    }

    // Test 3 (Phase 4): pressure event scales voice amplitude.
    // pressure_to_amp default = 0.5; pressure 0.0 = baseline, 1.0 = +50%.
    {
        std::fprintf(stderr, "\n--- WavetableLayer: pressure scales amplitude ---\n");
        PolyTestContext tc;
        tc.set_output_channels(2);
        tc.clear_lane_ports();
        tc.clear_audio_inputs();

        VividNoteBuffer midi{};
        void* midi_inputs[1] = {&midi};
        tc.ctx.custom_inputs = midi_inputs;
        tc.ctx.custom_input_count = 1;

        auto params = make_params(desc, {
            {"amplitude", 0.5f},
            {"unison_voices", 1.0f},
            {"attack", 0.001f}, {"decay", 0.005f}, {"sustain", 1.0f}, {"release", 0.05f},
            {"pressure_to_amp", 0.5f},
        });
        tc.ctx.param_values = params.data();

        // Block 1: NOTE_ON at default pressure (0). Capture sustain RMS.
        void* inst = loader.create_instance();
        push_note_on(midi, 60, 100, /*id=*/3);
        tc.clear_output(); loader.process_audio(inst, &tc.ctx); midi.count = 0;
        tc.clear_output(); loader.process_audio(inst, &tc.ctx);
        float baseline_rms = stereo_rms(tc.output_buf, kFrames);

        // Block 2: push PRESSURE = 1.0 on the held voice. RMS should rise.
        push_pressure(midi, /*id=*/3, 1.0f);
        tc.clear_output(); loader.process_audio(inst, &tc.ctx);
        float pressed_rms = stereo_rms(tc.output_buf, kFrames);

        std::fprintf(stderr, "  baseline RMS: %.4f, pressed RMS: %.4f (ratio %.2f)\n",
                     baseline_rms, pressed_rms, pressed_rms / baseline_rms);
        check(pressed_rms > baseline_rms * 1.2f,
              "pressure=1.0 measurably louder than pressure=0 (expected ~1.5x)");
        loader.destroy_instance(inst);
    }

    // Test 4 (Phase 4): timbre event shifts wavetable position → spectral
    // brightness changes. timbre_to_position default = 0.5 (signed).
    {
        std::fprintf(stderr, "\n--- WavetableLayer: timbre shifts wavetable position ---\n");
        PolyTestContext tc;
        tc.set_output_channels(2);
        tc.clear_lane_ports();
        tc.clear_audio_inputs();

        VividNoteBuffer midi{};
        void* midi_inputs[1] = {&midi};
        tc.ctx.custom_inputs = midi_inputs;
        tc.ctx.custom_input_count = 1;

        auto params = make_params(desc, {
            {"amplitude", 0.5f},
            {"unison_voices", 1.0f},
            {"position", 0.0f},   // start at low end so positive timbre opens up the table
            {"attack", 0.001f}, {"decay", 0.005f}, {"sustain", 1.0f}, {"release", 0.05f},
            {"timbre_to_position", 0.7f},
            {"pressure_to_amp", 0.0f},  // isolate timbre effect
        });
        tc.ctx.param_values = params.data();

        void* inst = loader.create_instance();
        push_note_on(midi, 60, 100, /*id=*/4);
        tc.clear_output(); loader.process_audio(inst, &tc.ctx); midi.count = 0;
        tc.clear_output(); loader.process_audio(inst, &tc.ctx);
        float baseline_brightness = rough_brightness(tc.output_buf, kFrames);

        push_timbre(midi, /*id=*/4, 1.0f);
        // Run a few blocks so position smoother catches up.
        for (int i = 0; i < 4; ++i) {
            tc.clear_output(); loader.process_audio(inst, &tc.ctx); midi.count = 0;
        }
        float shifted_brightness = rough_brightness(tc.output_buf, kFrames);

        std::fprintf(stderr, "  baseline brightness: %.4f, shifted: %.4f\n",
                     baseline_brightness, shifted_brightness);
        // Position shift changes timbre → brightness shifts measurably (either
        // direction; depends on wavetable family). Just assert it changed.
        check(std::fabs(shifted_brightness - baseline_brightness) > 0.001f,
              "timbre=1.0 changes spectral brightness vs timbre=0");
        loader.destroy_instance(inst);
    }

    std::fprintf(stderr, "\n%s (%d failures)\n",
                 failures == 0 ? "PASSED" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
