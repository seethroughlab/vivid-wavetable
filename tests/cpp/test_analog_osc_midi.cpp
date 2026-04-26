// AnalogOsc MIDI-input smoke tests.
//
// Verifies that:
//   1. AnalogOsc declares a midi_in custom-ref port.
//   2. With midi_in connected and a note-on, channel 0/1 receive summed audio.
//   3. With note-off, envelope releases and audio decays.
//   4. Polyphonic chord (3 simultaneous notes) is louder than mono.
//   5. The legacy lane-array path still works when midi_in is absent.
//   6. Empty MIDI buffer with no held notes is silent.

#include "operator_api/note_types.h"
#include "operator_api/types.h"
#include "test_support.h"

#include <algorithm>
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
constexpr uint32_t kSampleRate = 48000;
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

// Stereo-channel RMS over the L+R buffer (channels 0 and 1 of the
// per-voice multichannel output).
static float stereo_rms(const float* out, uint32_t frames) {
    double s = 0.0;
    const float* L = out;
    const float* R = out + frames;
    for (uint32_t i = 0; i < frames; ++i) {
        s += L[i] * L[i];
        s += R[i] * R[i];
    }
    return static_cast<float>(std::sqrt(s / (2.0 * frames)));
}

// Allocate a fresh note_id for each note-on so the synth's allocator
// treats overlapping same-pitch notes as distinct voices.
void push_note_on(VividNoteBuffer& buf, uint8_t note, uint8_t vel, uint64_t id) {
    if (buf.count >= VIVID_NOTE_BUFFER_CAPACITY) return;
    auto& e = buf.events[buf.count++];
    e.type = VIVID_NOTE_ON;
    e.note_number = note;
    e.value = static_cast<float>(vel) / 127.0f;
    e.note_id = id;
    e.frame_offset_samples = 0;
}
void push_note_off(VividNoteBuffer& buf, uint64_t id) {
    if (buf.count >= VIVID_NOTE_BUFFER_CAPACITY) return;
    auto& e = buf.events[buf.count++];
    e.type = VIVID_NOTE_OFF;
    e.note_id = id;
    e.frame_offset_samples = 0;
}

} // namespace

int main(int argc, char** argv) {
    const std::string staging = (argc > 1) ? argv[1] : ".";
    const std::string dylib_path = staging + "/analog_osc.dylib";

    if (!std::filesystem::exists(dylib_path)) {
        std::fprintf(stderr, "FATAL: %s not found\n", dylib_path.c_str());
        return 1;
    }

    MiniLoader loader;
    if (!loader.load(dylib_path.c_str())) {
        std::fprintf(stderr, "FATAL: failed to load %s\n", dylib_path.c_str());
        return 1;
    }

    const auto* desc = loader.descriptor();
    check(desc != nullptr, "AnalogOsc descriptor not null");
    if (!desc) return 1;
    check(std::strcmp(desc->name, "AnalogOsc") == 0, "operator name is AnalogOsc");
    check(has_port(desc, "notes_in"), "AnalogOsc declares notes_in port");
    check(has_port(desc, "frequencies"), "AnalogOsc keeps lane frequencies port (override)");
    check(find_param(desc, "attack") >= 0, "AnalogOsc declares attack param");
    check(find_param(desc, "release") >= 0, "AnalogOsc declares release param");

    // ---------------------------------------------------------------------
    // Test 1: midi_in note-on produces audio in channel 0/1.
    // ---------------------------------------------------------------------
    {
        std::fprintf(stderr, "\n--- AnalogOsc: midi_in note-on produces audio ---\n");
        PolyTestContext tc;
        tc.set_output_channels(kMaxVoices);
        tc.clear_lane_ports();
        tc.clear_audio_inputs();

        VividNoteBuffer midi{};
        push_note_on(midi, 60, 100, /*id=*/1);  // note-on C4 vel 100
        void* midi_inputs[1] = {&midi};
        tc.ctx.custom_inputs = midi_inputs;
        tc.ctx.custom_input_count = 1;

        auto params = make_params(desc, {
            {"waveform", 0.0f},     // sine — easiest to reason about
            {"amplitude", 0.5f},
            {"attack", 0.001f}, {"decay", 0.05f}, {"sustain", 0.9f},
            {"release", 0.05f},
        });
        tc.ctx.param_values = params.data();

        void* inst = loader.create_instance();
        check(inst != nullptr, "instance created");
        tc.clear_output();
        loader.process_audio(inst, &tc.ctx);

        float on_rms = stereo_rms(tc.output_buf, kFrames);
        check(on_rms > 0.05f, "stereo RMS > 0.05 after note-on");
        std::fprintf(stderr, "  note-on RMS: %.4f\n", on_rms);
        // Verify channels 2..N stay zero (we only sum into 0/1).
        bool higher_zero = true;
        for (int ch = 2; ch < kMaxVoices; ++ch) {
            for (uint32_t s = 0; s < kFrames; ++s) {
                if (std::fabs(tc.output_buf[ch * kFrames + s]) > 1e-6f) {
                    higher_zero = false; break;
                }
            }
            if (!higher_zero) break;
        }
        check(higher_zero, "channels 2..15 are zero on MIDI path");

        loader.destroy_instance(inst);
    }

    // ---------------------------------------------------------------------
    // Test 2: note-off triggers release decay.
    // ---------------------------------------------------------------------
    {
        std::fprintf(stderr, "\n--- AnalogOsc: note-off releases envelope ---\n");
        PolyTestContext tc;
        tc.set_output_channels(kMaxVoices);
        tc.clear_lane_ports();
        tc.clear_audio_inputs();

        VividNoteBuffer midi{};
        void* midi_inputs[1] = {&midi};
        tc.ctx.custom_inputs = midi_inputs;
        tc.ctx.custom_input_count = 1;

        auto params = make_params(desc, {
            {"waveform", 0.0f},
            {"amplitude", 0.5f},
            {"attack", 0.001f}, {"decay", 0.005f}, {"sustain", 1.0f},
            {"release", 0.01f},
        });
        tc.ctx.param_values = params.data();

        void* inst = loader.create_instance();

        // Block 1: note on
        push_note_on(midi, 60, 100, /*id=*/2);
        tc.clear_output();
        loader.process_audio(inst, &tc.ctx);
        midi.count = 0;
        // Block 2: sustain
        tc.clear_output();
        loader.process_audio(inst, &tc.ctx);
        float sustain_rms = stereo_rms(tc.output_buf, kFrames);
        // Block 3: note off
        push_note_off(midi, /*id=*/2);
        tc.clear_output();
        loader.process_audio(inst, &tc.ctx);
        midi.count = 0;
        // Block 4: release tail
        tc.clear_output();
        loader.process_audio(inst, &tc.ctx);
        float release_rms = stereo_rms(tc.output_buf, kFrames);

        check(release_rms < sustain_rms, "release-tail RMS lower than sustain RMS");
        std::fprintf(stderr, "  sustain RMS: %.4f, release RMS: %.4f\n",
                     sustain_rms, release_rms);

        loader.destroy_instance(inst);
    }

    // ---------------------------------------------------------------------
    // Test 3: polyphonic chord (3 voices) — sums into stereo.
    // ---------------------------------------------------------------------
    {
        std::fprintf(stderr, "\n--- AnalogOsc: 3-voice chord summed to stereo ---\n");
        PolyTestContext tc;
        tc.set_output_channels(kMaxVoices);
        tc.clear_lane_ports();
        tc.clear_audio_inputs();

        VividNoteBuffer midi{};
        void* midi_inputs[1] = {&midi};
        tc.ctx.custom_inputs = midi_inputs;
        tc.ctx.custom_input_count = 1;

        auto params = make_params(desc, {
            {"waveform", 0.0f},
            {"amplitude", 0.3f},
            {"attack", 0.001f}, {"decay", 0.05f}, {"sustain", 0.9f},
            {"release", 0.05f},
        });
        tc.ctx.param_values = params.data();

        // Single note baseline.
        void* inst1 = loader.create_instance();
        push_note_on(midi, 60, 100, /*id=*/3);
        tc.clear_output();
        loader.process_audio(inst1, &tc.ctx);
        midi.count = 0;
        tc.clear_output();
        loader.process_audio(inst1, &tc.ctx);
        float mono_rms = stereo_rms(tc.output_buf, kFrames);
        loader.destroy_instance(inst1);

        // Three-note chord — distinct ids per voice.
        void* inst2 = loader.create_instance();
        push_note_on(midi, 60, 100, /*id=*/4);
        push_note_on(midi, 64, 100, /*id=*/5);
        push_note_on(midi, 67, 100, /*id=*/6);
        tc.clear_output();
        loader.process_audio(inst2, &tc.ctx);
        midi.count = 0;
        tc.clear_output();
        loader.process_audio(inst2, &tc.ctx);
        float chord_rms = stereo_rms(tc.output_buf, kFrames);

        check(chord_rms > 1.5f * mono_rms, "chord RMS > 1.5x mono RMS");
        std::fprintf(stderr, "  mono RMS: %.4f, chord RMS: %.4f\n", mono_rms, chord_rms);
        loader.destroy_instance(inst2);
    }

    // ---------------------------------------------------------------------
    // Test 4: legacy lane-array path still works (midi_in absent).
    // ---------------------------------------------------------------------
    {
        std::fprintf(stderr, "\n--- AnalogOsc: legacy lane-array path ---\n");
        PolyTestContext tc;
        tc.set_output_channels(kMaxVoices);
        tc.clear_audio_inputs();
        tc.setup_analog_voice(440.0f);
        tc.ctx.custom_inputs = nullptr;
        tc.ctx.custom_input_count = 0;

        auto params = make_params(desc, {
            {"waveform", 0.0f},
            {"amplitude", 0.5f},
        });
        tc.ctx.param_values = params.data();

        void* inst = loader.create_instance();
        tc.clear_output();
        loader.process_audio(inst, &tc.ctx);
        // In lane mode, voice 0 emits to channel 0. Check it's non-silent.
        double s = 0.0;
        for (uint32_t i = 0; i < kFrames; ++i)
            s += tc.output_buf[i] * tc.output_buf[i];
        float ch0_rms = static_cast<float>(std::sqrt(s / kFrames));
        check(ch0_rms > 0.05f, "lane-array path channel 0 RMS > 0.05");
        std::fprintf(stderr, "  lane RMS (ch0): %.4f\n", ch0_rms);
        loader.destroy_instance(inst);
    }

    // ---------------------------------------------------------------------
    // Test 5: silent at rest with no notes.
    // ---------------------------------------------------------------------
    {
        std::fprintf(stderr, "\n--- AnalogOsc: silent with no notes ---\n");
        PolyTestContext tc;
        tc.set_output_channels(kMaxVoices);
        tc.clear_lane_ports();
        tc.clear_audio_inputs();

        VividNoteBuffer midi{};
        void* midi_inputs[1] = {&midi};
        tc.ctx.custom_inputs = midi_inputs;
        tc.ctx.custom_input_count = 1;

        auto params = make_params(desc);
        tc.ctx.param_values = params.data();

        void* inst = loader.create_instance();
        tc.clear_output();
        loader.process_audio(inst, &tc.ctx);
        float rms = stereo_rms(tc.output_buf, kFrames);
        check(rms < 1e-4f, "stereo RMS near zero with no MIDI events");
        loader.destroy_instance(inst);
    }

    std::fprintf(stderr, "\n%s (%d failures)\n",
                 failures == 0 ? "PASSED" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
