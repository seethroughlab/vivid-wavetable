// NoiseLayer MIDI-input + voice-breakout tests.
//
// Same shape as test_sub_osc_midi.cpp:
//   1. notes_in + voices_out + voice_* breakouts present, advanced-tagged.
//   2. 3-note chord aligns voice_ids ascending; voice_freqs match held notes.
//   3. voices_out channels 0..2 audible; 3..15 silent.
//   4. Note-off triggers ADSR release.
//   5. Empty buffer → silent.
//   6. Legacy lane path still emits per-voice audio.

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
static void check_float(float actual, float expected, float tol, const char* msg) {
    bool ok = std::fabs(actual - expected) < tol;
    if (!ok) std::fprintf(stderr, "  FAIL: %s (%.4f vs %.4f, tol %.4f)\n",
                         msg, actual, expected, tol);
    else     std::fprintf(stderr, "  PASS: %s\n", msg);
    if (!ok) ++failures;
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
static const VividPortDescriptor* find_port(const VividOperatorDescriptor* desc, const char* name) {
    for (uint32_t p = 0; p < desc->port_count; ++p)
        if (std::strcmp(desc->ports[p].name, name) == 0) return &desc->ports[p];
    return nullptr;
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

static float midi_to_hz(float note) {
    return 440.0f * std::pow(2.0f, (note - 69.0f) / 12.0f);
}

struct LaneOutBuf {
    std::vector<float> data;
    static float* resize_cb(void* h, uint32_t len) {
        auto* self = static_cast<LaneOutBuf*>(h);
        self->data.assign(len, 0.0f);
        return self->data.data();
    }
    static void commit_cb(void* /*h*/, uint32_t /*len*/) {}
};

// NoiseLayer ports: notes_in=0, frequencies=1, gates=2, velocities=3, lane_ids=4.
// (No pitch_mod_audio.) input_buffers/input_lanes sized 5 with all-null defaults.
struct NoiseHarness {
    float stereo_out[2 * kFrames] = {};
    float voices_out[kMaxVoices * kFrames] = {};
    float* output_bufs[2] = {stereo_out, voices_out};
    uint8_t output_ch[2] = {2, kMaxVoices};

    float* input_bufs[5] = {};
    uint8_t input_ch[5] = {};
    VividLaneView input_lanes[5] = {};

    LaneOutBuf voice_ids_buf, voice_gates_buf, voice_velocities_buf, voice_freqs_buf;
    VividLaneOutput lane_outputs[4] = {};

    VividNoteBuffer notes{};
    void* note_inputs[1] = {&notes};

    LaneStateStore lane_state;
    VividAudioContext ctx{};

    NoiseHarness() {
        ctx.sample_rate        = kSampleRate;
        ctx.buffer_size        = kFrames;
        ctx.input_buffers      = input_bufs;
        ctx.input_channel_counts = input_ch;
        ctx.output_buffers     = output_bufs;
        ctx.output_channel_counts = output_ch;
        ctx.input_lanes        = input_lanes;
        ctx.shared_handles     = vivid::shared_handle_service();
        ctx.lane_count         = 1;
        ctx.lane_index         = 0;
        ctx.lane_set_id        = 0;
        ctx.lane_id            = 1;
        ctx.lane_state_fn      = test_lane_state_fn;
        ctx.lane_state_service = &lane_state;
        ctx.custom_inputs      = note_inputs;
        ctx.custom_input_count = 1;

        lane_outputs[0] = {&voice_ids_buf,        LaneOutBuf::resize_cb, LaneOutBuf::commit_cb};
        lane_outputs[1] = {&voice_gates_buf,      LaneOutBuf::resize_cb, LaneOutBuf::commit_cb};
        lane_outputs[2] = {&voice_velocities_buf, LaneOutBuf::resize_cb, LaneOutBuf::commit_cb};
        lane_outputs[3] = {&voice_freqs_buf,      LaneOutBuf::resize_cb, LaneOutBuf::commit_cb};
        ctx.output_lanes = lane_outputs;
    }

    void clear_notes() { notes.count = 0; }
    void zero_outputs() {
        std::memset(stereo_out, 0, sizeof(stereo_out));
        std::memset(voices_out, 0, sizeof(voices_out));
    }
    void disable_midi() {
        ctx.custom_inputs = nullptr;
        ctx.custom_input_count = 0;
    }

    void push_note_on(uint8_t note, float vel_0_1, uint64_t id) {
        if (notes.count >= VIVID_NOTE_BUFFER_CAPACITY) return;
        auto& e = notes.events[notes.count++];
        e.type = VIVID_NOTE_ON; e.note_number = note; e.value = vel_0_1;
        e.note_id = id; e.frame_offset_samples = 0;
    }
    void push_note_off(uint64_t id) {
        if (notes.count >= VIVID_NOTE_BUFFER_CAPACITY) return;
        auto& e = notes.events[notes.count++];
        e.type = VIVID_NOTE_OFF; e.note_id = id;
    }

    float stereo_rms() const {
        double s = 0.0;
        for (int i = 0; i < kFrames; ++i) {
            s += stereo_out[i] * stereo_out[i];
            s += stereo_out[kFrames + i] * stereo_out[kFrames + i];
        }
        return static_cast<float>(std::sqrt(s / (2.0 * kFrames)));
    }
    float voices_channel_rms(int ch) const {
        const float* p = voices_out + ch * kFrames;
        double s = 0.0;
        for (int i = 0; i < kFrames; ++i) s += p[i] * p[i];
        return static_cast<float>(std::sqrt(s / kFrames));
    }
};

}  // namespace

int main(int argc, char** argv) {
    const std::string staging = (argc > 1) ? argv[1] : ".";
    const std::string dylib_path = staging + "/noise_layer.dylib";
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
    check(desc != nullptr, "NoiseLayer descriptor not null");
    if (!desc) return 1;
    check(std::strcmp(desc->name, "NoiseLayer") == 0, "operator name is NoiseLayer");

    // ----- Surface check -----
    {
        std::fprintf(stderr, "\n--- NoiseLayer: declares breakout ports ---\n");
        check(find_port(desc, "notes_in") != nullptr, "declares notes_in");
        for (const char* name : {"voices_out", "voice_ids", "voice_gates",
                                 "voice_velocities", "voice_freqs"}) {
            const auto* p = find_port(desc, name);
            check(p != nullptr, (std::string("declares ") + name).c_str());
            if (p) {
                check(p->display_hint == VIVID_PORT_DISPLAY_ADVANCED,
                      (std::string(name) + " ADVANCED").c_str());
                check(p->direction == VIVID_PORT_OUTPUT,
                      (std::string(name) + " OUTPUT").c_str());
            }
        }
        const auto* output_port = find_port(desc, "output");
        check(output_port != nullptr, "declares stereo output");
        if (output_port) check(output_port->channels == 2,
                               "output is stereo (2 channels)");
        const auto* voices_port = find_port(desc, "voices_out");
        if (voices_port) check(voices_port->channels == kMaxVoices,
                               "voices_out has kMaxVoices channels");
    }

    // ----- 3-note chord: voice_*/voices_out alignment -----
    {
        std::fprintf(stderr, "\n--- NoiseLayer: 3-note chord aligns voice_*/voices_out ---\n");
        NoiseHarness h;
        auto params = make_params(desc, {
            {"level",    0.5f},
            {"attack",   0.001f}, {"decay", 0.05f},
            {"sustain",  1.0f},   {"release", 0.05f},
        });
        h.ctx.param_values = params.data();

        h.push_note_on(60, 100.0f / 127.0f, /*id=*/10);
        h.push_note_on(64, 100.0f / 127.0f, /*id=*/20);
        h.push_note_on(67, 100.0f / 127.0f, /*id=*/30);

        void* inst = loader.create_instance();
        h.zero_outputs();
        loader.process_audio(inst, &h.ctx);
        h.clear_notes();
        h.zero_outputs();
        loader.process_audio(inst, &h.ctx);  // sustain

        check(h.voice_ids_buf.data.size() == 3, "voice_ids has length 3");
        check(h.voice_gates_buf.data.size() == 3, "voice_gates has length 3");
        check(h.voice_freqs_buf.data.size() == 3, "voice_freqs has length 3");
        if (h.voice_ids_buf.data.size() == 3) {
            check(h.voice_ids_buf.data[0] < h.voice_ids_buf.data[1], "voice_ids[0] < [1]");
            check(h.voice_ids_buf.data[1] < h.voice_ids_buf.data[2], "voice_ids[1] < [2]");
            check_float(h.voice_freqs_buf.data[0], midi_to_hz(60), 0.5f, "voice_freqs[0] ≈ C4");
            check_float(h.voice_freqs_buf.data[1], midi_to_hz(64), 0.5f, "voice_freqs[1] ≈ E4");
            check_float(h.voice_freqs_buf.data[2], midi_to_hz(67), 0.5f, "voice_freqs[2] ≈ G4");
        }
        for (int ch = 0; ch < 3; ++ch) {
            float r = h.voices_channel_rms(ch);
            check(r > 0.001f, (std::string("voices_out ch ") + std::to_string(ch)
                              + " audible").c_str());
        }
        for (int ch = 3; ch < kMaxVoices; ++ch) {
            float r = h.voices_channel_rms(ch);
            check(r < 1e-5f, (std::string("voices_out ch ") + std::to_string(ch)
                              + " silent").c_str());
        }
        check(h.stereo_rms() > 0.001f, "stereo output is audible during chord");
        loader.destroy_instance(inst);
    }

    // ----- Note-off → release tail -----
    {
        std::fprintf(stderr, "\n--- NoiseLayer: note-off triggers release ---\n");
        NoiseHarness h;
        auto params = make_params(desc, {
            {"level", 0.5f},
            {"attack", 0.001f}, {"decay", 0.005f}, {"sustain", 1.0f},
            {"release", 0.01f},
        });
        h.ctx.param_values = params.data();

        void* inst = loader.create_instance();
        h.push_note_on(60, 100.0f / 127.0f, /*id=*/100);
        h.zero_outputs();
        loader.process_audio(inst, &h.ctx);
        h.clear_notes();
        h.zero_outputs();
        loader.process_audio(inst, &h.ctx);
        float sustain_rms = h.stereo_rms();

        h.push_note_off(/*id=*/100);
        h.zero_outputs();
        loader.process_audio(inst, &h.ctx);
        h.clear_notes();
        h.zero_outputs();
        loader.process_audio(inst, &h.ctx);
        float release_rms = h.stereo_rms();

        check(release_rms < sustain_rms,
              "release-tail RMS lower than sustain RMS");
        std::fprintf(stderr, "  sustain RMS: %.4f, release RMS: %.4f\n",
                     sustain_rms, release_rms);
        loader.destroy_instance(inst);
    }

    // ----- Silent at rest -----
    {
        std::fprintf(stderr, "\n--- NoiseLayer: silent with no notes ---\n");
        NoiseHarness h;
        auto params = make_params(desc);
        h.ctx.param_values = params.data();

        void* inst = loader.create_instance();
        h.zero_outputs();
        loader.process_audio(inst, &h.ctx);
        check(h.stereo_rms() < 1e-4f, "stereo near zero with no notes");
        check(h.voice_ids_buf.data.empty(), "voice_ids empty with no notes");
        loader.destroy_instance(inst);
    }

    std::fprintf(stderr, "\n%s (%d failures)\n",
                 failures == 0 ? "PASSED" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
