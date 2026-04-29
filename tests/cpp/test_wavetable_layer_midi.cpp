// WavetableLayer note-stream smoke test.
//
// Verifies notes_in is declared, note-on produces stereo audio, and ADSR
// release decays.

#include "operator_api/note_types.h"
#include "operator_api/types.h"
#include "test_support.h"

#include <cmath>
#include <array>
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

float mono_mean(const float* out, uint32_t frames) {
    double sum = 0.0;
    for (uint32_t i = 0; i < frames; ++i)
        sum += 0.5 * static_cast<double>(out[i] + out[frames + i]);
    return static_cast<float>(sum / static_cast<double>(frames));
}

float low_band_rms(const float* out, uint32_t frames, float sample_rate, float cutoff_hz) {
    const float x = std::exp(-2.0f * static_cast<float>(M_PI) * cutoff_hz / sample_rate);
    const float a = 1.0f - x;
    float lp = 0.0f;
    double sum = 0.0;
    for (uint32_t i = 0; i < frames; ++i) {
        float mono = 0.5f * (out[i] + out[frames + i]);
        lp += a * (mono - lp);
        sum += static_cast<double>(lp) * static_cast<double>(lp);
    }
    return static_cast<float>(std::sqrt(sum / static_cast<double>(frames)));
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

void push_major_triad(VividNoteBuffer& buf, uint8_t root, uint64_t id_base) {
    push_note_on(buf, root, 100, id_base + 0);
    push_note_on(buf, static_cast<uint8_t>(root + 4), 96, id_base + 1);
    push_note_on(buf, static_cast<uint8_t>(root + 7), 92, id_base + 2);
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

    // Test 1: notes_in note-on produces stereo audio.
    {
        std::fprintf(stderr, "\n--- WavetableLayer: notes_in note-on produces stereo ---\n");
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
            {"wavetable_member", 0.0f},
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
        auto baseline_metrics = tc.analyze_output(2);

        push_timbre(midi, /*id=*/4, 1.0f);
        // Run a few blocks so position smoother catches up.
        for (int i = 0; i < 4; ++i) {
            tc.clear_output(); loader.process_audio(inst, &tc.ctx); midi.count = 0;
        }
        auto shifted_metrics = tc.analyze_output(2);

        std::fprintf(stderr,
                     "  baseline brightness: %.4f centroid: %.1fHz | shifted brightness: %.4f centroid: %.1fHz\n",
                     baseline_metrics.spectral_brightness, baseline_metrics.spectral_centroid_hz,
                     shifted_metrics.spectral_brightness, shifted_metrics.spectral_centroid_hz);
        check(std::fabs(shifted_metrics.spectral_brightness - baseline_metrics.spectral_brightness) > 0.0005f ||
                  std::fabs(shifted_metrics.spectral_centroid_hz - baseline_metrics.spectral_centroid_hz) > 20.0f,
              "timbre=1.0 changes spectral brightness vs timbre=0");
        loader.destroy_instance(inst);
    }

    {
        std::fprintf(stderr, "\n--- WavetableLayer: soft member stays darker than bright/formant references ---\n");

        auto render_metrics = [&](float family, float member, float position) {
            PolyTestContext tc;
            tc.set_output_channels(2);
            tc.clear_lane_ports();
            tc.clear_audio_inputs();

            VividNoteBuffer midi{};
            void* midi_inputs[1] = {&midi};
            tc.ctx.custom_inputs = midi_inputs;
            tc.ctx.custom_input_count = 1;

            auto params = make_params(desc, {
                {"amplitude", 0.35f},
                {"unison_voices", 1.0f},
                {"wavetable_family", family},
                {"wavetable_member", member},
                {"position", position},
                {"attack", 0.001f}, {"decay", 0.01f}, {"sustain", 1.0f}, {"release", 0.05f},
                {"pressure_to_amp", 0.0f},
                {"timbre_to_position", 0.0f},
            });
            tc.ctx.param_values = params.data();

            void* inst = loader.create_instance();
            push_note_on(midi, 60, 100, /*id=*/5);
            for (int i = 0; i < 5; ++i) {
                tc.clear_output();
                loader.process_audio(inst, &tc.ctx);
                midi.count = 0;
            }
            auto metrics = tc.analyze_output(2);
            loader.destroy_instance(inst);
            return metrics;
        };

        auto soft = render_metrics(0.0f, 1.0f, 0.24f);
        auto digital = render_metrics(1.0f, 5.0f, 0.24f);
        auto vocal = render_metrics(2.0f, 2.0f, 0.24f);

        std::fprintf(stderr,
                     "  soft brightness=%.4f centroid=%.1fHz | digital brightness=%.4f centroid=%.1fHz | vocal brightness=%.4f centroid=%.1fHz\n",
                     soft.spectral_brightness, soft.spectral_centroid_hz,
                     digital.spectral_brightness, digital.spectral_centroid_hz,
                     vocal.spectral_brightness, vocal.spectral_centroid_hz);
        check(soft.rms > 0.05f, "soft wavetable reference remains audible");
        check(soft.spectral_centroid_hz + 150.0f < digital.spectral_centroid_hz ||
                  soft.spectral_brightness + 0.002f < digital.spectral_brightness,
              "AnalogWarm/Soft stays darker than bright digital glass");
        check(soft.spectral_brightness + 0.004f < vocal.spectral_brightness,
              "AnalogWarm/Soft stays darker than the richer vocal reference");
    }

    {
        std::fprintf(stderr, "\n--- WavetableLayer: default triads stay free of subsonic rumble ---\n");

        auto render_triad = [&](uint8_t root_note, uint64_t id_base) {
            PolyTestContext tc;
            tc.set_output_channels(2);
            tc.clear_lane_ports();
            tc.clear_audio_inputs();

            VividNoteBuffer midi{};
            void* midi_inputs[1] = {&midi};
            tc.ctx.custom_inputs = midi_inputs;
            tc.ctx.custom_input_count = 1;

            auto params = make_params(desc, {
                {"amplitude", 0.30f},
                {"wavetable_family", 0.0f},
                {"wavetable_member", 1.0f},
                {"position", 0.24f},
                {"unison_voices", 1.0f},
                {"attack", 0.008f}, {"decay", 0.08f}, {"sustain", 0.85f}, {"release", 0.08f},
                {"pressure_to_amp", 0.0f},
                {"timbre_to_position", 0.0f},
            });
            tc.ctx.param_values = params.data();

            void* inst = loader.create_instance();
            push_major_triad(midi, root_note, id_base);
            for (int i = 0; i < 4; ++i) {
                tc.clear_output();
                loader.process_audio(inst, &tc.ctx);
                midi.count = 0;
            }

            struct Metrics {
                float rms;
                float mean;
                float low20_rms;
                float low35_rms;
                float low70_rms;
            };
            Metrics metrics{
                stereo_rms(tc.output_buf, kFrames),
                mono_mean(tc.output_buf, kFrames),
                low_band_rms(tc.output_buf, kFrames, static_cast<float>(PolyTestContext::kSampleRate), 20.0f),
                low_band_rms(tc.output_buf, kFrames, static_cast<float>(PolyTestContext::kSampleRate), 35.0f),
                low_band_rms(tc.output_buf, kFrames, static_cast<float>(PolyTestContext::kSampleRate), 70.0f),
            };
            loader.destroy_instance(inst);
            return metrics;
        };

        const auto c = render_triad(60, 100);
        const auto f = render_triad(65, 200);
        const auto g = render_triad(67, 300);

        std::fprintf(stderr,
                     "  C rms=%.4f mean=%.6f low20=%.6f low35=%.6f low70=%.6f | F rms=%.4f mean=%.6f low20=%.6f low35=%.6f low70=%.6f | G rms=%.4f mean=%.6f low20=%.6f low35=%.6f low70=%.6f\n",
                     c.rms, c.mean, c.low20_rms, c.low35_rms, c.low70_rms,
                     f.rms, f.mean, f.low20_rms, f.low35_rms, f.low70_rms,
                     g.rms, g.mean, g.low20_rms, g.low35_rms, g.low70_rms);
        check(std::fabs(c.mean) < 0.01f, "C triad mean stays near zero");
        check(std::fabs(f.mean) < 0.01f, "F triad mean stays near zero");
        check(std::fabs(g.mean) < 0.01f, "G triad mean stays near zero");
        check(c.low35_rms < 0.022f, "C triad keeps subsonic low-band energy under control");
        check(f.low35_rms < 0.022f, "F triad keeps subsonic low-band energy under control");
        check(g.low35_rms < 0.022f, "G triad keeps subsonic low-band energy under control");
    }

    {
        std::fprintf(stderr, "\n--- WavetableLayer: chord progression transitions avoid low-end wobble ---\n");

        auto progression_metrics = [&](float member, float position, float attack, float decay,
                                       float sustain, float release) {
            PolyTestContext tc;
            tc.set_output_channels(2);
            tc.clear_lane_ports();
            tc.clear_audio_inputs();

            VividNoteBuffer midi{};
            void* midi_inputs[1] = {&midi};
            tc.ctx.custom_inputs = midi_inputs;
            tc.ctx.custom_input_count = 1;

            auto params = make_params(desc, {
                {"amplitude", 0.30f},
                {"wavetable_family", 0.0f},
                {"wavetable_member", member},
                {"position", position},
                {"unison_voices", 1.0f},
                {"attack", attack}, {"decay", decay}, {"sustain", sustain}, {"release", release},
                {"pressure_to_amp", 0.0f},
                {"timbre_to_position", 0.0f},
            });
            tc.ctx.param_values = params.data();

            auto transition_low35 = [&](void* inst,
                                        std::initializer_list<uint64_t> offs,
                                        uint8_t root,
                                        uint64_t id_base) {
                for (uint64_t id : offs) push_note_off(midi, id);
                push_major_triad(midi, root, id_base);
                tc.clear_output();
                loader.process_audio(inst, &tc.ctx);
                midi.count = 0;
                return low_band_rms(tc.output_buf, kFrames,
                                    static_cast<float>(PolyTestContext::kSampleRate), 35.0f);
            };

            void* inst = loader.create_instance();
            std::array<float, 4> result = {
                transition_low35(inst, {}, 60, 400),
                transition_low35(inst, {400, 401, 402}, 65, 500),
                transition_low35(inst, {500, 501, 502}, 67, 600),
                transition_low35(inst, {600, 601, 602}, 60, 700),
            };
            loader.destroy_instance(inst);
            return result;
        };

        const auto legacy_prog = progression_metrics(0.0f, 0.0f, 0.005f, 0.1f, 0.8f, 0.2f);
        const auto default_prog = progression_metrics(1.0f, 0.24f, 0.008f, 0.08f, 0.85f, 0.08f);

        std::fprintf(stderr,
                     "  legacy  low35: C1=%.6f F=%.6f G=%.6f C2=%.6f\n",
                     legacy_prog[0], legacy_prog[1], legacy_prog[2], legacy_prog[3]);
        std::fprintf(stderr,
                     "  default low35: C1=%.6f F=%.6f G=%.6f C2=%.6f\n",
                     default_prog[0], default_prog[1], default_prog[2], default_prog[3]);
        check(default_prog[2] + 0.002f < legacy_prog[2],
              "new default baseline reduces G-step low-end wobble vs legacy core");
        check(default_prog[3] + 0.012f < legacy_prog[3],
              "new default baseline reduces return-to-C wobble vs legacy core");
    }

    std::fprintf(stderr, "\n%s (%d failures)\n",
                 failures == 0 ? "PASSED" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
