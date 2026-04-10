// Surface contract test for WavetableLayer.
// Verifies the frozen public interface: descriptor name, param count/names,
// port count/names/channels, lane behavior, and Phase 2 silence contract.

#include "test_support.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++failures;
    }
}

#ifndef VIVID_PLUGIN_SUFFIX_STR
#if defined(_WIN32)
#define VIVID_PLUGIN_SUFFIX_STR ".dll"
#elif defined(__APPLE__)
#define VIVID_PLUGIN_SUFFIX_STR ".dylib"
#else
#define VIVID_PLUGIN_SUFFIX_STR ".so"
#endif
#endif

int main() {
    MiniLoader loader;
    std::string lib_path = std::string("./wavetable_layer") + VIVID_PLUGIN_SUFFIX_STR;
    if (!loader.load(lib_path.c_str())) {
        std::fprintf(stderr, "FAIL: could not load %s\n", lib_path.c_str());
        return 1;
    }

    // --- Descriptor checks ---
    const VividOperatorDescriptor* desc = loader.descriptor();
    check(desc != nullptr, "descriptor is non-null");
    if (!desc) return 1;

    check(std::string(desc->name) == "WavetableLayer", "name is WavetableLayer");
    check(desc->has_process_audio == 1, "has_process_audio");
    check(desc->has_process_frame == 0, "no process_frame");
    check(desc->has_process_gpu == 0, "no process_gpu");
    check(desc->lane_behavior == VIVID_LANE_REDUCTION, "lane_behavior is REDUCTION");

    // --- Param checks ---
    const std::vector<std::string> expected_params = {
        "wavetable_source", "wavetable_family", "wavetable_member", "wav_file",
        "position", "amplitude",
        "warp_mode", "warp_amount",
        "position_smooth_ms", "warp_smooth_ms",
        "drift_amount", "drift_rate_hz",
        "phase_reset_mode", "start_phase", "phase_random", "stereo_phase_offset",
        "unison_voices", "unison_spread", "unison_stereo", "unison_spread_mode",
        "detune", "portamento"
    };

    check(desc->param_count == expected_params.size(),
          "param count matches frozen list");
    for (uint32_t i = 0; i < desc->param_count && i < expected_params.size(); ++i) {
        if (std::string(desc->params[i].name) != expected_params[i]) {
            std::fprintf(stderr, "FAIL: param[%u] name is '%s', expected '%s'\n",
                         i, desc->params[i].name, expected_params[i].c_str());
            ++failures;
        }
    }

    int warp_mode_param = -1;
    for (uint32_t i = 0; i < desc->param_count; ++i) {
        if (std::string(desc->params[i].name) == "warp_mode") {
            warp_mode_param = static_cast<int>(i);
            break;
        }
    }
    check(warp_mode_param >= 0, "warp_mode param exists");
    if (warp_mode_param >= 0) {
        const auto& param = desc->params[warp_mode_param];
        const std::vector<std::string> expected_modes = {
            "None", "Sync", "BendPlus", "BendMinus", "Mirror", "Asym", "Quantize", "Flip"
        };
        check(param.choice_count == expected_modes.size(), "warp_mode choice count excludes FM");
        for (uint32_t i = 0; i < param.choice_count && i < expected_modes.size(); ++i) {
            if (std::string(param.choice_labels[i]) != expected_modes[i]) {
                std::fprintf(stderr, "FAIL: warp_mode choice[%u] is '%s', expected '%s'\n",
                             i, param.choice_labels[i], expected_modes[i].c_str());
                ++failures;
            }
        }
        bool found_fm = false;
        for (uint32_t i = 0; i < param.choice_count; ++i) {
            if (std::string(param.choice_labels[i]) == "FM") {
                found_fm = true;
                break;
            }
        }
        check(!found_fm, "warp_mode choices omit legacy FM feedback warp");
    }

    // --- Port checks ---
    const std::vector<std::string> expected_ports = {
        "frequencies", "gates", "velocities", "lane_ids",
        "pitch_mod", "position_mod", "warp_mod",
        "pitch_mod_audio", "position_mod_audio", "warp_mod_audio",
        "voice_gain_audio",
        "output"
    };

    check(desc->port_count == expected_ports.size(),
          "port count matches frozen list");
    for (uint32_t i = 0; i < desc->port_count && i < expected_ports.size(); ++i) {
        if (std::string(desc->ports[i].name) != expected_ports[i]) {
            std::fprintf(stderr, "FAIL: port[%u] name is '%s', expected '%s'\n",
                         i, desc->ports[i].name, expected_ports[i].c_str());
            ++failures;
        }
    }

    // Output port must be stereo (2 channels)
    if (desc->port_count == expected_ports.size()) {
        uint32_t output_idx = desc->port_count - 1;
        check(desc->ports[output_idx].channels == 2,
              "output port declares 2 channels (stereo)");
        check(desc->ports[output_idx].direction == VIVID_PORT_OUTPUT,
              "output port direction is OUTPUT");
    }

    // Lane-array ports (indices 0-6) should be LANE_ARRAY inputs
    for (uint32_t i = 0; i < 7 && i < desc->port_count; ++i) {
        check(desc->ports[i].type == VIVID_PORT_LANE_ARRAY,
              "lane port type is LANE_ARRAY");
        check(desc->ports[i].direction == VIVID_PORT_INPUT,
              "lane port direction is INPUT");
    }

    // Audio-rate modulation ports (indices 7-10) should be AUDIO_BUFFER inputs
    for (uint32_t i = 7; i < 11 && i < desc->port_count; ++i) {
        check(desc->ports[i].type == VIVID_PORT_AUDIO_BUFFER,
              "audio mod port type is AUDIO_BUFFER");
        check(desc->ports[i].direction == VIVID_PORT_INPUT,
              "audio mod port direction is INPUT");
        check(desc->ports[i].channels == 0,
              "audio mod port channels is 0 (auto)");
    }

    // --- Audio output contract ---
    void* inst = loader.create_instance();
    check(inst != nullptr, "create_instance succeeds");
    if (inst) {
        PolyTestContext tc;
        tc.set_output_channels(2);
        tc.setup_wavetable_layer_voice(440.0f);

        // Sync params from descriptor defaults
        std::vector<float> param_vals(desc->param_count, 0.0f);
        for (uint32_t i = 0; i < desc->param_count; ++i)
            param_vals[i] = desc->params[i].default_value;
        tc.ctx.param_values = param_vals.data();

        loader.process_audio(inst, &tc.ctx);

        // Verify stereo output is non-silent (renderer produces sound)
        bool has_left = false, has_right = false;
        for (uint32_t i = 0; i < tc.kFrames; ++i) {
            if (tc.output_buf[i] != 0.0f) has_left = true;
            if (tc.output_buf[tc.kFrames + i] != 0.0f) has_right = true;
        }
        check(has_left, "left channel is non-silent");
        check(has_right, "right channel is non-silent");

        loader.destroy_instance(inst);
    }

    if (failures == 0) {
        std::printf("WavetableLayer surface contract: PASS (%zu params, %zu ports)\n",
                    expected_params.size(), expected_ports.size());
    } else {
        std::fprintf(stderr, "WavetableLayer surface contract: %d FAILURES\n", failures);
    }
    return failures > 0 ? 1 : 0;
}
