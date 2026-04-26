#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "runtime/graph/graph.h"
#include "runtime/graph/subgraph_module.h"
#include "test_support.h"

namespace fs = std::filesystem;

static int failures = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
        ++failures;
    } else {
        std::fprintf(stderr, "PASS: %s\n", msg.c_str());
    }
}

static std::string read_file(const fs::path& path) {
    std::ifstream ifs(path);
    if (!ifs) return {};
    return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

static float mono_rms(const float* data, int count) {
    double sum = 0.0;
    for (int i = 0; i < count; ++i) sum += static_cast<double>(data[i]) * static_cast<double>(data[i]);
    return std::sqrt(static_cast<float>(sum / static_cast<double>(count)));
}

static float average_abs_diff(const float* a, const float* b, int count) {
    double sum = 0.0;
    for (int i = 0; i < count; ++i) sum += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    return static_cast<float>(sum / static_cast<double>(count));
}

static int find_param_index(const VividOperatorDescriptor* desc, const char* name) {
    for (uint32_t i = 0; i < desc->param_count; ++i) {
        if (std::strcmp(desc->params[i].name, name) == 0) return static_cast<int>(i);
    }
    return -1;
}

static int find_port_index(const VividOperatorDescriptor* desc, const char* name) {
    for (uint32_t i = 0; i < desc->port_count; ++i) {
        if (std::strcmp(desc->ports[i].name, name) == 0) return static_cast<int>(i);
    }
    return -1;
}

static std::vector<float> default_params(const VividOperatorDescriptor* desc) {
    std::vector<float> params(desc->param_count, 0.0f);
    for (uint32_t i = 0; i < desc->param_count; ++i)
        params[i] = desc->params[i].default_value;
    return params;
}

static bool has_node_type(const vivid::Graph& graph, const std::string& type_name) {
    for (const auto& node : graph.nodes()) {
        if (node.type == type_name) return true;
    }
    return false;
}

static bool has_connection(const vivid::Graph& graph,
                           const std::string& from_node,
                           const std::string& from_port,
                           const std::string& to_node,
                           const std::string& to_port) {
    for (const auto& conn : graph.connections()) {
        if (conn.from_node == from_node && conn.from_port == from_port &&
            conn.to_node == to_node && conn.to_port == to_port) {
            return true;
        }
    }
    return false;
}

static bool flatten_layerpad_demo(const fs::path& authored_graph_path,
                                  vivid::Graph& authored_graph,
                                  vivid::Graph& flattened_graph) {
    vivid::SubgraphModuleRegistry registry;
    if (!registry.load("modules/layer_pad.vivid-module.json")) {
        std::fprintf(stderr, "failed to load LayerPad module for flattening\n");
        return false;
    }

    const std::string graph_json = read_file(authored_graph_path);
    if (graph_json.empty()) {
        std::fprintf(stderr, "failed to read %s\n", authored_graph_path.c_str());
        return false;
    }

    if (!authored_graph.load_from_string(graph_json.c_str(), graph_json.size())) {
        std::fprintf(stderr, "failed to parse %s\n", authored_graph_path.c_str());
        return false;
    }

    auto flattened = vivid::flatten_subgraphs(authored_graph, registry);
    flattened_graph = std::move(flattened.graph);
    return true;
}

static bool render_layerpad_reference_audio(const fs::path& package_build_dir,
                                            std::vector<float>& stereo_output) {
    MiniLoader loader;
    const auto op_path = package_build_dir / ("wavetable_layer" VIVID_PLUGIN_SUFFIX_STR);
    if (!loader.load(op_path.c_str())) {
        std::fprintf(stderr, "failed to load %s\n", op_path.c_str());
        return false;
    }

    const auto* desc = loader.descriptor();
    if (!desc) return false;

    auto params = default_params(desc);
    auto set_param = [&](const char* name, float value) {
        int idx = find_param_index(desc, name);
        if (idx >= 0) params[static_cast<size_t>(idx)] = value;
    };

    set_param("amplitude", 0.22f);
    set_param("position", 0.35f);
    set_param("wavetable_family", 2.0f);
    set_param("wavetable_member", 2.0f);
    set_param("unison_voices", 4.0f);
    set_param("unison_spread", 16.0f);
    set_param("unison_stereo", 0.78f);

    // Phase 3 PR3: lane inputs and voice_gain_audio retired. Drive the
    // reference voices via MIDI; pin attack near zero and sustain at 1.0
    // so the synth's internal ADSR doesn't shape the buffer (test compares
    // unenveloped multi-voice render).
    auto pin_envelope = [&](float attack, float sustain, float release) {
        for (uint32_t p = 0; p < desc->param_count; ++p) {
            const char* name = desc->params[p].name;
            if (std::strcmp(name, "attack") == 0)  params[p] = attack;
            if (std::strcmp(name, "sustain") == 0) params[p] = sustain;
            if (std::strcmp(name, "release") == 0) params[p] = release;
        }
    };
    pin_envelope(0.001f, 1.0f, 0.001f);

    void* inst = loader.create_instance();
    if (!inst) return false;

    PolyTestContext tc;
    tc.set_output_channels(2);
    tc.ctx.param_values = params.data();
    tc.clear_lane_ports();
    tc.clear_audio_inputs();
    tc.clear_notes();
    tc.push_note_on(60, 0.9f);   // C4
    tc.push_note_on(64, 0.85f);  // E4
    tc.push_note_on(67, 0.8f);   // G4
    tc.push_note_on(72, 0.78f);  // C5

    for (int block = 0; block < 6; ++block) {
        tc.clear_output();
        loader.process_audio(inst, &tc.ctx);
        tc.ctx.time += static_cast<double>(tc.kFrames) / tc.kSampleRate;
        tc.ctx.frame++;
    }

    stereo_output.assign(tc.output_buf, tc.output_buf + 2 * PolyTestContext::kFrames);
    loader.destroy_instance(inst);
    return true;
}

static bool render_filter_reference_audio(const fs::path& core_build_dir,
                                          const std::vector<float>& stereo_input,
                                          std::vector<float>& filtered_output) {
    MiniLoader loader;
    const auto op_path = core_build_dir / ("filter" VIVID_PLUGIN_SUFFIX_STR);
    if (!loader.load(op_path.c_str())) {
        std::fprintf(stderr, "failed to load %s\n", op_path.c_str());
        return false;
    }

    const auto* desc = loader.descriptor();
    if (!desc) return false;

    auto params = default_params(desc);
    auto set_param = [&](const char* name, float value) {
        int idx = find_param_index(desc, name);
        if (idx >= 0) params[static_cast<size_t>(idx)] = value;
    };
    set_param("mode", 1.0f);
    set_param("cutoff", 1800.0f);
    set_param("resonance", 0.15f);

    std::vector<float> mono_input(static_cast<size_t>(PolyTestContext::kFrames), 0.0f);
    const float* left = stereo_input.data();
    const float* right = stereo_input.data() + PolyTestContext::kFrames;
    for (int i = 0; i < PolyTestContext::kFrames; ++i)
        mono_input[static_cast<size_t>(i)] = 0.5f * (left[i] + right[i]);

    void* inst = loader.create_instance();
    if (!inst) return false;

    PolyTestContext tc;
    tc.set_output_channels(1);
    tc.ctx.param_values = params.data();
    tc.clear_audio_inputs();
    tc.clear_lane_ports();
    tc.bind_audio_input(0, mono_input.data(), 1);

    for (int block = 0; block < 4; ++block) {
        tc.clear_output();
        loader.process_audio(inst, &tc.ctx);
        tc.ctx.time += static_cast<double>(tc.kFrames) / tc.kSampleRate;
        tc.ctx.frame++;
    }

    filtered_output.assign(tc.output_buf, tc.output_buf + PolyTestContext::kFrames);
    loader.destroy_instance(inst);
    return true;
}

int main() {
    const fs::path package_root = fs::current_path();
    const fs::path package_build_dir = fs::path(VIVID_PACKAGE_BUILD_DIR_STR);
    const fs::path core_build_dir = fs::path(VIVID_CORE_BUILD_DIR_STR);

    check(fs::exists(package_root / "modules" / "layer_pad.vivid-module.json"), "LayerPad module file exists");
    check(fs::exists(package_build_dir / ("wavetable_layer" VIVID_PLUGIN_SUFFIX_STR)),
          "WavetableLayer package operator build artifact exists");
    check(fs::exists(core_build_dir / ("filter" VIVID_PLUGIN_SUFFIX_STR)),
          "core Filter build artifact exists");

    const std::string manifest = read_file(package_root / "vivid-package.json");
    check(manifest.find("\"graphs/core/wavetable_layer_stress.json\"") != std::string::npos,
          "stress graph remains on the package smoke surface");

    const fs::path pad_demo = package_root / "graphs" / "core" / "wavetable_layer_pad_demo.json";
    const fs::path filter_demo = package_root / "graphs" / "core" / "wavetable_layer_filter_integration.json";
    const fs::path stress_demo = package_root / "graphs" / "core" / "wavetable_layer_stress.json";

    vivid::Graph authored_pad_graph;
    vivid::Graph flattened_pad_graph;
    check(flatten_layerpad_demo(pad_demo, authored_pad_graph, flattened_pad_graph),
          "LayerPad demo graph loads and flattens through the shipped module definition");
    check(has_node_type(authored_pad_graph, "LayerPad"), "pad demo authors against the LayerPad module");
    check(!has_node_type(flattened_pad_graph, "LayerPad"), "flattened pad demo removes the module wrapper node");
    check(has_node_type(flattened_pad_graph, "WavetableLayer"),
          "flattened pad demo contains the production WavetableLayer operator");
    check(has_node_type(flattened_pad_graph, "audio_out"),
          "flattened pad demo still terminates in audio_out");

    const std::string filter_json = read_file(filter_demo);
    vivid::Graph filter_graph;
    check(!filter_json.empty() && filter_graph.load_from_string(filter_json.c_str(), filter_json.size()),
          "filter integration graph loads from package content");
    check(has_node_type(filter_graph, "WavetableLayer"), "filter integration graph uses WavetableLayer");
    check(has_node_type(filter_graph, "Filter"), "filter integration graph uses external Filter");
    check(has_connection(filter_graph, "wt", "output", "filter", "input"),
          "filter integration graph routes WavetableLayer output into Filter");
    check(has_connection(filter_graph, "filter", "output", "out", "input"),
          "filter integration graph routes Filter output into audio_out");

    std::vector<float> layer_output;
    check(render_layerpad_reference_audio(package_build_dir, layer_output),
          "LayerPad reference voice render succeeds with the packaged WavetableLayer");
    if (!layer_output.empty()) {
        const float* left = layer_output.data();
        const float* right = layer_output.data() + PolyTestContext::kFrames;
        float left_rms = mono_rms(left, PolyTestContext::kFrames);
        float right_rms = mono_rms(right, PolyTestContext::kFrames);
        float stereo_delta = average_abs_diff(left, right, PolyTestContext::kFrames);
        check(left_rms > 0.005f && right_rms > 0.005f,
              "wavetable_layer_pad_demo reference path is non-silent on both stereo channels");
        check(stereo_delta > 1.0e-4f,
              "wavetable_layer_pad_demo reference path preserves audible stereo spread");
    }

    std::vector<float> filtered_output;
    check(!layer_output.empty() &&
              render_filter_reference_audio(core_build_dir, layer_output, filtered_output),
          "filter integration reference chain renders through the external mono Filter");
    if (!filtered_output.empty()) {
        std::vector<float> mono_input(static_cast<size_t>(PolyTestContext::kFrames), 0.0f);
        const float* left = layer_output.data();
        const float* right = layer_output.data() + PolyTestContext::kFrames;
        for (int i = 0; i < PolyTestContext::kFrames; ++i)
            mono_input[static_cast<size_t>(i)] = 0.5f * (left[i] + right[i]);
        float filtered_rms = mono_rms(filtered_output.data(), PolyTestContext::kFrames);
        float filtered_delta = average_abs_diff(filtered_output.data(), mono_input.data(),
                                                PolyTestContext::kFrames);
        check(filtered_rms > 0.001f,
              "wavetable_layer_filter_integration reference path produces non-silent filtered output");
        check(filtered_delta > 1.0e-4f,
              "wavetable_layer_filter_integration reference path materially changes the mono signal");
    }

    const std::string stress_json = read_file(stress_demo);
    vivid::Graph stress_graph;
    check(!stress_json.empty() && stress_graph.load_from_string(stress_json.c_str(), stress_json.size()),
          "wavetable_layer_stress remains a loadable smoke graph in Phase 4");

    if (failures == 0) {
        std::printf("LayerPad reference graph checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "LayerPad reference graph checks failed: %d\n", failures);
    return 1;
}
