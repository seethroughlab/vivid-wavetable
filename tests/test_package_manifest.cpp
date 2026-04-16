#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static bool contains(const std::string& s, const std::string& needle) {
    return s.find(needle) != std::string::npos;
}

static bool lacks(const std::string& s, const std::string& needle) {
    return s.find(needle) == std::string::npos;
}

int main() {
    std::ifstream ifs("../vivid-package.json");
    if (!ifs) {
        std::cerr << "failed to open ../vivid-package.json\n";
        return 1;
    }

    std::string json((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    const std::vector<std::string> required = {
        "\"name\": \"vivid-wavetable\"",
        "\"version\": \"0.2.0\"",
        "\"graphs/core/wavetable_modular_demo.json\"",
        "\"graphs/core/wavetable_asset_smoke.json\"",
        "\"poly_voice_allocator\"",
        "\"wavetable_osc\"",
        "\"wavetable_layer\"",
        "\"voice_mixer\"",
        "\"voice_drive\"",
        "\"sub_osc\"",
        "\"analog_osc\"",
        "\"noise_layer\"",
        "\"modules\"",
        "\"modules/hybrid_keys.vivid-module.json\"",
        "\"modules/glass_interaction_keys.vivid-module.json\"",
        "\"modules/dual_wavetable_pad.vivid-module.json\"",
        "\"modules/sub_air_pad.vivid-module.json\"",
        "\"modules/layer_pad.vivid-module.json\"",
        "\"graphs/core/wavetable_layer_pad_demo.json\"",
        "\"graphs/core/wavetable_layer_stress.json\"",
        "\"graphs/core/wavetable_layer_filter_integration.json\"",
        "\"tests/cpp/test_audio_correctness.cpp\"",
        "\"tests/cpp/test_graph_envelope_wiring.cpp\"",
        "\"tests/cpp/test_module_surface_contract.cpp\"",
        "\"tests/cpp/test_layerpad_reference_graphs.cpp\"",
        "\"assets\"",
        "\"wavetables\"",
        "\"assets/wavetables\""
    };

    for (const auto& needle : required) {
        if (!contains(json, needle)) {
            std::cerr << "missing expected entry: " << needle << "\n";
            return 1;
        }
    }

    const std::vector<std::string> forbidden = {
        "WavetableSynth",
        "wavetable_synth",
        "archive/",
        "graphs/extended/"
    };

    for (const auto& needle : forbidden) {
        if (!lacks(json, needle)) {
            std::cerr << "found legacy or non-active-surface entry: " << needle << "\n";
            return 1;
        }
    }

    std::cout << "manifest active-surface check passed\n";
    return 0;
}
