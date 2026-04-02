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
        "\"poly_voice_allocator\"",
        "\"wavetable_osc\"",
        "\"voice_mixer\"",
        "\"sub_osc\"",
        "\"analog_osc\"",
        "\"noise_layer\"",
        "\"tests/cpp/test_audio_correctness.cpp\"",
        "\"tests/cpp/test_graph_envelope_wiring.cpp\""
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
