#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static bool contains(const std::string& s, const std::string& needle) {
    return s.find(needle) != std::string::npos;
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
        "\"wavetable_synth\""
    };

    for (const auto& needle : required) {
        if (!contains(json, needle)) {
            std::cerr << "missing expected entry: " << needle << "\n";
            return 1;
        }
    }

    std::cout << "manifest smoke check passed\n";
    return 0;
}
