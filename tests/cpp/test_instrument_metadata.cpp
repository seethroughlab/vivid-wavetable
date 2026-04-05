#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string read_file(const std::filesystem::path& path) {
    std::ifstream ifs(path);
    return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

int failures = 0;
int instruments_found = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        ++failures;
    } else {
        std::cerr << "PASS: " << msg << "\n";
    }
}

void validate_instrument(const std::filesystem::path& path) {
    const std::string json = read_file(path);
    const std::string name = path.filename().string();

    // Only validate graphs with content_kind = instrument
    if (!contains(json, "\"content_kind\"") || !contains(json, "\"instrument\""))
        return;

    ++instruments_found;

    check(contains(json, "\"category\""),
          name + ": has category");
    check(contains(json, "\"family\""),
          name + ": has family");
    check(contains(json, "\"role\""),
          name + ": has role");
    check(contains(json, "\"playability\""),
          name + ": has playability");
    check(contains(json, "\"preview_controls\""),
          name + ": has preview_controls");

    // Validate that preview_controls is non-empty (has at least one param entry)
    check(contains(json, "\"param\""),
          name + ": preview_controls has at least one param entry");

    // Standard meta fields
    check(contains(json, "\"id\""),
          name + ": has id");
    check(contains(json, "\"title\""),
          name + ": has title");
    check(contains(json, "\"description\""),
          name + ": has description");
    check(contains(json, "\"tags\""),
          name + ": has tags");
    check(contains(json, "\"requires_packages\""),
          name + ": has requires_packages");

    // If playability is midi, require MidiInput
    if (contains(json, "\"midi\"") && contains(json, "\"playability\"")) {
        check(contains(json, "\"MidiInput\""),
              name + ": midi playability requires MidiInput node");
    }
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    // Scan both graphs/presets/ and graphs/core/ from the test working directory
    std::vector<fs::path> search_dirs = {
        "../graphs/presets",
        "../graphs/core"
    };

    for (const auto& dir : search_dirs) {
        if (!fs::exists(dir)) continue;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() != ".json") continue;
            validate_instrument(entry.path());
        }
    }

    check(instruments_found >= 6,
          "at least 6 instrument graphs found (got " + std::to_string(instruments_found) + ")");

    std::cerr << "\n" << instruments_found << " instrument graph(s) validated, "
              << failures << " failure(s)\n";
    return failures > 0 ? 1 : 0;
}
