// Integration test: no checked-in graph references VoiceAllocator.
//
// PR4 of the Phase 3 plan deletes the VoiceAllocator operator + the synth
// lane-note inputs. This test stands guard so a future graph edit can't
// silently re-introduce a VoiceAllocator-era pattern.
//
// Scans every *.json under graphs/, modules/, archive/graphs/ and asserts:
//   1. no node carries type == "VoiceAllocator" or "PolyVoiceAllocator"
//   2. no synth port matches the removed lane-input names (frequencies,
//      gates, velocities, lane_ids — synths get notes_in instead)
//
// Test runs from CMAKE_SOURCE_DIR so relative paths resolve.

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

static int failures = 0;
static void check(bool cond, const std::string& msg) {
    if (!cond) { std::fprintf(stderr, "  FAIL: %s\n", msg.c_str()); ++failures; }
    else       { std::fprintf(stderr, "  PASS: %s\n", msg.c_str()); }
}

namespace {

// Ports that PR3 deletes from voice synths. A graph wiring any of these
// is using the legacy lane-driven path (which goes away).
const std::unordered_set<std::string> REMOVED_SYNTH_LANE_PORTS = {
    "frequencies", "gates", "velocities", "lane_ids",
    "pitch_mod", "position_mod", "warp_mod",
};

const std::unordered_set<std::string> SYNTH_TYPES = {
    "WavetableOsc", "AnalogOsc", "WavetableLayer", "SubOsc", "NoiseLayer",
    "FmSynth", "Sampler", "SP404", "Slicer",
};

const std::unordered_set<std::string> ALLOCATOR_TYPES = {
    "VoiceAllocator", "PolyVoiceAllocator",
};

std::vector<fs::path> find_jsons() {
    std::vector<fs::path> out;
    for (const char* dir : {"graphs", "modules", "archive/graphs"}) {
        if (!fs::is_directory(dir)) continue;
        for (auto it = fs::recursive_directory_iterator(dir);
             it != fs::recursive_directory_iterator(); ++it) {
            if (it->is_regular_file() && it->path().extension() == ".json")
                out.push_back(it->path());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

void scan(const fs::path& path) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "  SKIP: cannot open %s\n", path.string().c_str());
        return;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    json doc;
    try {
        doc = json::parse(ss.str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  WARN: %s parse failed: %s\n",
                     path.string().c_str(), e.what());
        return;
    }

    auto nodes = doc.value("nodes", json::object());

    // Check 1: no VoiceAllocator nodes.
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        std::string type = it.value().value("type", "");
        if (ALLOCATOR_TYPES.count(type)) {
            check(false, path.string() + ": node '" + it.key() +
                          "' is type '" + type + "' (forbidden)");
            return;
        }
    }

    // Check 2: no synth wires consume a removed lane-input port.
    for (const auto& conn : doc.value("connections", json::array())) {
        std::string to = conn.value("to", "");
        auto slash = to.find('/');
        if (slash == std::string::npos) continue;
        std::string node = to.substr(0, slash);
        std::string port = to.substr(slash + 1);
        auto nit = nodes.find(node);
        if (nit == nodes.end()) continue;
        std::string ntype = nit->value("type", "");
        if (SYNTH_TYPES.count(ntype) && REMOVED_SYNTH_LANE_PORTS.count(port)) {
            check(false, path.string() + ": " + to + " (" + ntype +
                          ") wires removed lane-input port");
        }
    }
}

}  // namespace

int main() {
    std::fprintf(stderr, "\n--- No graph references VoiceAllocator ---\n");

    auto jsons = find_jsons();
    if (jsons.empty()) {
        std::fprintf(stderr, "FATAL: no graph JSON files found (working dir = %s)\n",
                     fs::current_path().string().c_str());
        return 1;
    }
    std::fprintf(stderr, "Scanning %zu JSON files\n", jsons.size());

    for (const auto& p : jsons) scan(p);

    if (failures == 0)
        std::fprintf(stderr, "\nPASS: no checked-in graph references "
                             "VoiceAllocator or removed synth lane inputs\n");
    else
        std::fprintf(stderr, "\nFAIL: %d offending file(s)\n", failures);

    return failures == 0 ? 0 : 1;
}
