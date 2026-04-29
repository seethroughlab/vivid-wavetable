// Integration test: no NEW checked-in graph adds a WavetableOsc node.
//
// Phase 3 of the simplification plan deprecates WavetableOsc in favor of
// WavetableLayer. WavetableOsc still works and existing references are
// grandfathered (allowlist below), but new graphs should use WavetableLayer
// for the production path. The exceptions stay grandfathered because they
// rely on Osc-only features (interaction_mode FM/PM/RM/AM cross-modulation
// or per-voice voices_out routing) that WavetableLayer doesn't yet expose.
//
// This test scans every *.json under graphs/, modules/, archive/graphs/
// and asserts that any node typed "WavetableOsc" appears in a file from
// the allowlist below. To migrate a grandfathered file: replace the
// WavetableOsc node with WavetableLayer, verify the patch sounds right,
// then remove the file from the allowlist here.
//
// To add a NEW use of WavetableOsc (e.g. an advanced demo that needs
// FM cross-modulation): add the file to the allowlist with a comment
// explaining which Osc-only feature it uses. Don't disable the test.

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

// Files that are allowed to reference WavetableOsc. Each one uses an
// Osc-only feature that WavetableLayer doesn't expose yet:
//   - interaction_mode (FM/PM/RM/AM cross-modulation via mod_input)
//   - voices_out per-voice audio routing
//   - audio-rate self-feedback for position/pitch modulation
const std::unordered_set<std::string> GRANDFATHERED = {
    // Uses interaction_mode (FM/PM/RM/AM) — no migration target until
    // WavetableLayer gains an interaction surface
    "graphs/presets/fm_glass_keys.json",
    "graphs/presets/growl_crossmod_bass.json",
    "graphs/presets/controlled_metallic_lead.json",
    "graphs/presets/hybrid_motion_arp.json",
    "graphs/presets/spectral_interaction_texture.json",
    "modules/glass_interaction_keys.vivid-module.json",
    // Uses audio-rate self-feedback (osc/output → osc/position_mod_audio).
    // Migratable to WavetableLayer in principle, but the tonal character
    // depends on Osc's exact feedback path; defer migration until a
    // listening review confirms parity.
    "graphs/presets/orbit_drone.json",
    // Legacy archived stress fixture from before the Layer/Osc split. Not
    // runnable on the current Vivid (uses retired operator names like
    // EnvelopeAu). Kept for git history; never fixed.
    "archive/graphs/wavetable_osc_stress.json",
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

// Returns the relative path string in unix form for stable allowlist matching.
std::string rel_str(const fs::path& path) {
    std::string s = path.lexically_normal().generic_string();
    // Strip any leading "./"
    if (s.rfind("./", 0) == 0) s = s.substr(2);
    return s;
}

bool references_wavetable_osc(const fs::path& path) {
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    json doc;
    try {
        doc = json::parse(ss.str());
    } catch (const std::exception&) {
        return false;
    }
    auto nodes = doc.value("nodes", json::object());
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        if (it.value().value("type", "") == "WavetableOsc") return true;
    }
    return false;
}

}  // namespace

int main() {
    std::fprintf(stderr, "\n--- No NEW graph references WavetableOsc ---\n");

    auto jsons = find_jsons();
    if (jsons.empty()) {
        std::fprintf(stderr, "FATAL: no graph JSON files found (working dir = %s)\n",
                     fs::current_path().string().c_str());
        return 1;
    }

    int unauthorized = 0;
    int grandfathered_seen = 0;
    for (const auto& p : jsons) {
        if (!references_wavetable_osc(p)) continue;
        std::string rel = rel_str(p);
        if (GRANDFATHERED.count(rel)) {
            ++grandfathered_seen;
        } else {
            check(false, rel + " adds a NEW WavetableOsc reference. "
                                "Use WavetableLayer instead, or add this file to the "
                                "GRANDFATHERED allowlist with a comment explaining "
                                "which Osc-only feature it requires.");
            ++unauthorized;
        }
    }

    std::fprintf(stderr, "Scanned %zu JSON files; %d grandfathered WavetableOsc users; %d unauthorized\n",
                 jsons.size(), grandfathered_seen, unauthorized);

    return failures == 0 ? 0 : 1;
}
