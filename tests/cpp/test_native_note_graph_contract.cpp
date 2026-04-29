#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

int failures = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        std::fprintf(stderr, "  FAIL: %s\n", msg.c_str());
        ++failures;
    } else {
        std::fprintf(stderr, "  PASS: %s\n", msg.c_str());
    }
}

json load_json(const fs::path& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path.string());
    json doc;
    f >> doc;
    return doc;
}

std::unordered_map<std::string, std::string> node_types(const json& doc) {
    std::unordered_map<std::string, std::string> out;
    for (auto it = doc["nodes"].begin(); it != doc["nodes"].end(); ++it) {
        out[it.key()] = it.value().value("type", "");
    }
    return out;
}

bool has_connection(const json& doc, const std::string& from, const std::string& to) {
    for (const auto& conn : doc.value("connections", json::array())) {
        if (conn.value("from", "") == from && conn.value("to", "") == to) return true;
    }
    return false;
}

std::set<std::string> incoming_sources(const json& doc, const std::string& node, const std::string& port) {
    const std::string target = node + "/" + port;
    std::set<std::string> out;
    for (const auto& conn : doc.value("connections", json::array())) {
        if (conn.value("to", "") == target) out.insert(conn.value("from", ""));
    }
    return out;
}

void require_connection(const json& doc, const fs::path& path, const std::string& from, const std::string& to) {
    check(has_connection(doc, from, to), path.string() + ": missing " + from + " -> " + to);
}

void require_node_type(const std::unordered_map<std::string, std::string>& types,
                       const fs::path& path, const std::string& node, const std::string& type) {
    auto it = types.find(node);
    check(it != types.end(), path.string() + ": missing node '" + node + "'");
    if (it != types.end()) {
        check(it->second == type,
              path.string() + ": node '" + node + "' is '" + it->second +
              "', expected '" + type + "'");
    }
}

void check_graph_metadata() {
    const fs::path graphs_dir = "graphs";
    const std::set<std::string> kAllowedDifficulty = {"beginner", "intermediate", "advanced"};

    int graph_count = 0;
    for (auto it = fs::recursive_directory_iterator(graphs_dir);
         it != fs::recursive_directory_iterator(); ++it) {
        if (!it->is_regular_file() || it->path().extension() != ".json") continue;
        ++graph_count;

        json doc = load_json(it->path());
        check(doc.contains("meta") && doc["meta"].is_object(),
              it->path().string() + ": missing top-level meta object");
        if (!doc.contains("meta") || !doc["meta"].is_object()) continue;

        const auto& meta = doc["meta"];
        for (const char* key : {"id", "title", "description", "tags", "difficulty",
                                "featured_rank", "requires_packages"}) {
            check(meta.contains(key),
                  it->path().string() + ": meta missing '" + std::string(key) + "'");
        }
        if (meta.contains("difficulty")) {
            check(kAllowedDifficulty.count(meta["difficulty"].get<std::string>()) > 0,
                  it->path().string() + ": difficulty is outside the allowed vocabulary");
        }
        if (meta.contains("requires_packages") && meta["requires_packages"].is_array()) {
            bool has_pkg = false;
            for (const auto& pkg : meta["requires_packages"]) {
                if (pkg.is_string() && pkg.get<std::string>() == "vivid-wavetable") {
                    has_pkg = true;
                    break;
                }
            }
            check(has_pkg, it->path().string() + ": requires_packages must include vivid-wavetable");
        }

        const auto types = node_types(doc);
        if (it->path().string().find("graphs/presets/") != std::string::npos) {
            bool has_core_noise = false;
            for (const auto& [node, type] : types) {
                if (type == "Noise") {
                    has_core_noise = true;
                    break;
                }
            }
            check(!has_core_noise,
                  it->path().string() + ": preset graph still uses core Noise inside the package library");
        }
    }

    check(graph_count > 0, "found maintained package graphs");
}

void check_fixture_contracts() {
    struct Fixture {
        fs::path path;
        std::vector<std::pair<std::string, std::string>> nodes;
        std::vector<std::pair<std::string, std::string>> connections;
        std::vector<std::tuple<std::string, std::string, std::vector<std::string>>> exact_sources;
    };

    const std::vector<Fixture> fixtures = {
        {
            "graphs/core/wavetable_layer_filter_integration.json",
            {
                {"wt", "WavetableLayer"},
                {"voice_breakout", "NoteBreakout"},
                {"amp_env", "EnvelopeAu"},
                {"filter", "Filter"},
            },
            {
                {"chords/notes_out", "wt/notes_in"},
                {"chords/notes_out", "voice_breakout/notes_in"},
                {"voice_breakout/voice_gates", "amp_env/gate"},
                {"voice_breakout/voice_ids", "amp_env/lane_ids"},
                {"amp_env/value", "wt/voice_gain_audio"},
            },
            {
                {"wt", "notes_in", {"chords/notes_out"}},
                {"voice_breakout", "notes_in", {"chords/notes_out"}},
            },
        },
        {
            "graphs/core/wavetable_modular_demo.json",
            {
                {"osc_a", "WavetableLayer"},
                {"osc_b", "WavetableLayer"},
                {"voice_breakout", "NoteBreakout"},
                {"filter_a", "Filter"},
                {"filter_b", "Filter"},
            },
            {
                {"cp1/notes_out", "osc_a/notes_in"},
                {"cp1/notes_out", "osc_b/notes_in"},
                {"cp1/notes_out", "voice_breakout/notes_in"},
                {"voice_breakout/voice_freqs", "filter_a/frequencies"},
                {"voice_breakout/voice_freqs", "filter_b/frequencies"},
                {"amp_env/value", "osc_a/voice_gain_audio"},
                {"amp_env/value", "osc_b/voice_gain_audio"},
            },
            {
                {"osc_a", "notes_in", {"cp1/notes_out"}},
                {"osc_b", "notes_in", {"cp1/notes_out"}},
                {"voice_breakout", "notes_in", {"cp1/notes_out"}},
            },
        },
        {
            "modules/layer_pad.vivid-module.json",
            {
                {"wt", "WavetableLayer"},
                {"voice_breakout", "NoteBreakout"},
                {"amp_env", "EnvelopeAu"},
            },
            {
                {"voice_breakout/notes_out", "wt/notes_in"},
                {"voice_breakout/voice_gates", "amp_env/gate"},
                {"voice_breakout/voice_ids", "amp_env/lane_ids"},
                {"amp_env/value", "wt/voice_gain_audio"},
            },
            {
                {"wt", "notes_in", {"voice_breakout/notes_out"}},
            },
        },
        {
            "modules/hybrid_keys.vivid-module.json",
            {
                {"wt", "WavetableLayer"},
                {"analog", "AnalogOsc"},
                {"voice_breakout", "NoteBreakout"},
                {"amp_env", "EnvelopeAu"},
                {"mix_analog", "VoiceMixer"},
                {"dual_filter", "DualFilter"},
            },
            {
                {"voice_breakout/notes_out", "analog/notes_in"},
                {"voice_breakout/notes_out", "wt/notes_in"},
                {"voice_breakout/voice_gates", "amp_env/gate"},
                {"voice_breakout/voice_ids", "amp_env/lane_ids"},
                {"amp_env/value", "wt/voice_gain_audio"},
                {"amp_env/value", "mix_analog/amp_env_audio"},
                {"voice_breakout/voice_freqs", "dual_filter/frequencies"},
            },
            {
                {"analog", "notes_in", {"voice_breakout/notes_out"}},
                {"wt", "notes_in", {"voice_breakout/notes_out"}},
            },
        },
    };

    for (const auto& fixture : fixtures) {
        const json doc = load_json(fixture.path);
        const auto types = node_types(doc);

        for (const auto& [node, type] : fixture.nodes) {
            require_node_type(types, fixture.path, node, type);
        }
        for (const auto& [from, to] : fixture.connections) {
            require_connection(doc, fixture.path, from, to);
        }
        for (const auto& [node, port, sources] : fixture.exact_sources) {
            std::set<std::string> want(sources.begin(), sources.end());
            const auto got = incoming_sources(doc, node, port);
            check(got == want,
                  fixture.path.string() + ": unexpected sources into " + node + "/" + port);
        }
    }
}

}  // namespace

int main() {
    std::fprintf(stderr, "\n--- Native-note package graph contract ---\n");

    try {
        check_graph_metadata();
        check_fixture_contracts();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  FATAL: %s\n", e.what());
        return 1;
    }

    if (failures == 0) {
        std::fprintf(stderr, "\nPASS: maintained graphs and modules match the native-note contract\n");
    } else {
        std::fprintf(stderr, "\nFAIL: %d contract failures\n", failures);
    }
    return failures == 0 ? 0 : 1;
}
