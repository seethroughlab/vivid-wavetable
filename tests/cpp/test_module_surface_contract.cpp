#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "runtime/graph/graph.h"
#include "runtime/graph/subgraph_module.h"

namespace {

std::string read_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) return {};
    return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

bool check(bool cond, const std::string& msg) {
    if (!cond) {
        std::cerr << msg << "\n";
        return false;
    }
    return true;
}

bool expect_module_loads(vivid::SubgraphModuleRegistry& registry,
                         const std::string& path,
                         const std::string& type_name) {
    if (!check(registry.load(path), "failed to load module: " + path)) return false;
    return check(registry.find(type_name) != nullptr, "module not registered: " + type_name);
}

bool test_shipped_modules_load() {
    vivid::SubgraphModuleRegistry registry;
    bool ok = true;

    ok &= expect_module_loads(registry, "modules/hybrid_keys.vivid-module.json", "HybridKeys");
    ok &= expect_module_loads(registry, "modules/dual_wavetable_pad.vivid-module.json", "DualWavetablePad");
    ok &= expect_module_loads(registry, "modules/sub_air_pad.vivid-module.json", "SubAirPad");
    ok &= expect_module_loads(registry, "modules/glass_interaction_keys.vivid-module.json", "GlassInteractionKeys");

    const auto* glass = registry.find("GlassInteractionKeys");
    if (!check(glass != nullptr, "GlassInteractionKeys definition missing")) return false;

    ok &= check(glass->find_port("pressures") != nullptr, "GlassInteractionKeys missing pressures port");
    ok &= check(glass->find_port("slides") != nullptr, "GlassInteractionKeys missing slides port");
    ok &= check(glass->find_port("aftertouch") == nullptr, "GlassInteractionKeys still exposes aftertouch port");
    ok &= check(glass->find_port("expression") == nullptr, "GlassInteractionKeys still exposes expression port");
    ok &= check(glass->find_mod_source("pressures") != nullptr, "GlassInteractionKeys missing pressures mod source");
    ok &= check(glass->find_mod_source("slides") != nullptr, "GlassInteractionKeys missing slides mod source");
    ok &= check(glass->find_mod_source("aftertouch") == nullptr, "GlassInteractionKeys still exposes aftertouch mod source");
    ok &= check(glass->find_mod_source("expression") == nullptr, "GlassInteractionKeys still exposes expression mod source");

    const auto* hybrid = registry.find("HybridKeys");
    const auto* dual = registry.find("DualWavetablePad");
    const auto* sub_air = registry.find("SubAirPad");

    ok &= check(hybrid != nullptr, "HybridKeys definition missing");
    ok &= check(dual != nullptr, "DualWavetablePad definition missing");
    ok &= check(sub_air != nullptr, "SubAirPad definition missing");

    if (hybrid) {
        ok &= check(hybrid->find_param("filter_cutoff") != nullptr, "HybridKeys missing stable filter_cutoff param");
        ok &= check(hybrid->find_param("drive") != nullptr, "HybridKeys missing stable drive param");
    }
    if (dual) {
        ok &= check(dual->find_param("motion_amount") != nullptr, "DualWavetablePad missing stable motion_amount param");
        ok &= check(dual->find_param("filter_tone") != nullptr, "DualWavetablePad missing stable filter_tone param");
    }
    if (sub_air) {
        ok &= check(sub_air->find_param("air_level") != nullptr, "SubAirPad missing stable air_level param");
        ok &= check(sub_air->find_param("filter_tone") != nullptr, "SubAirPad missing stable filter_tone param");
        ok &= check(sub_air->find_param("body") != nullptr, "SubAirPad missing stable body param");
    }
    if (glass) {
        ok &= check(glass->find_param("interaction_depth") != nullptr, "GlassInteractionKeys missing stable interaction_depth param");
        ok &= check(glass->find_param("filter_cutoff") != nullptr, "GlassInteractionKeys missing stable filter_cutoff param");
    }

    return ok;
}

bool test_expressive_graph_ports_match_module() {
    vivid::SubgraphModuleRegistry registry;
    if (!expect_module_loads(registry, "modules/glass_interaction_keys.vivid-module.json", "GlassInteractionKeys")) {
        return false;
    }
    const auto* glass = registry.find("GlassInteractionKeys");
    if (!check(glass != nullptr, "GlassInteractionKeys definition missing")) return false;

    std::string graph_json = read_file("graphs/presets/expressive_glass_keys.json");
    if (!check(!graph_json.empty(), "failed to read expressive_glass_keys.json")) return false;

    vivid::Graph graph;
    if (!check(graph.load_from_string(graph_json.c_str(), graph_json.size()), "failed to parse expressive_glass_keys.json")) {
        return false;
    }

    const auto* instrument = graph.find_node("instrument");
    if (!check(instrument != nullptr, "expressive graph missing instrument node")) return false;
    if (!check(instrument->type == "GlassInteractionKeys", "instrument node is not GlassInteractionKeys")) return false;

    bool ok = true;
    for (const auto& conn : graph.connections()) {
        if (conn.to_node == "instrument") {
            ok &= check(glass->find_port(conn.to_port) != nullptr,
                        "graph references missing GlassInteractionKeys input port: " + conn.to_port);
        }
        if (conn.from_node == "instrument") {
            ok &= check(glass->find_port(conn.from_port) != nullptr,
                        "graph references missing GlassInteractionKeys output port: " + conn.from_port);
        }
    }
    return ok;
}

bool test_expressive_graph_mod_assignments_round_trip() {
    std::string graph_json = read_file("graphs/presets/expressive_glass_keys.json");
    if (!check(!graph_json.empty(), "failed to read expressive_glass_keys.json")) return false;

    vivid::Graph graph;
    if (!check(graph.load_from_string(graph_json.c_str(), graph_json.size()), "failed to load expressive graph")) {
        return false;
    }

    const auto* assignments = graph.find_mod_assignments("instrument");
    if (!check(assignments != nullptr, "expressive graph missing instrument mod_assignments")) return false;
    if (!check(assignments->size() == 2, "expressive graph should have exactly 2 mod_assignments")) return false;

    std::unordered_map<std::string, vivid::ModAssignmentDef> by_destination;
    for (const auto& assignment : *assignments) {
        by_destination[assignment.destination] = assignment;
    }

    bool ok = true;
    auto pressure_it = by_destination.find("interaction");
    ok &= check(pressure_it != by_destination.end(), "missing interaction modulation assignment");
    if (pressure_it != by_destination.end()) {
        ok &= check(pressure_it->second.source == "pressures", "interaction assignment should use pressures");
        ok &= check(pressure_it->second.amount == 0.4f, "interaction assignment amount should stay normalized");
    }

    auto slide_it = by_destination.find("brightness");
    ok &= check(slide_it != by_destination.end(), "missing brightness modulation assignment");
    if (slide_it != by_destination.end()) {
        ok &= check(slide_it->second.source == "slides", "brightness assignment should use slides");
        ok &= check(slide_it->second.amount >= 1000.0f, "brightness assignment amount should be authored in audible Hz units");
    }

    std::string saved;
    ok &= check(graph.save_to_string(saved), "failed to save expressive graph");

    vivid::Graph reloaded;
    ok &= check(reloaded.load_from_string(saved.c_str(), saved.size()), "failed to reload expressive graph");
    const auto* reloaded_assignments = reloaded.find_mod_assignments("instrument");
    ok &= check(reloaded_assignments != nullptr, "reloaded expressive graph missing mod_assignments");
    ok &= check(reloaded_assignments && reloaded_assignments->size() == 2, "reloaded expressive graph should preserve 2 mod_assignments");
    return ok;
}

bool test_dual_pad_brightness_mod_amount() {
    std::string graph_json = read_file("graphs/presets/dual_wavetable_pad_module_demo.json");
    if (!check(!graph_json.empty(), "failed to read dual_wavetable_pad_module_demo.json")) return false;

    vivid::Graph graph;
    if (!check(graph.load_from_string(graph_json.c_str(), graph_json.size()), "failed to load dual_wavetable_pad_module_demo.json")) {
        return false;
    }

    const auto* assignments = graph.find_mod_assignments("instrument");
    if (!check(assignments != nullptr, "dual_wavetable_pad_module_demo missing mod_assignments")) return false;

    auto it = std::find_if(assignments->begin(), assignments->end(), [](const vivid::ModAssignmentDef& assignment) {
        return assignment.source == "motion_lfo" && assignment.destination == "brightness";
    });
    if (!check(it != assignments->end(), "dual_wavetable pad demo missing motion_lfo -> brightness assignment")) {
        return false;
    }
    return check(it->amount >= 500.0f, "dual_wavetable pad brightness modulation amount should be authored in Hz units");
}

}  // namespace

int main() {
    bool ok = true;
    ok &= test_shipped_modules_load();
    ok &= test_expressive_graph_ports_match_module();
    ok &= test_expressive_graph_mod_assignments_round_trip();
    ok &= test_dual_pad_brightness_mod_amount();

    if (!ok) return 1;

    std::cout << "module surface contract checks passed\n";
    return 0;
}
