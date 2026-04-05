#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

struct PreviewControl {
    std::string node;
    std::string param;
    std::string label;
};

struct GraphNodeInfo {
    std::string type;
    std::set<std::string> float_params;
    std::set<std::string> string_params;
};

json read_json(const fs::path& path) {
    std::ifstream ifs(path);
    json j;
    ifs >> j;
    return j;
}

std::unordered_map<std::string, std::set<std::string>> load_module_param_sets() {
    std::unordered_map<std::string, std::set<std::string>> out;
    const fs::path modules_dir = "../modules";
    if (!fs::exists(modules_dir)) return out;

    for (const auto& entry : fs::directory_iterator(modules_dir)) {
        if (entry.path().extension() != ".json") continue;
        const json module_json = read_json(entry.path());
        const auto& mod = module_json.at("module");
        const std::string type = mod.at("name").get<std::string>();
        std::set<std::string> params;
        if (mod.contains("params") && mod["params"].is_array()) {
            for (const auto& item : mod["params"]) {
                if (item.contains("name") && item["name"].is_string())
                    params.insert(item["name"].get<std::string>());
            }
        }
        out[type] = std::move(params);
    }

    return out;
}

std::unordered_map<std::string, GraphNodeInfo> parse_graph_nodes(const json& graph_json) {
    std::unordered_map<std::string, GraphNodeInfo> nodes;
    const auto& graph_nodes = graph_json.at("nodes");
    for (auto it = graph_nodes.begin(); it != graph_nodes.end(); ++it) {
        GraphNodeInfo info;
        if (it.value().contains("type") && it.value()["type"].is_string())
            info.type = it.value()["type"].get<std::string>();
        if (it.value().contains("params") && it.value()["params"].is_object()) {
            for (auto pit = it.value()["params"].begin(); pit != it.value()["params"].end(); ++pit)
                info.float_params.insert(pit.key());
        }
        if (it.value().contains("string_params") && it.value()["string_params"].is_object()) {
            for (auto sit = it.value()["string_params"].begin(); sit != it.value()["string_params"].end(); ++sit)
                info.string_params.insert(sit.key());
        }
        nodes[it.key()] = std::move(info);
    }
    return nodes;
}

std::vector<PreviewControl> parse_preview_controls(const json& graph_json) {
    std::vector<PreviewControl> controls;
    if (!graph_json.contains("meta")) return controls;
    const auto& meta = graph_json.at("meta");
    if (!meta.contains("preview_controls") || !meta["preview_controls"].is_array()) return controls;

    for (const auto& item : meta["preview_controls"]) {
        PreviewControl ctrl;
        if (item.contains("node") && item["node"].is_string())
            ctrl.node = item["node"].get<std::string>();
        if (item.contains("param") && item["param"].is_string())
            ctrl.param = item["param"].get<std::string>();
        if (item.contains("label") && item["label"].is_string())
            ctrl.label = item["label"].get<std::string>();
        controls.push_back(std::move(ctrl));
    }
    return controls;
}

bool has_midi_input(const json& graph_json) {
    if (!graph_json.contains("nodes")) return false;
    for (auto it = graph_json["nodes"].begin(); it != graph_json["nodes"].end(); ++it) {
        if (it.value().contains("type") && it.value()["type"].is_string() &&
            it.value()["type"].get<std::string>() == "MidiInput") {
            return true;
        }
    }
    return false;
}

bool node_has_saved_param(const GraphNodeInfo& node, const std::string& param) {
    return node.float_params.count(param) || node.string_params.count(param);
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

void validate_instrument(const fs::path& path,
                         const std::unordered_map<std::string, std::set<std::string>>& module_params) {
    const json graph_json = read_json(path);
    const std::string name = path.filename().string();
    if (!graph_json.contains("meta") || !graph_json["meta"].is_object())
        return;
    const auto& meta = graph_json["meta"];

    if (meta.value("content_kind", "") != "instrument")
        return;

    ++instruments_found;

    check(meta.contains("category") && meta["category"].is_string() && !meta["category"].get<std::string>().empty(),
          name + ": has category");
    check(meta.contains("family") && meta["family"].is_string() && !meta["family"].get<std::string>().empty(),
          name + ": has family");
    check(meta.contains("role") && meta["role"].is_string() && !meta["role"].get<std::string>().empty(),
          name + ": has role");
    check(meta.contains("playability") && meta["playability"].is_string() && !meta["playability"].get<std::string>().empty(),
          name + ": has playability");
    check(meta.contains("id") && meta["id"].is_string() && !meta["id"].get<std::string>().empty(),
          name + ": has id");
    check(meta.contains("title") && meta["title"].is_string() && !meta["title"].get<std::string>().empty(),
          name + ": has title");
    check(meta.contains("description") && meta["description"].is_string() && !meta["description"].get<std::string>().empty(),
          name + ": has description");
    check(meta.contains("tags") && meta["tags"].is_array() && !meta["tags"].empty(),
          name + ": has tags");
    check(meta.contains("requires_packages") && meta["requires_packages"].is_array() && !meta["requires_packages"].empty(),
          name + ": has requires_packages");

    const auto nodes = parse_graph_nodes(graph_json);
    const auto controls = parse_preview_controls(graph_json);

    check(!controls.empty(), name + ": has at least one preview control");
    for (size_t i = 0; i < controls.size(); ++i) {
        const auto& ctrl = controls[i];
        const std::string prefix = name + ": preview_control[" + std::to_string(i) + "]";

        check(!ctrl.node.empty(), prefix + " has node");
        check(!ctrl.param.empty(), prefix + " has param");
        auto nit = nodes.find(ctrl.node);
        check(nit != nodes.end(), prefix + " references an existing node");
        if (nit == nodes.end()) continue;

        auto mit = module_params.find(nit->second.type);
        if (mit != module_params.end()) {
            check(mit->second.count(ctrl.param) > 0,
                  prefix + " references an exposed module param");
        }
        check(node_has_saved_param(nit->second, ctrl.param),
              prefix + " references a saved param value");
    }

    if (meta.value("playability", "") == "midi") {
        check(has_midi_input(graph_json),
              name + ": midi playability requires MidiInput node");
    }
}

void test_preview_control_regressions() {
    {
        const json graph_json = {
            {"meta", {
                {"content_kind", "instrument"},
                {"category", "Keys"},
                {"family", "Hybrid"},
                {"role", "reference"},
                {"playability", "midi"},
                {"id", "missing_node"},
                {"title", "Missing Node"},
                {"description", "Regression fixture"},
                {"tags", json::array({"keys"})},
                {"requires_packages", json::array({"vivid-wavetable"})},
                {"preview_controls", json::array({{{"param", "drive"}, {"label", "Body"}}})}
            }},
            {"nodes", {
                {"instrument", {
                    {"type", "HybridKeys"},
                    {"params", {{"drive", 0.1}}}
                }},
                {"midi", {{"type", "MidiInput"}, {"params", json::object()}}}
            }}
        };
        const auto controls = parse_preview_controls(graph_json);
        check(!controls.empty(), "regression: missing-node fixture parsed");
        check(controls[0].node.empty(), "regression: missing node stays empty");
    }

    {
        const json graph_json = {
            {"meta", {
                {"content_kind", "instrument"},
                {"category", "Keys"},
                {"family", "Hybrid"},
                {"role", "reference"},
                {"playability", "midi"},
                {"id", "missing_param"},
                {"title", "Missing Param"},
                {"description", "Regression fixture"},
                {"tags", json::array({"keys"})},
                {"requires_packages", json::array({"vivid-wavetable"})},
                {"preview_controls", json::array({{{"node", "instrument"}, {"label", "Body"}}})}
            }},
            {"nodes", {
                {"instrument", {
                    {"type", "HybridKeys"},
                    {"params", {{"drive", 0.1}}}
                }},
                {"midi", {{"type", "MidiInput"}, {"params", json::object()}}}
            }}
        };
        const auto controls = parse_preview_controls(graph_json);
        check(!controls.empty(), "regression: missing-param fixture parsed");
        check(controls[0].param.empty(), "regression: missing param stays empty");
    }

    {
        const json graph_json = {
            {"meta", {
                {"content_kind", "instrument"},
                {"category", "Keys"},
                {"family", "Hybrid"},
                {"role", "reference"},
                {"playability", "midi"},
                {"id", "bad_reference"},
                {"title", "Bad Reference"},
                {"description", "Regression fixture"},
                {"tags", json::array({"keys"})},
                {"requires_packages", json::array({"vivid-wavetable"})},
                {"preview_controls", json::array({{{"node", "missing"}, {"param", "drive"}, {"label", "Body"}}})}
            }},
            {"nodes", {
                {"instrument", {
                    {"type", "HybridKeys"},
                    {"params", {{"drive", 0.1}}}
                }},
                {"midi", {{"type", "MidiInput"}, {"params", json::object()}}}
            }}
        };
        const auto nodes = parse_graph_nodes(graph_json);
        const auto controls = parse_preview_controls(graph_json);
        check(!controls.empty(), "regression: bad-reference fixture parsed");
        check(nodes.find(controls[0].node) == nodes.end(),
              "regression: non-existent node reference is detectable");
    }
}

} // namespace

int main() {
    const auto module_params = load_module_param_sets();
    test_preview_control_regressions();

    // Scan both graphs/presets/ and graphs/core/ from the test working directory
    std::vector<fs::path> search_dirs = {
        "../graphs/presets",
        "../graphs/core"
    };

    for (const auto& dir : search_dirs) {
        if (!fs::exists(dir)) continue;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() != ".json") continue;
            validate_instrument(entry.path(), module_params);
        }
    }

    check(instruments_found >= 8,
          "at least 8 instrument graphs found (got " + std::to_string(instruments_found) + ")");

    std::cerr << "\n" << instruments_found << " instrument graph(s) validated, "
              << failures << " failure(s)\n";
    return failures > 0 ? 1 : 0;
}
