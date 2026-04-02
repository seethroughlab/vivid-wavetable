#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>
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

std::string regex_escape(const std::string& s) {
    static const std::regex kMeta(R"([-[\]{}()*+?.,\^$|#\s])");
    return std::regex_replace(s, kMeta, R"(\$&)");
}

bool has_connection(const std::string& text, const std::string& from, const std::string& to) {
    const std::string from_esc = regex_escape(from);
    const std::string to_esc = regex_escape(to);
    const std::regex forward("\\{[^\\{\\}]*\"from\"\\s*:\\s*\"" + from_esc +
                             "\"[^\\{\\}]*\"to\"\\s*:\\s*\"" + to_esc + "\"[^\\{\\}]*\\}");
    const std::regex reverse("\\{[^\\{\\}]*\"to\"\\s*:\\s*\"" + to_esc +
                             "\"[^\\{\\}]*\"from\"\\s*:\\s*\"" + from_esc + "\"[^\\{\\}]*\\}");
    return std::regex_search(text, forward) || std::regex_search(text, reverse);
}

size_t skip_ws(const std::string& text, size_t pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    return pos;
}

size_t find_matching(const std::string& text, size_t open_pos, char open_ch, char close_ch) {
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t i = open_pos; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == open_ch) {
            ++depth;
        } else if (c == close_ch) {
            --depth;
            if (depth == 0) return i;
        }
    }
    throw std::runtime_error("unmatched delimiter");
}

std::pair<std::string, size_t> parse_json_string(const std::string& text, size_t pos) {
    if (pos >= text.size() || text[pos] != '"') {
        throw std::runtime_error("expected JSON string");
    }
    std::string result;
    bool escaped = false;
    for (size_t i = pos + 1; i < text.size(); ++i) {
        const char c = text[i];
        if (escaped) {
            result.push_back(c);
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            return {result, i + 1};
        }
        result.push_back(c);
    }
    throw std::runtime_error("unterminated JSON string");
}

std::map<std::string, std::string> parse_object_members(const std::string& object_text) {
    std::map<std::string, std::string> members;
    size_t pos = skip_ws(object_text, 0);
    if (pos >= object_text.size() || object_text[pos] != '{') {
        throw std::runtime_error("expected object");
    }
    ++pos;
    while (true) {
        pos = skip_ws(object_text, pos);
        if (pos >= object_text.size()) break;
        if (object_text[pos] == '}') break;

        const auto [key, after_key] = parse_json_string(object_text, pos);
        pos = skip_ws(object_text, after_key);
        if (pos >= object_text.size() || object_text[pos] != ':') {
            throw std::runtime_error("expected colon");
        }
        pos = skip_ws(object_text, pos + 1);
        if (pos >= object_text.size()) {
            throw std::runtime_error("expected value");
        }

        const size_t value_start = pos;
        size_t value_end = pos;
        if (object_text[pos] == '{') {
            value_end = find_matching(object_text, pos, '{', '}') + 1;
        } else if (object_text[pos] == '[') {
            value_end = find_matching(object_text, pos, '[', ']') + 1;
        } else if (object_text[pos] == '"') {
            value_end = parse_json_string(object_text, pos).second;
        } else {
            while (value_end < object_text.size() && object_text[value_end] != ',' &&
                   object_text[value_end] != '}') {
                ++value_end;
            }
        }

        members[key] = object_text.substr(value_start, value_end - value_start);
        pos = skip_ws(object_text, value_end);
        if (pos < object_text.size() && object_text[pos] == ',') ++pos;
    }
    return members;
}

std::string parse_string_value(const std::string& raw_value) {
    const auto [value, _] = parse_json_string(raw_value, skip_ws(raw_value, 0));
    return value;
}

bool is_difficulty_value_valid(const std::string& raw_value) {
    static const std::set<std::string> kAllowed = {"beginner", "intermediate", "advanced"};
    return kAllowed.count(parse_string_value(raw_value)) > 0;
}

bool requires_package(const std::string& raw_value, const std::string& package_name) {
    return contains(raw_value, "\"" + package_name + "\"");
}

}  // namespace

int main() {
    const std::filesystem::path graphs_dir = "../graphs";
    const std::set<std::string> note_env_names = {"amp_env", "filt_env", "pos_env", "air_env", "noise_env"};
    const std::set<std::string> per_voice_source_types = {"WavetableOsc", "AnalogOsc", "SubOsc", "NoiseLayer"};
    const std::set<std::string> meta_fields = {
        "id", "title", "description", "tags", "difficulty", "featured_rank", "requires_packages"};

    int checked_graphs = 0;
    int failures = 0;

    for (const auto& entry : std::filesystem::recursive_directory_iterator(graphs_dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".json") continue;
        ++checked_graphs;

        const std::string text = read_file(entry.path());
        const auto root = parse_object_members(text);
        const auto meta_it = root.find("meta");
        if (meta_it == root.end()) {
            std::cerr << entry.path().string() << ": missing top-level meta object\n";
            ++failures;
            continue;
        }

        const auto meta = parse_object_members(meta_it->second);
        for (const auto& field : meta_fields) {
            if (meta.count(field) == 0) {
                std::cerr << entry.path().string() << ": meta is missing required field '" << field << "'\n";
                ++failures;
            }
        }
        if (meta.count("difficulty") > 0 && !is_difficulty_value_valid(meta.at("difficulty"))) {
            std::cerr << entry.path().string() << ": meta difficulty is outside the allowed vocabulary\n";
            ++failures;
        }
        if (meta.count("requires_packages") > 0 &&
            !requires_package(meta.at("requires_packages"), "vivid-wavetable")) {
            std::cerr << entry.path().string()
                      << ": meta.requires_packages must include vivid-wavetable\n";
            ++failures;
        }

        const auto nodes_it = root.find("nodes");
        if (nodes_it == root.end()) {
            std::cerr << entry.path().string() << ": missing nodes object\n";
            ++failures;
            continue;
        }
        const auto nodes = parse_object_members(nodes_it->second);

        std::map<std::string, std::string> node_types;
        bool has_allocator = false;
        bool has_voice_mixer = false;
        bool has_package_voice_source = false;
        bool has_core_noise = false;

        for (const auto& [name, node_text] : nodes) {
            const auto fields = parse_object_members(node_text);
            const auto type_it = fields.find("type");
            if (type_it == fields.end()) continue;
            const std::string type = parse_string_value(type_it->second);
            node_types[name] = type;
            has_allocator = has_allocator || type == "PolyVoiceAllocator";
            has_voice_mixer = has_voice_mixer || type == "VoiceMixer";
            has_package_voice_source = has_package_voice_source || per_voice_source_types.count(type) > 0;
            has_core_noise = has_core_noise || type == "Noise";
        }

        if (!has_allocator) continue;

        for (const auto& [name, type] : node_types) {
            if (note_env_names.count(name) > 0 && type == "EnvelopeAu") {
                if (contains(text, "\"to\": \"" + name + "/beat_phase\"")) {
                    std::cerr << entry.path().string() << ": " << name
                              << " still routes note articulation through beat_phase\n";
                    ++failures;
                }
                if (!has_connection(text, "voices/gates", name + "/gate")) {
                    std::cerr << entry.path().string()
                              << ": missing voices/gates -> " << name << "/gate\n";
                    ++failures;
                }
            }

            if (per_voice_source_types.count(type) > 0) {
                const std::vector<std::pair<std::string, std::string>> required_connections = {
                    {"voices/frequencies", name + "/frequencies"},
                    {"voices/gates", name + "/gates"},
                    {"voices/velocities", name + "/velocities"},
                    {"voices/lane_ids", name + "/lane_ids"},
                };
                for (const auto& [from, to] : required_connections) {
                    if (!has_connection(text, from, to)) {
                        std::cerr << entry.path().string() << ": missing " << from
                                  << " -> " << to << "\n";
                        ++failures;
                    }
                }
            }

            if (type == "Filter" && has_package_voice_source &&
                !has_connection(text, "voices/frequencies", name + "/frequencies")) {
                std::cerr << entry.path().string()
                          << ": polyphonic filter " << name
                          << " is missing voices/frequencies for tracking\n";
                ++failures;
            }
        }

        if (node_types.count("amp_env") > 0 && node_types.at("amp_env") == "EnvelopeAu" &&
            has_voice_mixer && has_package_voice_source) {
            bool mixer_has_amp_env = false;
            for (const auto& [name, type] : node_types) {
                if (type != "VoiceMixer") continue;
                mixer_has_amp_env = mixer_has_amp_env ||
                                    has_connection(text, "amp_env/value", name + "/amp_env_audio") ||
                                    has_connection(text, "amp_env/value", name + "/amp_env");
            }
            if (!mixer_has_amp_env) {
                std::cerr << entry.path().string()
                          << ": no VoiceMixer receives per-note amp_env modulation\n";
                ++failures;
            }
        }

        if (entry.path().string().find("graphs/presets/") != std::string::npos && has_core_noise) {
            std::cerr << entry.path().string()
                      << ": preset graph still uses core Noise inside the package library\n";
            ++failures;
        }
    }

    if (checked_graphs == 0) {
        std::cerr << "no package graphs were checked\n";
        return 1;
    }

    if (failures > 0) {
        std::cerr << "graph preset audit failed with " << failures << " issue(s)\n";
        return 1;
    }

    std::cout << "checked " << checked_graphs
              << " package graphs; metadata and lane-aware poly wiring passed\n";
    return 0;
}
