#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

static int failures = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
        ++failures;
    } else {
        std::fprintf(stderr, "PASS: %s\n", msg.c_str());
    }
}

static std::string read_file(const fs::path& path) {
    std::ifstream ifs(path);
    if (!ifs) return {};
    return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

static bool run_single_graph(const fs::path& demo_runner,
                             const fs::path& core_build_dir,
                             const fs::path& graph_path,
                             const fs::path& package_root) {
    fs::path sandbox = core_build_dir / ".test_layerpad_reference_graphs_packages";
    std::error_code ec;
    fs::remove_all(sandbox, ec);
    fs::create_directories(sandbox, ec);
    if (ec) {
        std::fprintf(stderr, "could not create package sandbox: %s\n", ec.message().c_str());
        return false;
    }

    fs::path linked_pkg = sandbox / "vivid-wavetable";
    fs::create_directory_symlink(package_root, linked_pkg, ec);
    if (ec) {
        std::fprintf(stderr, "could not symlink package into sandbox: %s\n", ec.message().c_str());
        return false;
    }

    std::string old_paths = std::getenv("VIVID_PACKAGE_PATHS") ? std::getenv("VIVID_PACKAGE_PATHS") : "";
    setenv("VIVID_PACKAGE_PATHS", sandbox.c_str(), 1);

    std::string command = demo_runner.string() + " " + core_build_dir.string() + " --single " + graph_path.string();
    int rc = std::system(command.c_str());

    if (!old_paths.empty()) setenv("VIVID_PACKAGE_PATHS", old_paths.c_str(), 1);
    else unsetenv("VIVID_PACKAGE_PATHS");

    if (rc == -1) {
        std::fprintf(stderr, "system() failed for %s: %s\n", graph_path.c_str(), std::strerror(errno));
        return false;
    }
    if (!WIFEXITED(rc)) {
        std::fprintf(stderr, "graph runner did not exit cleanly for %s\n", graph_path.c_str());
        return false;
    }
    if (WEXITSTATUS(rc) != 0) {
        std::fprintf(stderr, "graph runner exit=%d for %s\n", WEXITSTATUS(rc), graph_path.c_str());
        return false;
    }
    return true;
}

int main() {
    const fs::path package_root = fs::current_path();
    const fs::path core_build_dir = fs::path(VIVID_CORE_BUILD_DIR_STR);
    const fs::path demo_runner = core_build_dir / "test_demo_graphs";

    check(fs::exists(demo_runner), "core demo graph runner exists");
    check(fs::exists(package_root / "modules" / "layer_pad.vivid-module.json"), "LayerPad module file exists");
    if (!fs::exists(demo_runner)) return 1;

    const std::string manifest = read_file(package_root / "vivid-package.json");
    check(manifest.find("\"graphs/core/wavetable_layer_stress.json\"") != std::string::npos,
          "stress graph remains on the package smoke surface");

    const fs::path pad_demo = package_root / "graphs" / "core" / "wavetable_layer_pad_demo.json";
    const fs::path filter_demo = package_root / "graphs" / "core" / "wavetable_layer_filter_integration.json";
    const fs::path stress_demo = package_root / "graphs" / "core" / "wavetable_layer_stress.json";

    check(run_single_graph(demo_runner, core_build_dir, pad_demo, package_root),
          "wavetable_layer_pad_demo loads, builds, and produces audible graph output");
    check(run_single_graph(demo_runner, core_build_dir, filter_demo, package_root),
          "wavetable_layer_filter_integration loads, builds, and produces audible graph output");
    check(run_single_graph(demo_runner, core_build_dir, stress_demo, package_root),
          "wavetable_layer_stress remains a smoke/load graph in Phase 4");

    if (failures == 0) {
        std::printf("LayerPad reference graph checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "LayerPad reference graph checks failed: %d\n", failures);
    return 1;
}
