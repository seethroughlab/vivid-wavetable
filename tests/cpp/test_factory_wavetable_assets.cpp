#include "wavetable_bank.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

using namespace vivid_wavetable::bank;

static int failures = 0;

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++failures;
    } else {
        std::fprintf(stderr, "PASS: %s\n", msg);
    }
}

int main() {
    namespace fs = std::filesystem;
    const fs::path assets_dir = "../assets/wavetables";

    check(fs::exists(assets_dir) && fs::is_directory(assets_dir),
          "assets/wavetables directory exists");
    if (!fs::exists(assets_dir)) return 1;

    int files_checked = 0;

    for (const auto& entry : fs::directory_iterator(assets_dir)) {
        if (entry.path().extension() != ".wav") continue;
        ++files_checked;

        const std::string path_str = entry.path().string();
        const std::string name = entry.path().filename().string();

        Wavetable* wt = load_wavetable_from_wav(path_str);

        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s: load_wavetable_from_wav returns non-null", name.c_str());
        check(wt != nullptr, buf);
        if (!wt) continue;

        std::snprintf(buf, sizeof(buf), "%s: frame_count >= 1 (got %u)", name.c_str(), wt->frame_count);
        check(wt->frame_count >= 1, buf);

        // Sample at a few positions and verify finite values
        bool all_finite = true;
        for (float pos : {0.0f, 0.25f, 0.5f, 0.75f, 1.0f}) {
            float s = wt->sample(0.25f, pos, 220.0f, 48000.0f);
            if (!std::isfinite(s)) all_finite = false;
        }
        std::snprintf(buf, sizeof(buf), "%s: samples are finite", name.c_str());
        check(all_finite, buf);

        delete wt;
    }

    char buf[128];
    std::snprintf(buf, sizeof(buf), "at least 12 factory wavetable files found (got %d)", files_checked);
    check(files_checked >= 12, buf);

    std::fprintf(stderr, "\n%d factory wavetable asset%s checked, %d failure%s\n",
                 files_checked, files_checked == 1 ? "" : "s",
                 failures, failures == 1 ? "" : "s");
    return failures > 0 ? 1 : 0;
}
