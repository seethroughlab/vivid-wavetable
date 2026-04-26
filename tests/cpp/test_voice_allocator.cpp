// Regression tests for VoiceAllocator voice limiting and lane release.
// (Renamed from PolyVoiceAllocator in 2026-04; the operator alias in vivid
// core lets graphs that reference the old name continue to load.)
// behavior. These run the package dylib directly via a minimal loader.

#include "operator_api/types.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "  FAIL: %s\n", msg);
        failures++;
    } else {
        std::fprintf(stderr, "  PASS: %s\n", msg);
    }
}

struct MiniLoader {
    void* handle_ = nullptr;
    VividDescriptorFn desc_fn_ = nullptr;
    VividCreateFn create_fn_ = nullptr;
    VividDestroyFn destroy_fn_ = nullptr;
    VividProcessAudioFn audio_fn_ = nullptr;

    bool load(const char* path) {
        handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle_) {
            std::fprintf(stderr, "  dlopen: %s\n", dlerror());
            return false;
        }
        desc_fn_ = reinterpret_cast<VividDescriptorFn>(dlsym(handle_, "vivid_descriptor"));
        create_fn_ = reinterpret_cast<VividCreateFn>(dlsym(handle_, "vivid_create"));
        destroy_fn_ = reinterpret_cast<VividDestroyFn>(dlsym(handle_, "vivid_destroy"));
        audio_fn_ = reinterpret_cast<VividProcessAudioFn>(dlsym(handle_, "vivid_process_audio"));
        return desc_fn_ && create_fn_ && destroy_fn_ && audio_fn_;
    }

    ~MiniLoader() {
        if (handle_) dlclose(handle_);
    }

    const VividOperatorDescriptor* descriptor() const { return desc_fn_ ? desc_fn_() : nullptr; }
    void* create_instance() const { return create_fn_ ? create_fn_() : nullptr; }
    void destroy_instance(void* inst) const { if (destroy_fn_ && inst) destroy_fn_(inst); }
    void process_audio(void* inst, VividAudioContext* ctx) const { if (audio_fn_) audio_fn_(inst, ctx); }
};

struct LaneIdService {
    uint32_t next_id = 100;
    std::vector<uint32_t> retired;
};

static uint32_t alloc_lane_id(void* service) {
    auto* ids = static_cast<LaneIdService*>(service);
    return ids->next_id++;
}

static void retire_lane_id(void* service, uint32_t lane_id) {
    auto* ids = static_cast<LaneIdService*>(service);
    ids->retired.push_back(lane_id);
}

struct AllocatorTestContext {
    static constexpr uint32_t kFrames = 256;
    static constexpr uint32_t kSampleRate = 48000;
    static constexpr uint32_t kMaxVoices = 16;
    static constexpr uint32_t kInputPorts = 3;
    static constexpr uint32_t kOutputPorts = 5;

    float notes_in[kMaxVoices] = {};
    float velocities_in[kMaxVoices] = {};
    float gates_in[kMaxVoices] = {};

    // Mock lane output backing storage
    struct MockLaneOutput {
        float data[kMaxVoices] = {};
        uint32_t length = 0;
    };
    MockLaneOutput mock_outputs[kOutputPorts] = {};

    static float* mock_resize(void* handle, uint32_t length) {
        auto* mock = static_cast<MockLaneOutput*>(handle);
        mock->length = 0; // reset until commit
        return mock->data;
    }
    static void mock_commit(void* handle, uint32_t length) {
        auto* mock = static_cast<MockLaneOutput*>(handle);
        mock->length = length;
    }

    VividLaneView input_lanes[kInputPorts] = {};
    VividLaneOutput output_lanes[kOutputPorts] = {};
    float* input_buffers[1] = {};
    float* output_buffers[1] = {};
    uint8_t input_channels[1] = {};
    uint8_t output_channels[1] = {};
    LaneIdService lane_ids;
    VividAudioContext ctx{};

    AllocatorTestContext() {
        input_lanes[0] = {notes_in, 0, 0, 0};
        input_lanes[1] = {velocities_in, 0, 0, 0};
        input_lanes[2] = {gates_in, 0, 0, 0};

        for (int i = 0; i < kOutputPorts; ++i)
            output_lanes[i] = {&mock_outputs[i], mock_resize, mock_commit};

        ctx.sample_rate = kSampleRate;
        ctx.buffer_size = kFrames;
        ctx.input_buffers = input_buffers;
        ctx.output_buffers = output_buffers;
        ctx.input_channel_counts = input_channels;
        ctx.output_channel_counts = output_channels;
        ctx.input_lanes = input_lanes;
        ctx.output_lanes = output_lanes;
        ctx.custom_inputs = nullptr;
        ctx.custom_input_count = 0;
        ctx.custom_outputs = nullptr;
        ctx.custom_output_count = 0;
        ctx.allocate_lane_id_fn = alloc_lane_id;
        ctx.retire_lane_id_fn = retire_lane_id;
        ctx.lane_state_service = &lane_ids;
    }

    void set_inputs(const std::vector<float>& notes, const std::vector<float>& gates, float velocity = 0.8f) {
        std::fill(std::begin(notes_in), std::end(notes_in), 0.0f);
        std::fill(std::begin(velocities_in), std::end(velocities_in), 0.0f);
        std::fill(std::begin(gates_in), std::end(gates_in), 0.0f);

        uint32_t len = static_cast<uint32_t>(std::min(notes.size(), gates.size()));
        len = std::min(len, kMaxVoices);
        for (uint32_t i = 0; i < len; ++i) {
            notes_in[i] = notes[i];
            velocities_in[i] = velocity;
            gates_in[i] = gates[i];
        }
        input_lanes[0].length = len;
        input_lanes[1].length = len;
        input_lanes[2].length = len;
    }

    uint32_t output_length() const {
        return mock_outputs[0].length;
    }

    bool retired_contains(uint32_t lane_id) const {
        return std::find(lane_ids.retired.begin(), lane_ids.retired.end(), lane_id) != lane_ids.retired.end();
    }
};

static int find_param_index(const VividOperatorDescriptor* desc, const char* name) {
    for (uint32_t i = 0; i < desc->param_count; ++i) {
        if (std::strcmp(desc->params[i].name, name) == 0) return static_cast<int>(i);
    }
    return -1;
}

static void test_max_voices_limit(const MiniLoader& loader, const VividOperatorDescriptor* desc) {
    std::fprintf(stderr, "\n--- VoiceAllocator: max_voices is enforced ---\n");

    int max_voices_idx = find_param_index(desc, "max_voices");
    check(max_voices_idx >= 0, "max_voices param found");
    if (max_voices_idx < 0) return;

    std::vector<float> params(desc->param_count);
    for (uint32_t i = 0; i < desc->param_count; ++i)
        params[i] = desc->params[i].default_value;
    params[max_voices_idx] = 2.0f;

    AllocatorTestContext tc;
    tc.ctx.param_values = params.data();
    tc.set_inputs({60.0f, 64.0f, 67.0f}, {1.0f, 1.0f, 1.0f});

    void* inst = loader.create_instance();
    loader.process_audio(inst, &tc.ctx);

    check(tc.mock_outputs[0].length == 2, "allocator emits no more than max_voices lanes");
    check(tc.mock_outputs[1].length == 2, "velocity output matches limited lane count");
    check(tc.mock_outputs[2].length == 2, "gate output matches limited lane count");
    check(tc.mock_outputs[4].length == 2, "lane_ids output matches limited lane count");

    loader.destroy_instance(inst);
}

static void test_lane_release_tail(const MiniLoader& loader, const VividOperatorDescriptor* desc) {
    std::fprintf(stderr, "\n--- VoiceAllocator: lane release tail is time-based ---\n");

    int max_voices_idx = find_param_index(desc, "max_voices");
    check(max_voices_idx >= 0, "max_voices param found");
    if (max_voices_idx < 0) return;

    std::vector<float> params(desc->param_count);
    for (uint32_t i = 0; i < desc->param_count; ++i)
        params[i] = desc->params[i].default_value;
    params[max_voices_idx] = 4.0f;

    AllocatorTestContext tc;
    tc.ctx.param_values = params.data();

    void* inst = loader.create_instance();

    tc.set_inputs({60.0f}, {1.0f});
    loader.process_audio(inst, &tc.ctx);
    check(tc.output_length() == 1, "initial note-on produces one active lane");
    uint32_t lane_id = static_cast<uint32_t>(tc.mock_outputs[4].data[0]);

    tc.set_inputs({60.0f}, {0.0f});
    loader.process_audio(inst, &tc.ctx);
    check(tc.output_length() == 1, "released lane survives one handoff buffer");
    check(tc.mock_outputs[2].data[0] == 0.0f, "handoff buffer marks the lane gate low");
    check(static_cast<uint32_t>(tc.mock_outputs[4].data[0]) == lane_id, "handoff keeps the same lane id");

    for (int i = 0; i < 300; ++i)
        loader.process_audio(inst, &tc.ctx);

    check(tc.output_length() == 1, "lane-driven release tail stays alive for long-release envelopes");

    for (int i = 0; i < 100; ++i)
        loader.process_audio(inst, &tc.ctx);

    check(tc.output_length() == 0, "lane-driven release tail eventually retires");
    check(tc.retired_contains(lane_id), "expired release retires the lane id");

    loader.destroy_instance(inst);
}

static void test_releasing_voice_is_stolen_first(const MiniLoader& loader, const VividOperatorDescriptor* desc) {
    std::fprintf(stderr, "\n--- VoiceAllocator: releasing voices are stolen before sustaining voices ---\n");

    int max_voices_idx = find_param_index(desc, "max_voices");
    check(max_voices_idx >= 0, "max_voices param found");
    if (max_voices_idx < 0) return;

    std::vector<float> params(desc->param_count);
    for (uint32_t i = 0; i < desc->param_count; ++i)
        params[i] = desc->params[i].default_value;
    params[max_voices_idx] = 4.0f;

    AllocatorTestContext tc;
    tc.ctx.param_values = params.data();

    void* inst = loader.create_instance();

    tc.set_inputs({60.0f, 64.0f, 67.0f, 71.0f}, {1.0f, 1.0f, 1.0f, 1.0f});
    loader.process_audio(inst, &tc.ctx);
    check(tc.output_length() == 4, "initial chord fills all available voices");

    uint32_t released_lane_id = static_cast<uint32_t>(tc.mock_outputs[4].data[0]);
    uint32_t sustain_lane_a = static_cast<uint32_t>(tc.mock_outputs[4].data[1]);
    uint32_t sustain_lane_b = static_cast<uint32_t>(tc.mock_outputs[4].data[2]);
    uint32_t sustain_lane_c = static_cast<uint32_t>(tc.mock_outputs[4].data[3]);

    tc.set_inputs({60.0f, 64.0f, 67.0f, 71.0f}, {0.0f, 1.0f, 1.0f, 1.0f});
    loader.process_audio(inst, &tc.ctx);
    check(tc.output_length() == 4, "released voice remains during the long tail");

    tc.set_inputs({72.0f, 64.0f, 67.0f, 71.0f}, {1.0f, 1.0f, 1.0f, 1.0f});
    loader.process_audio(inst, &tc.ctx);
    check(tc.output_length() == 4, "new note still respects max_voices after stealing");
    check(tc.retired_contains(released_lane_id), "allocator retires the released voice when it is stolen");

    bool kept_sustain_a = false;
    bool kept_sustain_b = false;
    bool kept_sustain_c = false;
    bool found_released = false;
    for (uint32_t i = 0; i < tc.output_length(); ++i) {
        uint32_t lane = static_cast<uint32_t>(tc.mock_outputs[4].data[i]);
        if (lane == sustain_lane_a) kept_sustain_a = true;
        if (lane == sustain_lane_b) kept_sustain_b = true;
        if (lane == sustain_lane_c) kept_sustain_c = true;
        if (lane == released_lane_id) found_released = true;
    }
    check(kept_sustain_a && kept_sustain_b && kept_sustain_c, "sustaining voices are preserved when a released voice can be stolen");
    check(!found_released, "released voice lane is replaced by the new note");

    loader.destroy_instance(inst);
}

static void test_chord_retrigger_settles(const MiniLoader& loader, const VividOperatorDescriptor* desc) {
    std::fprintf(stderr, "\n--- VoiceAllocator: retriggered chords settle back to current notes ---\n");

    int max_voices_idx = find_param_index(desc, "max_voices");
    check(max_voices_idx >= 0, "max_voices param found");
    if (max_voices_idx < 0) return;

    std::vector<float> params(desc->param_count);
    for (uint32_t i = 0; i < desc->param_count; ++i)
        params[i] = desc->params[i].default_value;
    params[max_voices_idx] = 8.0f;

    AllocatorTestContext tc;
    tc.ctx.param_values = params.data();

    void* inst = loader.create_instance();

    tc.set_inputs({60.0f, 64.0f, 67.0f}, {1.0f, 1.0f, 1.0f});
    loader.process_audio(inst, &tc.ctx);
    check(tc.output_length() == 3, "first chord emits three lanes");

    tc.set_inputs({59.0f, 62.0f, 67.0f}, {1.0f, 1.0f, 1.0f});
    loader.process_audio(inst, &tc.ctx);
    check(tc.output_length() <= 5, "retriggered chord produces only a brief overlap");

    for (int i = 0; i < 400; ++i)
        loader.process_audio(inst, &tc.ctx);

    check(tc.output_length() == 3, "retriggered chord settles back to the current three notes");

    loader.destroy_instance(inst);
}

int main() {
    std::string build_dir = ".";
    std::string dylib_path = build_dir + "/voice_allocator.dylib";

    if (!std::filesystem::exists(dylib_path)) {
        std::fprintf(stderr, "  FAIL: %s not found\n", dylib_path.c_str());
        return 1;
    }

    MiniLoader loader;
    if (!loader.load(dylib_path.c_str())) {
        std::fprintf(stderr, "  FAIL: could not load voice_allocator.dylib\n");
        return 1;
    }

    const auto* desc = loader.descriptor();
    if (!desc) {
        std::fprintf(stderr, "  FAIL: missing descriptor\n");
        return 1;
    }

    std::fprintf(stderr, "\n=== Test: VoiceAllocator Regression Coverage ===\n");
    test_max_voices_limit(loader, desc);
    test_lane_release_tail(loader, desc);
    test_releasing_voice_is_stolen_first(loader, desc);
    test_chord_retrigger_settles(loader, desc);

    std::fprintf(stderr, "\n=== %s (%d failures) ===\n\n",
                 failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures == 0 ? 0 : 1;
}
