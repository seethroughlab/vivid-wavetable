#pragma once

#include "operator_api/types.h"
#include "runtime/debug/output_analyzer.h"
#include "runtime/core/shared_handle_registry.h"

#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <unordered_map>
#include <vector>

struct LaneStateStore {
    std::unordered_map<uint64_t, std::vector<uint8_t>> slots;
};

inline void* test_lane_state_fn(void* service, uint32_t lane_id, uint32_t byte_size) {
    auto* store = static_cast<LaneStateStore*>(service);
    uint64_t key = (static_cast<uint64_t>(lane_id) << 32) | static_cast<uint64_t>(byte_size);
    auto& slot = store->slots[key];
    if (slot.size() != byte_size) slot.assign(byte_size, 0);
    return slot.data();
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
    void destroy_instance(void* inst) const {
        if (destroy_fn_ && inst) destroy_fn_(inst);
    }
    void process_audio(void* inst, VividAudioContext* ctx) const {
        if (audio_fn_) audio_fn_(inst, ctx);
    }
};

struct PolyTestContext {
    static constexpr int kFrames = 2048;
    static constexpr uint32_t kSampleRate = 48000;
    static constexpr int kMaxVoices = 16;
    static constexpr int kMaxAudioChannels = 32;
    static constexpr int kMaxPorts = 16;

    float freq_data[kMaxVoices] = {};
    float gate_data[kMaxVoices] = {};
    float vel_data[kMaxVoices] = {};
    float pitch_mod_lane_data[kMaxVoices] = {};
    float position_mod_lane_data[kMaxVoices] = {};
    float warp_mod_lane_data[kMaxVoices] = {};
    float lane_id_data[kMaxVoices] = {};

    VividLaneView input_lanes[kMaxPorts] = {};

    float output_buf[kMaxAudioChannels * kFrames] = {};
    float* output_ptrs[kMaxPorts] = {};
    uint8_t output_ch[kMaxPorts] = {};

    float* input_ptrs[kMaxPorts] = {};
    uint8_t input_ch[kMaxPorts] = {};

    LaneStateStore lane_state;
    VividAudioContext ctx{};

    PolyTestContext() {
        output_ptrs[0] = output_buf;
        output_ch[0] = kMaxAudioChannels;

        ctx.sample_rate = kSampleRate;
        ctx.buffer_size = kFrames;
        ctx.input_buffers = input_ptrs;
        ctx.output_buffers = output_ptrs;
        ctx.input_channel_counts = input_ch;
        ctx.output_channel_counts = output_ch;
        ctx.input_lanes = input_lanes;
        ctx.output_lanes = nullptr;
        ctx.param_values = nullptr;
        ctx.shared_handles = vivid::shared_handle_service();
        ctx.lane_count = 1;
        ctx.lane_index = 0;
        ctx.lane_set_id = 0;
        ctx.lane_id = 1;
        ctx.lane_state_fn = test_lane_state_fn;
        ctx.lane_state_service = &lane_state;
    }

    void clear_lane_ports() {
        for (auto& lane : input_lanes) lane = {nullptr, 0, 0, 0};
    }

    void bind_lane(uint32_t port_idx, float* data, uint32_t length) {
        input_lanes[port_idx].data = data;
        input_lanes[port_idx].length = length;
    }

    void setup_analog_voice(float freq, float velocity = 1.0f) {
        clear_lane_ports();
        freq_data[0] = freq;
        gate_data[0] = 1.0f;
        vel_data[0] = velocity;
        pitch_mod_lane_data[0] = 0.0f;
        lane_id_data[0] = 1.0f;
        // Port layout (post-2026-04 reorder): midi_in=0, frequencies=1,
        // gates=2, velocities=3, pitch_mod=4, lane_ids=5.
        bind_lane(1, freq_data, 1);
        bind_lane(2, gate_data, 1);
        bind_lane(3, vel_data, 1);
        bind_lane(4, pitch_mod_lane_data, 1);
        bind_lane(5, lane_id_data, 1);
    }

    void setup_sub_voice(float freq) {
        clear_lane_ports();
        freq_data[0] = freq;
        gate_data[0] = 1.0f;
        vel_data[0] = 1.0f;
        lane_id_data[0] = 1.0f;
        bind_lane(0, freq_data, 1);
        bind_lane(1, gate_data, 1);
        bind_lane(2, vel_data, 1);
        bind_lane(3, lane_id_data, 1);
    }

    void setup_wavetable_voice(float freq, float velocity = 1.0f) {
        clear_lane_ports();
        freq_data[0] = freq;
        gate_data[0] = 1.0f;
        vel_data[0] = velocity;
        pitch_mod_lane_data[0] = 0.0f;
        position_mod_lane_data[0] = 0.0f;
        warp_mod_lane_data[0] = 0.0f;
        lane_id_data[0] = 1.0f;
        // Port layout (post-2026-04 reorder): midi_in=0, frequencies=1,
        // gates=2, velocities=3, pitch_mod=4, position_mod=5, warp_mod=6,
        // lane_ids=7.
        bind_lane(1, freq_data, 1);
        bind_lane(2, gate_data, 1);
        bind_lane(3, vel_data, 1);
        bind_lane(4, pitch_mod_lane_data, 1);
        bind_lane(5, position_mod_lane_data, 1);
        bind_lane(6, warp_mod_lane_data, 1);
        bind_lane(7, lane_id_data, 1);
    }

    void setup_wavetable_layer_voice(float freq, float velocity = 1.0f) {
        clear_lane_ports();
        freq_data[0] = freq;
        gate_data[0] = 1.0f;
        vel_data[0] = velocity;
        pitch_mod_lane_data[0] = 0.0f;
        position_mod_lane_data[0] = 0.0f;
        warp_mod_lane_data[0] = 0.0f;
        lane_id_data[0] = 1.0f;
        // Port layout (post-2026-04 reorder): midi_in=0, frequencies=1,
        // gates=2, velocities=3, lane_ids=4, pitch_mod=5, position_mod=6,
        // warp_mod=7.
        bind_lane(1, freq_data, 1);              // frequencies
        bind_lane(2, gate_data, 1);              // gates
        bind_lane(3, vel_data, 1);               // velocities
        bind_lane(4, lane_id_data, 1);           // lane_ids
        bind_lane(5, pitch_mod_lane_data, 1);    // pitch_mod
        bind_lane(6, position_mod_lane_data, 1); // position_mod
        bind_lane(7, warp_mod_lane_data, 1);     // warp_mod
    }

    void setup_noise_voice(float freq, float velocity = 1.0f) {
        clear_lane_ports();
        freq_data[0] = freq;
        gate_data[0] = 1.0f;
        vel_data[0] = velocity;
        lane_id_data[0] = 1.0f;
        bind_lane(0, freq_data, 1);
        bind_lane(1, gate_data, 1);
        bind_lane(2, vel_data, 1);
        bind_lane(3, lane_id_data, 1);
    }

    void clear_output() {
        std::memset(output_buf, 0, sizeof(output_buf));
    }

    void set_output_channels(uint8_t channels) {
        output_ch[0] = channels;
    }

    void bind_audio_input(uint32_t port_idx, float* data, uint8_t channels) {
        input_ptrs[port_idx] = data;
        input_ch[port_idx] = channels;
    }

    void clear_audio_inputs() {
        for (int i = 0; i < kMaxPorts; ++i) {
            input_ptrs[i] = nullptr;
            input_ch[i] = 0;
        }
    }

    void silence_gate() {
        gate_data[0] = 0.0f;
    }

    vivid::AudioMetrics analyze_output(uint32_t channels = 1) const {
        return vivid::analyze_audio(output_buf, kFrames, kSampleRate, channels);
    }
};
