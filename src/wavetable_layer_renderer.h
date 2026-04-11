#pragma once

#include "wavetable_bank.h"
#include "wavetable_dsp.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>

namespace vivid_wavetable::layer {

// ---------------------------------------------------------------------------
// PreparedWavetable — guard-sample storage for branch-free linear interpolation
// ---------------------------------------------------------------------------

struct PreparedWavetable {
    static constexpr uint32_t kGuardedFrameSize = bank::kSamplesPerFrame + 1; // 2049

    // Each level: frame_count * kGuardedFrameSize contiguous floats.
    // All levels packed into one flat buffer for SIMD GatherIndex.
    std::vector<float> flat_data;
    uint32_t level_offset[bank::kNumMipLevels] = {}; // byte offset into flat_data for each level
    uint32_t frame_count = 0;

    void prepare_from(const bank::Wavetable& src);

    const float* frame_data(int level, uint32_t frame) const {
        return flat_data.data() + level_offset[level] + frame * kGuardedFrameSize;
    }

    const float* base() const { return flat_data.data(); }

    // Flat offset for a given (level, frame, sample_index)
    uint32_t flat_index(int level, uint32_t frame, uint32_t sample_index) const {
        return level_offset[level] + frame * kGuardedFrameSize + sample_index;
    }
};

struct RendererTelemetry {
    enum Backend : uint32_t {
        BACKEND_NONE = 0,
        BACKEND_SCALAR = 1,
        BACKEND_HIGHWAY = 2,
        BACKEND_SIMD = BACKEND_HIGHWAY,
        BACKEND_ACCELERATE = 3,
    };

    std::atomic<uint32_t> backend{BACKEND_NONE};
    std::atomic<uint32_t> prepared_rebuilds{0};
    std::atomic<uint32_t> active_voice_count{0};
    std::atomic<uint32_t> active_render_unit_count{0};
    std::atomic<uint64_t> prepared_rebuild_us{0};
    std::atomic<uint64_t> pack_update_us{0};
    std::atomic<uint64_t> render_us{0};

    void reset_block() {
        backend.store(BACKEND_NONE, std::memory_order_relaxed);
        pack_update_us.store(0, std::memory_order_relaxed);
        render_us.store(0, std::memory_order_relaxed);
        active_voice_count.store(0, std::memory_order_relaxed);
        active_render_unit_count.store(0, std::memory_order_relaxed);
    }
};

inline uint64_t steady_clock_us_since(std::chrono::steady_clock::time_point start) {
    auto elapsed = std::chrono::steady_clock::now() - start;
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
}

// ---------------------------------------------------------------------------
// RenderUnit — SoA arrays for batched voice × unison processing
// ---------------------------------------------------------------------------

static constexpr int kMaxRenderUnits = 256; // 16 voices × 16 unison
static constexpr int kControlSubBlock = 8;  // 8-sample sub-blocks for position/warp smoothing
static constexpr int kMaxVoices = 16;
static constexpr int kMaxUnisonVoices = 16;
static constexpr int kDeClickSamples = 16;

struct RenderUnit {
    alignas(64) float phase[kMaxRenderUnits];
    alignas(64) float phase_inc[kMaxRenderUnits];       // base phase inc (no pitch mod)
    alignas(64) float drift_phase[kMaxRenderUnits];
    alignas(64) float drift_phase_inc[kMaxRenderUnits];
    alignas(64) float pan_l[kMaxRenderUnits];
    alignas(64) float pan_r[kMaxRenderUnits];
    alignas(64) float gain[kMaxRenderUnits];             // amplitude × unison_gain
    alignas(64) int32_t mip_level[kMaxRenderUnits];      // quantized mip (no blending)
    alignas(64) int32_t voice_idx[kMaxRenderUnits];      // maps to polyphonic voice index

    int active_count = 0;

    void clear() {
        active_count = 0;
        std::memset(phase, 0, sizeof(phase));
        std::memset(phase_inc, 0, sizeof(phase_inc));
        std::memset(drift_phase, 0, sizeof(drift_phase));
        std::memset(drift_phase_inc, 0, sizeof(drift_phase_inc));
        std::memset(pan_l, 0, sizeof(pan_l));
        std::memset(pan_r, 0, sizeof(pan_r));
        std::memset(gain, 0, sizeof(gain));
        std::memset(mip_level, 0, sizeof(mip_level));
        std::memset(voice_idx, 0, sizeof(voice_idx));
    }
};

// ---------------------------------------------------------------------------
// VoiceBlock — per-voice (not per-render-unit) audio-rate and smoothing data
// ---------------------------------------------------------------------------

struct VoiceBlock {
    const float* pitch_mod_audio[kMaxVoices] = {};
    const float* position_mod_audio[kMaxVoices] = {};
    const float* warp_mod_audio[kMaxVoices] = {};
    const float* voice_gain_audio[kMaxVoices] = {};
    vivid_wavetable::dsp::MotionSmoother* pos_smoother[kMaxVoices] = {};
    vivid_wavetable::dsp::MotionSmoother* warp_smoother[kMaxVoices] = {};
    float pitch_lane_base[kMaxVoices] = {};
    float position_lane_base[kMaxVoices] = {};
    float warp_lane_base[kMaxVoices] = {};

    // Sub-block smoothing: start and target for linear interpolation
    float pos_from[kMaxVoices] = {};
    float pos_to[kMaxVoices] = {};
    float warp_from[kMaxVoices] = {};
    float warp_to[kMaxVoices] = {};

    // Per-voice declick gain (0→1 ramp over kDeClickSamples)
    int declick_remaining[kMaxVoices] = {};

    int voice_count = 0;
};

// ---------------------------------------------------------------------------
// RenderParams — block-level configuration extracted from operator params
// ---------------------------------------------------------------------------

struct RenderParams {
    int warp_mode = 0;
    float amplitude = 0.3f;
    float position_base = 0.0f;
    float warp_base = 0.0f;
    float drift_amount = 0.0f;
    float drift_rate_hz = 0.18f;
    bool drift_enabled = false;
    float pos_smooth_coeff = 1.0f;
    float warp_smooth_coeff = 1.0f;
    int num_unison = 1;
    float unison_spread = 20.0f;
    float unison_stereo = 1.0f;
    int unison_spread_mode = 0;
    float detune_cents = 0.0f;
    float portamento_ms = 0.0f;
    int phase_reset_mode = 0;
    float start_phase = 0.0f;
    float phase_random = 0.0f;
    float stereo_phase_offset = 0.25f;
};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

inline int quantized_mip_level(float freq_hz, float sample_rate) {
    if (freq_hz <= 0.0f) return 0;
    float max_h = sample_rate / (2.0f * freq_hz);
    float level_f = std::log2(static_cast<float>(bank::kSamplesPerFrame / 2) / std::max(max_h, 1.0f));
    if (!(level_f >= 0.0f)) level_f = 0.0f;
    int level = static_cast<int>(std::round(level_f));
    return std::clamp(level, 0, bank::kNumMipLevels - 1);
}

// ---------------------------------------------------------------------------
// Renderer functions
// ---------------------------------------------------------------------------

// Scalar reference backend — always available.
void render_block_scalar(
    float* stereo_out,
    uint32_t frames,
    float sample_rate,
    RenderUnit& ru,
    VoiceBlock& vb,
    const PreparedWavetable& pwt,
    const RenderParams& params
);

// Highway SIMD backend — only available when VIVID_HAS_HIGHWAY is defined.
#ifdef VIVID_HAS_HIGHWAY
void render_block_simd(
    float* stereo_out,
    uint32_t frames,
    float sample_rate,
    RenderUnit& ru,
    VoiceBlock& vb,
    const PreparedWavetable& pwt,
    const RenderParams& params
);
#endif

// macOS Accelerate backend — optional hot-path renderer.
#ifdef VIVID_HAS_ACCELERATE
bool render_block_accelerate(
    float* stereo_out,
    uint32_t frames,
    float sample_rate,
    RenderUnit& ru,
    VoiceBlock& vb,
    const PreparedWavetable& pwt,
    const RenderParams& params
);
#endif

} // namespace vivid_wavetable::layer
