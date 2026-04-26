#pragma once

#include "operator_api/adsr.h"
#include "operator_api/note_types.h"
#include "operator_api/operator.h"
#include "operator_api/thumbnail.h"
#include "operator_api/editor_ui.h"
#include "operator_api/editor_keys.h"
#include "operator_api/gpu_common.h"
#include "operator_api/voice_allocator.h"
#include "wavetable_bank.h"
#include "wavetable_dsp.h"

#include <array>
#include <atomic>
#include <string>

using vivid_wavetable::bank::Wavetable;
using vivid_wavetable::bank::kBuiltinWavetableCount;

/**
 * @brief Polyphonic wavetable oscillator with family/member selection, warp, drift, unison, and oscillator interaction.
 *
 * Drive with `notes_in` from any note source — voices are allocated internally
 * with a built-in ADSR. Stereo `output` carries the summed mix; advanced
 * `voices_out` (per-voice multichannel) plus the four `voice_*` control lanes
 * expose per-voice state for downstream VoiceMixer/VoiceDrive/Filter routing.
 * Use WavetableLayer for the production no-interaction wavetable path; use this
 * operator when a patch needs mod_input, FM/PM/RM/AM, or feedback-style warp.
 *
 * @input notes_in Native note stream — canonical input for note sources.
 * @input mod_input Audio-rate modulation input for oscillator interaction.
 * @input pitch_mod_audio Audio-rate per-voice pitch modulation.
 * @input position_mod_audio Audio-rate per-voice wavetable position modulation.
 * @input warp_mod_audio Audio-rate per-voice warp modulation.
 * @output output Stereo summed audio.
 * @output voices_out Advanced: per-voice audio channels (note_id sorted).
 * @tip Drive position_mod_audio or warp_mod_audio from per-note envelopes when you want note-shaped timbral movement instead of a global macro sweep.
 * @recipe Tracker/notes_out -> WavetableOsc/notes_in
 * @recipe ChordProgression/notes_out -> WavetableOsc/notes_in
 * @recipe EnvelopeAu/value -> WavetableOsc/position_mod_audio
 * @pitfall Use mod_input on the carrier oscillator and keep the modulator readable in the graph; interaction happens before VoiceMixer, not on the summed stereo bus.
 * @family voice_source
 * @best_used_with Tracker, ChordProgression, NoteBreakout, EnvelopeAu, VoiceMixer
 * @common_companions Filter, AnalogOsc, SubOsc
 */
struct WavetableOsc : vivid::OperatorBase, vivid::AudioProcessable {
    static constexpr const char* kName = "WavetableOsc";
    static constexpr bool kTimeDependent = true;

    static constexpr int kMaxVoices = 16;
    static constexpr int kMaxUnisonVoices = 16;
    static constexpr int kMaxStereoPairVoices = kMaxVoices / 2;
    static constexpr int kDeClickSamples = 16;
    static constexpr float kMaxDriftCents = 8.0f;
    static constexpr uint32_t kThumbWTCols = 128;

    enum WavetableSourceMode {
        SOURCE_BUILTIN = 0,
        SOURCE_CUSTOM = 1,
    };

    enum PhaseResetMode {
        PHASE_FREE_RUN = 0,
        PHASE_RESET = 1,
        PHASE_RANDOMIZED = 2,
    };

    enum UnisonOutputMode {
        UNISON_OUTPUT_MONO_MIX = 0,
        UNISON_OUTPUT_STEREO_PAIRS = 1,
    };

    enum InteractionMode {
        INTERACTION_OFF = 0,
        INTERACTION_FM = 1,
        INTERACTION_PM = 2,
        INTERACTION_RM = 3,
        INTERACTION_AM = 4,
    };

    vivid::Param<int> wavetable_source {"wavetable_source", 0, {"Builtin", "Custom"}};
    vivid::Param<int> wavetable_family {"wavetable_family", 0, {"AnalogWarm", "BrightDigital", "VocalFormant", "Metallic", "HarmonicSpectral", "TextureMotion"}};
    vivid::Param<int> wavetable_member {"wavetable_member", 0, {"Core", "Soft", "Rich", "Hollow", "Sweep", "Glass", "Edge", "Air"}};
    vivid::Param<vivid::FilePath> wav_file {"wav_file"};
    vivid::Param<float> position {"position", 0.0f, 0.0f, 1.0f};
    vivid::Param<float> amplitude {"amplitude", 0.3f, 0.0f, 1.0f};
    vivid::Param<int> warp_mode {"warp_mode", 0, {"None", "Sync", "BendPlus", "BendMinus", "Mirror", "Asym", "Quantize", "FM", "Flip"}};
    vivid::Param<float> warp_amount {"warp_amount", 0.0f, 0.0f, 1.0f};
    vivid::Param<float> position_smooth_ms {"position_smooth_ms", 8.0f, 0.0f, 40.0f};
    vivid::Param<float> warp_smooth_ms {"warp_smooth_ms", 8.0f, 0.0f, 40.0f};
    vivid::Param<int> phase_reset_mode {"phase_reset_mode", 0, {"FreeRun", "Reset", "Randomized"}};
    vivid::Param<float> start_phase {"start_phase", 0.0f, 0.0f, 1.0f};
    vivid::Param<float> phase_random {"phase_random", 0.0f, 0.0f, 1.0f};
    vivid::Param<float> stereo_phase_offset {"stereo_phase_offset", 0.25f, 0.0f, 1.0f};
    vivid::Param<float> drift_amount {"drift_amount", 0.0f, 0.0f, 1.0f};
    vivid::Param<float> drift_rate_hz {"drift_rate_hz", 0.18f, 0.02f, 2.0f};
    vivid::Param<int> unison_voices {"unison_voices", 1, 1, 16};
    vivid::Param<float> unison_spread {"unison_spread", 20.0f, 0.0f, 100.0f};
    vivid::Param<float> unison_stereo {"unison_stereo", 1.0f, 0.0f, 1.0f};
    vivid::Param<int> unison_spread_mode {"unison_spread_mode", 0, {"Linear", "Exponential", "Random"}};
    vivid::Param<int> unison_output_mode {"unison_output_mode", 0, {"MonoMix", "StereoPairs"}};
    vivid::Param<float> detune {"detune", 0.0f, 0.0f, 50.0f};
    vivid::Param<float> portamento {"portamento", 0.0f, 0.0f, 2000.0f};
    vivid::Param<int> interaction_mode {"interaction_mode", 0, {"Off", "FM", "PM", "RM", "AM"}};
    vivid::Param<float> interaction_depth {"interaction_depth", 0.0f, 0.0f, 1.0f};
    vivid::Param<float> interaction_input_gain {"interaction_input_gain", 1.0f, 0.0f, 4.0f};
    vivid::Param<float> interaction_tracking {"interaction_tracking", 1.0f, 0.0f, 1.0f};

    // ADSR for the MIDI-driven path. Lane-array driven graphs typically
    // run their envelope upstream (e.g., via VoiceMixer's amp_env_audio).
    vivid::Param<float> attack  {"attack",  0.005f, 0.001f, 5.0f};
    vivid::Param<float> decay   {"decay",   0.1f,   0.001f, 5.0f};
    vivid::Param<float> sustain {"sustain", 0.8f,   0.0f,   1.0f};
    vivid::Param<float> release {"release", 0.2f,   0.001f, 5.0f};

    std::atomic<Wavetable*> custom_table_{nullptr};
    Wavetable* deferred_delete_ = nullptr;
    std::string last_wav_path_;

    WGPURenderPipeline thumb_pipeline_ = nullptr;
    WGPUBindGroup thumb_bind_group_ = nullptr;
    WGPUBindGroupLayout thumb_bind_layout_ = nullptr;
    WGPUBuffer thumb_uniform_buf_ = nullptr;
    WGPUShaderModule thumb_shader_ = nullptr;
    WGPUPipelineLayout thumb_pipe_layout_ = nullptr;
    WGPUSampler thumb_sampler_ = nullptr;
    WGPUTexture thumb_wt_tex_ = nullptr;
    WGPUTextureView thumb_wt_view_ = nullptr;
    WGPUTextureFormat thumb_pipeline_format_ = WGPUTextureFormat_Undefined;
    int thumb_wt_family_ = -1;
    int thumb_wt_member_ = -1;
    uint32_t thumb_wt_frames_ = 0;

    struct Voice {
        double phase[kMaxUnisonVoices] = {};
        double drift_phase[kMaxUnisonVoices] = {};
        float last_sample[kMaxUnisonVoices] = {};
        float current_freq = 0.0f;
        float target_freq = 0.0f;
        vivid_wavetable::dsp::MotionSmoother pos_smoother;
        vivid_wavetable::dsp::MotionSmoother warp_smoother;
        vivid_wavetable::dsp::DCBlocker interaction_dc;
        bool was_gated = false;
        bool initialized = false;
        int declick_remaining = 0;
    };

    // MIDI-driven path: per-slot ADSR envelope state. The voice's phase /
    // unison / drift state is held in the lane-state Voice (above), keyed by
    // a synthetic lane_id derived from the slot index.
    struct MidiVoice {
        vivid::adsr::State env;
    };
    MidiVoice midi_voices_[kMaxVoices] = {};
    vivid::VoiceAllocator<kMaxVoices> midi_allocator_;
    uint64_t midi_frame_counter_ = 0;
    static constexpr uint32_t kMidiLaneIdBase = 0xCA00FE00u;  // synthetic lane-id namespace

    WavetableOsc();
    ~WavetableOsc();

    void collect_params(std::vector<vivid::ParamBase*>& out) override;
    void collect_ports(std::vector<VividPortDescriptor>& out) override;
    void prepare_instance_assets() override;
    void main_thread_update(double) override;
    void draw_thumbnail(const VividThumbnailContext* ctx) override;
    void process_audio(const VividAudioContext* ctx) override;

    // The lane-driven render body. process_audio() dispatches to this when
    // the user has wired lane-array note inputs, or after process_audio_midi()
    // has built synthetic lanes from MIDI events.
    void process_audio_lane_driven(const VividAudioContext* ctx);
    // The MIDI-driven entry: ingest MIDI events, build synthetic lanes,
    // call process_audio_lane_driven, then apply ADSR + sum to stereo.
    void process_audio_midi(const VividAudioContext* ctx);

    // Dedicated editor window. 1200×700 browser + preview + scatter
    // layout; see wavetable-osc.md for the design spec.
    static VividEditorMetadata editor_metadata();
    void draw_editor(VividEditorContext* ctx);

    // Editor UI state. Public so tests can arrange; mirrors the pattern
    // used by the other Tier-3 adopters.
    bool editor_drag_position_  = false;
    bool editor_drag_unison_    = false;
    int  editor_drag_voice_idx_ = -1;  // which voice is being dragged in the scatter

    // Live effective position (base + position_mod) sampled once per
    // audio buffer for voice 0. Written on the audio thread, read on
    // the UI thread; atomic<float> covers the narrow race.
    std::atomic<float> editor_effective_position_{0.0f};

    void release_thumb_gpu();
    void upload_wavetable_texture(WGPUDevice device, WGPUQueue queue, int family, int member);
    void rebuild_thumb_pipeline(const VividThumbnailContext* ctx);

    const Wavetable* resolve_table() const;
    static const std::array<Wavetable, kBuiltinWavetableCount>& builtin_tables();

    static float wrap_phase(double phase);
    static float smoothing_coeff(float sample_rate, float smooth_ms);
    static uint32_t hash_u32(uint32_t x);
    static float hash01(uint32_t seed);
    static float normalized_unison_position(int index, int count);
    static float unison_detune_offset(int index, int count, float spread_cents, int spread_mode, uint32_t lane_seed);
    static float unison_pan_position(int index, int count, float stereo_depth);
    static float base_phase_offset(int index, int count, bool stereo_pairs, float stereo_phase, uint32_t lane_seed);
    static float stereo_pair_phase_shift(int index, int count, float stereo_phase, uint32_t lane_seed);
    static float gate_on_phase(PhaseResetMode mode,
                               float start_phase_value,
                               float phase_random_amount,
                               float base_offset,
                               int index,
                               uint32_t lane_seed);
};
