#include "operator_api/operator.h"
#include "operator_api/audio_dsp.h"
#include "operator_api/type_id.h"
#include "wavetable_bank.h"
#include "wavetable_dsp.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using vivid_wavetable::bank::Wavetable;
using vivid_wavetable::bank::build_builtin_wavetables;
using vivid_wavetable::bank::kBuiltinWavetableCount;
using vivid_wavetable::bank::load_wavetable_from_wav;
using namespace vivid_wavetable::dsp;

static constexpr int kCustomWavetableIndex = kBuiltinWavetableCount;

// =============================================================================
// WavetableOsc — polyphonic wavetable oscillator with per-voice channel output
// =============================================================================

struct WavetableOsc : vivid::OperatorBase, vivid::AudioProcessable {
    static constexpr const char* kName   = "WavetableOsc";
    static constexpr bool kTimeDependent = true;

    static constexpr int kMaxVoices = 16;

    // --- Parameters ---
    vivid::Param<int>   wavetable          {"wavetable",          0,     {"Basic", "Analog", "Digital", "Vocal", "Texture", "PWM", "Formant", "Harmonic", "Metallic", "Custom"}};
    vivid::Param<float> position           {"position",           0.0f,  0.0f, 1.0f};
    vivid::Param<float> amplitude          {"amplitude",          0.3f,  0.0f, 1.0f};
    vivid::Param<int>   warp_mode          {"warp_mode",          0,     {"None", "Sync", "BendPlus", "BendMinus", "Mirror", "Asym", "Quantize", "FM", "Flip"}};
    vivid::Param<float> warp_amount        {"warp_amount",        0.0f,  0.0f, 1.0f};
    vivid::Param<int>   unison_voices      {"unison_voices",      1,     1, 16};
    vivid::Param<float> unison_spread      {"unison_spread",      20.0f, 0.0f, 100.0f};
    vivid::Param<float> unison_stereo      {"unison_stereo",      1.0f,  0.0f, 1.0f};
    vivid::Param<int>   unison_spread_mode {"unison_spread_mode", 0,     {"Linear", "Exponential", "Random"}};
    vivid::Param<float> detune             {"detune",             0.0f,  0.0f, 50.0f};
    vivid::Param<float> portamento         {"portamento",         0.0f,  0.0f, 2000.0f};
    vivid::Param<int>   mod_type           {"mod_type",           0,     {"Off", "FM", "RM", "AM"}};
    vivid::Param<float> mod_depth          {"mod_depth",          0.0f,  0.0f, 1.0f};
    vivid::Param<vivid::FilePath> wav_file {"wav_file"};

    // --- Custom wavetable ---
    std::atomic<Wavetable*> custom_table_{nullptr};
    Wavetable* deferred_delete_ = nullptr;
    std::string last_wav_path_;

    ~WavetableOsc() {
        delete custom_table_.load(std::memory_order_relaxed);
        delete deferred_delete_;
    }

    void main_thread_update(double) override {
        if (deferred_delete_) { delete deferred_delete_; deferred_delete_ = nullptr; }
        if (wav_file.str_value != last_wav_path_) {
            last_wav_path_ = wav_file.str_value;
            if (last_wav_path_.empty()) {
                deferred_delete_ = custom_table_.exchange(nullptr, std::memory_order_release);
            } else {
                Wavetable* t = load_wavetable_from_wav(last_wav_path_);
                deferred_delete_ = custom_table_.exchange(t, std::memory_order_release);
            }
        }
    }

    // --- Per-voice state ---
    struct Voice {
        double phase        = 0;
        float  last_sample  = 0;  // FM warp feedback
        float  current_freq = 0;  // for portamento glide
        float  target_freq  = 0;
        bool   was_gated    = false;
    };
    Voice voices_[kMaxVoices] = {};

    Wavetable all_tables_[kBuiltinWavetableCount];

    WavetableOsc() {
        vivid::semantic_tag(position, "phase_01");
        vivid::semantic_intent(position, "wavetable_position");
        vivid::semantic_tag(amplitude, "amplitude_linear");
        vivid::semantic_tag(portamento, "time_milliseconds");
        vivid::semantic_unit(portamento, "ms");

        build_builtin_wavetables(all_tables_, kBuiltinWavetableCount);
    }

    // --- Registration ---

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        param_group(wavetable, "Core");
        param_group(wav_file,  "Core");
        param_group(position,  "Core");
        param_group(amplitude, "Core");
        param_group(warp_mode,   "Warp");
        param_group(warp_amount, "Warp");
        param_group(unison_voices,      "Unison");
        param_group(unison_spread,      "Unison");
        param_group(unison_stereo,      "Unison");
        param_group(unison_spread_mode, "Unison");
        param_group(detune,     "Output");
        param_group(portamento, "Portamento");
        param_group(mod_type,  "Modulation");
        param_group(mod_depth, "Modulation");

        out.push_back(&wavetable);
        out.push_back(&wav_file);
        out.push_back(&position);
        out.push_back(&amplitude);
        out.push_back(&warp_mode);
        out.push_back(&warp_amount);
        out.push_back(&unison_voices);
        out.push_back(&unison_spread);
        out.push_back(&unison_stereo);
        out.push_back(&unison_spread_mode);
        out.push_back(&detune);
        out.push_back(&portamento);
        out.push_back(&mod_type);
        out.push_back(&mod_depth);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back({"frequencies",  VIVID_PORT_SPREAD, VIVID_PORT_INPUT});    // 0
        out.push_back({"gates",        VIVID_PORT_SPREAD, VIVID_PORT_INPUT});    // 1
        out.push_back({"velocities",   VIVID_PORT_SPREAD, VIVID_PORT_INPUT});    // 2
        out.push_back({"pitch_mod",    VIVID_PORT_SPREAD, VIVID_PORT_INPUT});    // 3
        out.push_back({"position_mod", VIVID_PORT_SPREAD, VIVID_PORT_INPUT});    // 4
        out.push_back({"warp_mod",     VIVID_PORT_SPREAD, VIVID_PORT_INPUT});    // 5
        // N-channel audio input for FM/RM/AM from another oscillator
        out.push_back({"mod_input", VIVID_PORT_AUDIO, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});  // 6 (auto channels)
        // Audio-rate modulation inputs (N-channel, one per voice)
        out.push_back({"pitch_mod_audio", VIVID_PORT_AUDIO, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});    // 7
        out.push_back({"position_mod_audio", VIVID_PORT_AUDIO, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});    // 8
        out.push_back({"warp_mod_audio", VIVID_PORT_AUDIO, VIVID_PORT_INPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 0});    // 9
        // Output: N-channel audio, one channel per voice
        out.push_back({"output", VIVID_PORT_AUDIO, VIVID_PORT_OUTPUT,
                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, kMaxVoices});
    }

    // --- Helpers ---

    static float cents_to_ratio(float cents) {
        return std::pow(2.0f, cents / 1200.0f);
    }

    static float read_spread(const VividSpreadPort* sp, int slot, float fallback = 0.0f) {
        if (sp && sp->data && slot >= 0 && static_cast<uint32_t>(slot) < sp->length)
            return sp->data[slot];
        return fallback;
    }

    static float* resolve_mod_channel(float* buf, uint32_t ch_count, uint32_t voice, uint32_t frames) {
        if (!buf || ch_count == 0) return nullptr;
        uint32_t ch = (voice < ch_count) ? voice : ch_count - 1;
        return buf + ch * frames;
    }

    // --- Main process ---

    void process_audio(const VividAudioContext* ctx) override {
        uint32_t frames = ctx->buffer_size;
        float sr = static_cast<float>(ctx->sample_rate);

        int   wt_idx   = std::clamp(wavetable.int_value(), 0, kCustomWavetableIndex);
        float pos      = position.value;
        float amp      = amplitude.value;
        int   warp_m   = warp_mode.int_value();
        float warp_a   = warp_amount.value;
        int   num_uni  = unison_voices.int_value();
        float uni_spr  = unison_spread.value;
        float det      = detune.value;
        float porta_ms = portamento.value;
        int   mtype    = mod_type.int_value();
        float mdepth   = mod_depth.value;

        const Wavetable* wt_ptr;
        if (wt_idx == kCustomWavetableIndex) {
            wt_ptr = custom_table_.load(std::memory_order_acquire);
            if (!wt_ptr) wt_ptr = &all_tables_[0];
        } else {
            wt_ptr = &all_tables_[wt_idx];
        }
        const Wavetable& wt = *wt_ptr;

        // Read input spreads
        const VividSpreadPort* freq_sp     = ctx->input_spreads ? &ctx->input_spreads[0] : nullptr;
        const VividSpreadPort* gates_sp    = ctx->input_spreads ? &ctx->input_spreads[1] : nullptr;
        const VividSpreadPort* pitch_sp    = ctx->input_spreads ? &ctx->input_spreads[3] : nullptr;
        const VividSpreadPort* pos_mod_sp  = ctx->input_spreads ? &ctx->input_spreads[4] : nullptr;
        const VividSpreadPort* warp_mod_sp = ctx->input_spreads ? &ctx->input_spreads[5] : nullptr;

        uint32_t voice_count = freq_sp ? freq_sp->length : 0;
        if (voice_count > kMaxVoices) voice_count = kMaxVoices;

        // Portamento rate
        float porta_rate = 1.0f;
        if (porta_ms > 0.0f) {
            float porta_samples = porta_ms * 0.001f * sr;
            porta_rate = 1.0f - std::exp(-4.0f / porta_samples);
        }

        // Modulation input (N-channel audio from another oscillator)
        // Port layout: [0-5] spread, [6] mod_input, [7-9] audio-rate mods
        float* mod_buf = (mtype > 0 && mdepth > 0.0f && ctx->input_buffers[6])
                         ? ctx->input_buffers[6] : nullptr;
        uint32_t mod_channels = mod_buf && ctx->input_channel_counts
                                ? ctx->input_channel_counts[6] : 0;

        // Audio-rate modulation buffers
        float* pitch_mod_buf = ctx->input_buffers[7];
        uint32_t pitch_mod_ch = pitch_mod_buf && ctx->input_channel_counts
                                ? ctx->input_channel_counts[7] : 0;
        float* pos_mod_buf = ctx->input_buffers[8];
        uint32_t pos_mod_ch = pos_mod_buf && ctx->input_channel_counts
                              ? ctx->input_channel_counts[8] : 0;
        float* warp_mod_buf = ctx->input_buffers[9];
        uint32_t warp_mod_ch = warp_mod_buf && ctx->input_channel_counts
                               ? ctx->input_channel_counts[9] : 0;

        // Zero all output channels
        float* out_buf = ctx->output_buffers[0];
        std::memset(out_buf, 0, kMaxVoices * frames * sizeof(float));

        for (uint32_t vi = 0; vi < voice_count; ++vi) {
            float gate = read_spread(gates_sp, vi);
            float freq_target = read_spread(freq_sp, vi);
            if (freq_target <= 0.0f) continue;

            Voice& v = voices_[vi];

            // Detect gate-on transition for phase reset
            bool gate_on = (gate > 0.5f);
            if (gate_on && !v.was_gated) {
                v.phase = 0.0;
                v.last_sample = 0.0f;
                v.current_freq = freq_target;
                v.target_freq = freq_target;
            }
            v.was_gated = gate_on;
            v.target_freq = freq_target;

            if (!gate_on) continue;  // Only generate audio for gated voices

            // Output channel for this voice
            float* ch_out = out_buf + vi * frames;

            // Spread-rate modulation (fallback when audio not connected)
            float pitch_offset_sp = read_spread(pitch_sp, vi);
            float pos_mod_sp_val  = read_spread(pos_mod_sp, vi);
            float warp_mod_sp_val = read_spread(warp_mod_sp, vi);

            // Audio-rate modulation channels for this voice
            float* mod_ch          = resolve_mod_channel(mod_buf, mod_channels, vi, frames);
            float* pitch_mod_voice = resolve_mod_channel(pitch_mod_buf, pitch_mod_ch, vi, frames);
            float* pos_mod_voice   = resolve_mod_channel(pos_mod_buf, pos_mod_ch, vi, frames);
            float* warp_mod_voice  = resolve_mod_channel(warp_mod_buf, warp_mod_ch, vi, frames);

            // Generate samples for this voice
            for (uint32_t s = 0; s < frames; ++s) {
                // Portamento glide
                if (porta_ms > 0.0f && v.current_freq != v.target_freq) {
                    v.current_freq += (v.target_freq - v.current_freq) * porta_rate;
                    if (std::abs(v.current_freq - v.target_freq) < 0.01f)
                        v.current_freq = v.target_freq;
                }

                // Per-sample modulation: audio buffer if connected, else spread value
                float pitch_offset = pitch_mod_voice ? pitch_mod_voice[s] : pitch_offset_sp;
                float pos_mod_val  = pos_mod_voice   ? pos_mod_voice[s]   : pos_mod_sp_val;
                float warp_mod_val = warp_mod_voice  ? warp_mod_voice[s]  : warp_mod_sp_val;

                float base_freq = v.current_freq *
                    cents_to_ratio(det) *
                    std::pow(2.0f, pitch_offset / 12.0f);
                if (!std::isfinite(base_freq) || base_freq <= 0.0f)
                    base_freq = v.current_freq;

                // FM: modulator offsets phase increment (±4 octaves at full depth)
                float phase_inc = base_freq / sr;
                if (mtype == 1 && mod_ch) {  // FM
                    phase_inc += mod_ch[s] * mdepth * 4.0f * (base_freq / sr);
                }

                // Effective warp
                float eff_warp = std::clamp(warp_a + warp_mod_val, 0.0f, 1.0f);

                // Effective position
                float eff_pos = std::clamp(pos + pos_mod_val, 0.0f, 1.0f);

                // Phase warp + wavetable sample
                float warped = warp_phase(static_cast<float>(v.phase), warp_m,
                                          eff_warp, v.last_sample);
                float sig = wt.sample(warped, eff_pos, base_freq, sr);
                v.last_sample = sig;

                // RM/AM: apply after carrier generation
                if (mod_ch) {
                    if (mtype == 2) {  // RM: bipolar multiply
                        sig *= mod_ch[s];
                    } else if (mtype == 3) {  // AM: unipolar
                        sig *= 1.0f + mdepth * mod_ch[s];
                    }
                }

                ch_out[s] = sig * amp;

                // Advance phase
                v.phase += static_cast<double>(phase_inc);
                if (v.phase >= 1.0) v.phase -= 1.0;
                if (v.phase < 0.0) v.phase += 1.0;
                if (!std::isfinite(v.phase)) v.phase = 0.0;
            }
        }
    }
};

VIVID_REGISTER(WavetableOsc)
