#include "operator_api/operator.h"
#include "operator_api/thumbnail.h"
#include "operator_api/draw_plot_helpers.h"
#include "operator_api/type_id.h"
#include "lane_audio_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// =============================================================================
// VoiceDrive — lane-preserving per-voice soft drive / glue stage
// =============================================================================

/**
 * @brief Polyphonic per-voice soft drive for body, glue, and controlled harmonic density.
 *
 * Processes one audio lane at a time and preserves the incoming voice layout so the
 * result can continue through per-note filters, envelopes, and reduction stages before
 * it reaches VoiceMixer. Velocity can optionally increase the effective drive amount.
 *
 * @input input Per-voice audio to saturate before VoiceMixer.
 * @input velocities Per-voice velocity values used for dynamic drive response.
 * @output output Per-voice driven audio with the same channel layout as the input.
 * @recipe WavetableOsc/output -> VoiceDrive/input -> VoiceMixer/input
 * @recipe AnalogOsc/output -> VoiceDrive/input with PolyVoiceAllocator/velocities -> VoiceDrive/velocities
 * @pitfall VoiceDrive belongs on the per-voice side of the graph. Put it before VoiceMixer, not after the stereo sum.
 * @family voice_shaper
 * @best_used_with VoiceMixer, PolyVoiceAllocator, EnvelopeAu
 * @common_companions WavetableOsc, AnalogOsc, SubOsc
 */
struct VoiceDrive : vivid::OperatorBase, vivid::AudioProcessable {
    static constexpr const char* kName = "VoiceDrive";
    static constexpr bool kTimeDependent = true;
    static constexpr VividLaneBehavior kLaneBehavior = VIVID_LANE_POINTWISE;
    static constexpr bool kStrategyIndependent = true;

    vivid::Param<float> drive {"drive", 0.22f, 0.0f, 1.0f};
    vivid::Param<float> tone {"tone", 0.52f, 0.0f, 1.0f};
    vivid::Param<float> mix {"mix", 1.0f, 0.0f, 1.0f};
    vivid::Param<float> output_level {"output_level", 1.0f, 0.0f, 2.0f};
    vivid::Param<float> velocity_to_drive {"velocity_to_drive", 0.30f, 0.0f, 1.0f};

    struct LaneState {
        float lp = 0.0f;
    };

    VoiceDrive() {
        vivid::semantic_tag(drive, "probability_01");
        vivid::semantic_shape(drive, "scalar");
        vivid::semantic_intent(drive, "drive_amount");
        vivid::description(drive, "Soft-saturation amount before shaping.");

        vivid::semantic_tag(tone, "probability_01");
        vivid::semantic_shape(tone, "scalar");
        vivid::semantic_intent(tone, "brightness");
        vivid::description(tone, "Post-drive tone balance (0 = warm, 1 = open).");

        vivid::semantic_tag(mix, "probability_01");
        vivid::semantic_shape(mix, "scalar");
        vivid::semantic_intent(mix, "wet_mix");
        vivid::description(mix, "Dry/wet blend between the clean and driven signals.");

        vivid::semantic_tag(output_level, "amplitude_linear");
        vivid::semantic_shape(output_level, "scalar");
        vivid::semantic_intent(output_level, "post_gain");
        vivid::description(output_level, "Final output trim after the drive stage.");

        vivid::semantic_tag(velocity_to_drive, "probability_01");
        vivid::semantic_shape(velocity_to_drive, "scalar");
        vivid::semantic_intent(velocity_to_drive, "velocity_response");
        vivid::description(velocity_to_drive, "How strongly note velocity increases the drive amount.");
    }

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        param_group(drive, "Drive");
        param_group(tone, "Drive");
        param_group(mix, "Drive");
        param_group(output_level, "Drive");
        param_group(velocity_to_drive, "Dynamics");

        out.push_back(&drive);
        out.push_back(&tone);
        out.push_back(&mix);
        out.push_back(&output_level);
        out.push_back(&velocity_to_drive);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        VividPortDescriptor input_port{"input", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_INPUT,
                                       VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 1, 0.0f};
        vivid::semantic_tag(input_port, "audio_signal");
        vivid::semantic_shape(input_port, "audio_buffer");
        vivid::semantic_intent(input_port, "audio_input");
        vivid::description(input_port, "Per-voice audio to saturate before mixing.");
        out.push_back(input_port);

        VividPortDescriptor velocities_port{"velocities", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT};
        vivid::semantic_tag(velocities_port, "velocity_01");
        vivid::semantic_shape(velocities_port, "lane_array");
        vivid::semantic_intent(velocities_port, "per_note_velocity");
        vivid::description(velocities_port, "Per-note velocity values that can push the drive harder.");
        out.push_back(velocities_port);

        VividPortDescriptor output_port{"output", VIVID_PORT_AUDIO_BUFFER, VIVID_PORT_OUTPUT,
                                        VIVID_PORT_TRANSPORT_AUDIO_BUFFER, 0, nullptr, 1, 0.0f};
        vivid::semantic_tag(output_port, "audio_signal");
        vivid::semantic_shape(output_port, "audio_buffer");
        vivid::semantic_intent(output_port, "audio_output");
        vivid::description(output_port, "Driven per-voice audio with the original channel layout preserved.");
        out.push_back(output_port);
    }

    void draw_thumbnail(const VividThumbnailContext* ctx) override {
        if (!ctx || !ctx->draw.opaque) return;
        auto& d = const_cast<VividDrawAPI&>(ctx->draw);
        void* o = d.opaque;

        float w = static_cast<float>(ctx->thumbnail_logical_width ? ctx->thumbnail_logical_width : ctx->thumbnail_width);
        float h = static_cast<float>(ctx->thumbnail_logical_height ? ctx->thumbnail_logical_height : ctx->thumbnail_height);

        float drv = (ctx->param_count > 0) ? std::clamp(ctx->param_values[0], 0.0f, 1.0f) : 0.22f;
        float ton = (ctx->param_count > 1) ? std::clamp(ctx->param_values[1], 0.0f, 1.0f) : 0.52f;

        vivid::draw_plot::draw_thumb_background(d, o, w, h);
        vivid::draw_plot::draw_thumb_label(d, o, 6.0f, 4.0f, "GLUE",
            {0.78f, 0.56f, 0.36f, 0.92f}, 0.82f);

        float bar_top = 22.0f;
        float bar_h = h - bar_top - 7.0f;
        vivid::draw_plot::draw_scalar_meter(d, o, w * 0.12f, bar_top, w * 0.18f, bar_h, drv,
            {0.16f, 0.16f, 0.19f, 0.8f},
            {0.55f, 0.42f, 0.28f, 0.86f},
            {0.90f, 0.62f, 0.36f, 0.92f},
            2.0f, -1.0f);
        vivid::draw_plot::draw_scalar_meter(d, o, w * 0.70f, bar_top, w * 0.18f, bar_h, ton,
            {0.16f, 0.16f, 0.19f, 0.8f},
            {0.36f, 0.36f, 0.40f, 0.86f},
            {0.84f, 0.80f, 0.70f, 0.92f},
            2.0f, -1.0f);
        if (d.draw_text) {
            d.draw_text(o, w * 0.12f, bar_top + bar_h + 1.0f, "DRV", {0.55f, 0.60f, 0.65f, 0.7f}, 0.55f);
            d.draw_text(o, w * 0.70f, bar_top + bar_h + 1.0f, "TONE", {0.55f, 0.60f, 0.65f, 0.7f}, 0.55f);
        }
    }

    void process_audio(const VividAudioContext* ctx) override {
        auto& lane = *vivid_lane_state(ctx, ctx->lane_id, LaneState);

        float* in = ctx->input_buffers[0];
        float* out = ctx->output_buffers[0];
        uint32_t frames = ctx->buffer_size;
        uint32_t channels = ctx->input_channel_counts ? ctx->input_channel_counts[0] : 1;
        if (channels == 0) channels = 1;

        const VividLanePort* vel_lane = ctx->input_lanes ? &ctx->input_lanes[0] : nullptr;
        float velocity = vivid_wavetable::lane_audio::clamp01(
            vivid_wavetable::lane_audio::read_lane(vel_lane, ctx->lane_index, 1.0f));

        float vel_drive = (velocity - 0.5f) * 2.0f;
        float effective_drive = vivid_wavetable::lane_audio::clamp01(
            drive.value + vel_drive * velocity_to_drive.value * 0.25f);
        float drive_gain = 1.0f + effective_drive * 11.0f;
        float norm = 1.0f / std::max(std::tanh(drive_gain), 1.0e-6f);

        float brightness = vivid_wavetable::lane_audio::clamp01(tone.value);
        float cutoff = 180.0f + std::pow(brightness, 1.45f) * 12000.0f;
        float lp_coeff = vivid_wavetable::lane_audio::one_pole_coeff(
            static_cast<float>(ctx->sample_rate), cutoff, 30.0f);

        float wet = mix.value;
        float dry = 1.0f - wet;
        float out_gain = output_level.value;

        for (uint32_t ch = 0; ch < channels; ++ch) {
            float* ch_in = in + ch * frames;
            float* ch_out = out + ch * frames;
            for (uint32_t i = 0; i < frames; ++i) {
                float dry_sig = ch_in[i];
                float saturated = std::tanh(dry_sig * drive_gain) * norm;
                lane.lp += lp_coeff * (saturated - lane.lp);
                float hp = saturated - lane.lp;
                float colored = lane.lp * (1.28f - brightness * 0.62f) + hp * (0.08f + brightness * 1.85f);
                float wet_sig = colored * (0.86f - effective_drive * 0.10f);
                ch_out[i] = (dry_sig * dry + wet_sig * wet) * out_gain;
            }
        }
    }
};

VIVID_REGISTER(VoiceDrive)
VIVID_THUMBNAIL(VoiceDrive)
