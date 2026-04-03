#include "wavetable_osc_internal.h"

#include "operator_api/thumbnail.h"

#include <cstring>
#include <vector>

namespace {

constexpr const char* kThumbShader = R"(
struct Uniforms {
    position: f32,
    warp_amount: f32,
    frame_count: f32,
    source_mode: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var wt_tex: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let fs = fullscreenTriangle(vi, true);
    var out: VertexOutput;
    out.position = fs.position;
    out.uv = fs.uv;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let uv = input.uv;
    let bg = vec4f(0.07, 0.08, 0.09, 0.9);

    let n_frames = max(u.frame_count, 1.0);
    let n_slices = 20.0;
    let pos = clamp(u.position, 0.0, 1.0);

    let margin_x = 0.08;
    let base_width = 1.0 - 2.0 * margin_x;
    let depth_shrink = 0.55;
    let y_bottom = 0.85;
    let y_top = 0.1;
    let wave_amp = 0.18;

    var color = bg;

    for (var i = 0u; i < u32(n_slices); i = i + 1u) {
        let t = f32(i) / (n_slices - 1.0);
        let frame_t = f32(i) / (n_slices - 1.0);
        let width = base_width * mix(depth_shrink, 1.0, t);
        let cx = 0.5;
        let left = cx - width * 0.5;
        let right = cx + width * 0.5;
        let baseline_y = mix(y_top, y_bottom, t);
        let frame_dist = abs(frame_t - pos);
        let is_active = f32(frame_dist < 0.5 / n_slices);
        let local_x = (uv.x - left) / width;
        if (local_x < 0.0 || local_x > 1.0) {
            continue;
        }

        let tex_uv = vec2f(local_x, frame_t);
        let sample_val = textureSampleLevel(wt_tex, samp, tex_uv, 0.0).r;
        let wave_y = baseline_y - sample_val * wave_amp * mix(0.6, 1.0, t);
        let depth_alpha = mix(0.15, 0.7, t * t);
        let thickness = mix(0.004, 0.012, t);
        let dist_to_line = abs(uv.y - wave_y);

        if (dist_to_line < thickness) {
            let line_alpha = (1.0 - dist_to_line / thickness) * depth_alpha;
            var line_col: vec3f;
            if (is_active > 0.5) {
                line_col = vec3f(0.95, 0.78, 0.35);
            } else {
                line_col = mix(vec3f(0.30, 0.40, 0.55), vec3f(0.50, 0.70, 0.90), t);
            }
            let src = vec4f(line_col, line_alpha);
            color = vec4f(mix(color.rgb, src.rgb, src.a), max(color.a, src.a * 0.5 + color.a));
        }

        if (is_active > 0.5 && uv.y > wave_y && uv.y < baseline_y) {
            let fill_t = (uv.y - wave_y) / max(baseline_y - wave_y, 0.001);
            let fill_alpha = 0.15 * (1.0 - fill_t);
            let fill_col = vec3f(0.95, 0.78, 0.35);
            color = vec4f(mix(color.rgb, fill_col, fill_alpha), max(color.a, fill_alpha * 0.5 + color.a));
        }
    }

    return color;
}
)";

} // namespace

void WavetableOsc::release_thumb_gpu() {
    vivid::gpu::release(thumb_pipeline_);
    vivid::gpu::release(thumb_bind_group_);
    vivid::gpu::release(thumb_bind_layout_);
    vivid::gpu::release(thumb_uniform_buf_);
    vivid::gpu::release(thumb_shader_);
    vivid::gpu::release(thumb_pipe_layout_);
    vivid::gpu::release(thumb_sampler_);
    vivid::gpu::release(thumb_wt_view_);
    vivid::gpu::release(thumb_wt_tex_);
}

void WavetableOsc::upload_wavetable_texture(WGPUDevice device, WGPUQueue queue, int family, int member) {
    vivid::gpu::release(thumb_wt_view_);
    vivid::gpu::release(thumb_wt_tex_);
    vivid::gpu::release(thumb_bind_group_);

    const auto& tables = builtin_tables();
    int idx = family * 8 + member;
    if (idx < 0 || idx >= static_cast<int>(tables.size())) return;
    const auto& wt = tables[static_cast<size_t>(idx)];
    uint32_t n_frames = wt.frame_count;
    if (n_frames == 0) return;

    thumb_wt_frames_ = n_frames;

    WGPUTextureDescriptor tex_desc{};
    tex_desc.label = vivid_sv("WT Thumb Tex");
    tex_desc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
    tex_desc.dimension = WGPUTextureDimension_2D;
    tex_desc.size = {kThumbWTCols, n_frames, 1};
    tex_desc.format = WGPUTextureFormat_R16Float;
    tex_desc.mipLevelCount = 1;
    tex_desc.sampleCount = 1;
    thumb_wt_tex_ = wgpuDeviceCreateTexture(device, &tex_desc);

    WGPUTextureViewDescriptor view_desc{};
    view_desc.label = vivid_sv("WT Thumb View");
    view_desc.format = WGPUTextureFormat_R16Float;
    view_desc.dimension = WGPUTextureViewDimension_2D;
    view_desc.mipLevelCount = 1;
    view_desc.arrayLayerCount = 1;
    thumb_wt_view_ = wgpuTextureCreateView(thumb_wt_tex_, &view_desc);

    constexpr uint32_t spf = vivid_wavetable::bank::kSamplesPerFrame;
    std::vector<uint16_t> pixels(kThumbWTCols * n_frames);
    for (uint32_t frame = 0; frame < n_frames; ++frame) {
        for (uint32_t col = 0; col < kThumbWTCols; ++col) {
            float phase = (static_cast<float>(col) + 0.5f) / static_cast<float>(kThumbWTCols);
            uint32_t s = static_cast<uint32_t>(phase * spf) % spf;
            float val = wt.data[frame * spf + s];
            uint32_t f32bits;
            std::memcpy(&f32bits, &val, 4);
            uint32_t sign = (f32bits >> 16) & 0x8000;
            int32_t exp_val = static_cast<int32_t>((f32bits >> 23) & 0xFF) - 127 + 15;
            uint32_t frac = (f32bits >> 13) & 0x3FF;
            uint16_t f16;
            if (exp_val <= 0) {
                f16 = static_cast<uint16_t>(sign);
            } else if (exp_val >= 31) {
                f16 = static_cast<uint16_t>(sign | 0x7C00);
            } else {
                f16 = static_cast<uint16_t>(sign | (exp_val << 10) | frac);
            }
            pixels[frame * kThumbWTCols + col] = f16;
        }
    }

    WGPUTexelCopyTextureInfo dst{};
    dst.texture = thumb_wt_tex_;
    dst.mipLevel = 0;
    dst.origin = {0, 0, 0};
    dst.aspect = WGPUTextureAspect_All;
    WGPUTexelCopyBufferLayout layout{};
    layout.offset = 0;
    layout.bytesPerRow = kThumbWTCols * 2;
    layout.rowsPerImage = n_frames;
    WGPUExtent3D extent = {kThumbWTCols, n_frames, 1};
    wgpuQueueWriteTexture(queue, &dst, pixels.data(),
                          pixels.size() * sizeof(uint16_t), &layout, &extent);

    thumb_wt_family_ = family;
    thumb_wt_member_ = member;
}

void WavetableOsc::rebuild_thumb_pipeline(const VividThumbnailContext* ctx) {
    release_thumb_gpu();
    thumb_wt_family_ = -1;
    thumb_wt_member_ = -1;

    thumb_shader_ = vivid::thumbnail::create_shader(ctx->device, kThumbShader, "WT Thumb Shader");
    thumb_uniform_buf_ = vivid::thumbnail::create_uniform_buffer(ctx->device, 16, "WT Thumb Uniforms");
    thumb_sampler_ = vivid::gpu::create_linear_sampler(ctx->device, "WT Thumb Sampler");
    thumb_bind_layout_ = vivid::gpu::create_standard_bind_layout(
        ctx->device, 1, "WT Thumb BGL", 16, WGPUShaderStage_Vertex);
    thumb_pipe_layout_ = vivid::thumbnail::create_pipeline_layout(
        ctx->device, thumb_bind_layout_, "WT Thumb PipeLayout");
    thumb_pipeline_ = vivid::thumbnail::create_pipeline(
        ctx->device, thumb_shader_, thumb_pipe_layout_,
        ctx->thumbnail_format, "WT Thumb Pipeline");
    thumb_pipeline_format_ = ctx->thumbnail_format;

    int family = (ctx->param_count > 1) ? static_cast<int>(ctx->param_values[1]) : 0;
    int member = (ctx->param_count > 2) ? static_cast<int>(ctx->param_values[2]) : 0;
    upload_wavetable_texture(ctx->device, ctx->queue, family, member);

    if (thumb_wt_view_) {
        thumb_bind_group_ = vivid::gpu::create_standard_bind_group(
            ctx->device, thumb_bind_layout_, thumb_uniform_buf_, 16,
            thumb_sampler_, &thumb_wt_view_, 1, "WT Thumb BG");
    }
}

void WavetableOsc::draw_thumbnail(const VividThumbnailContext* ctx) {
    if (!ctx) return;
    int source = (ctx->param_count > 0) ? static_cast<int>(ctx->param_values[0]) : 0;
    if (source != SOURCE_BUILTIN) {
        return;
    }

    if (!thumb_pipeline_ || thumb_pipeline_format_ != ctx->thumbnail_format) {
        rebuild_thumb_pipeline(ctx);
    }

    int family = (ctx->param_count > 1) ? static_cast<int>(ctx->param_values[1]) : 0;
    int member = (ctx->param_count > 2) ? static_cast<int>(ctx->param_values[2]) : 0;
    if (family != thumb_wt_family_ || member != thumb_wt_member_) {
        upload_wavetable_texture(ctx->device, ctx->queue, family, member);
        if (thumb_wt_view_ && thumb_bind_layout_ && thumb_uniform_buf_ && thumb_sampler_) {
            vivid::gpu::release(thumb_bind_group_);
            thumb_bind_group_ = vivid::gpu::create_standard_bind_group(
                ctx->device, thumb_bind_layout_, thumb_uniform_buf_, 16,
                thumb_sampler_, &thumb_wt_view_, 1, "WT Thumb BG");
        }
    }

    if (!thumb_pipeline_ || !thumb_bind_group_ || !thumb_uniform_buf_) {
        vivid_report_thumbnail_error(ctx, "wavetable thumbnail pipeline init failed");
        return;
    }

    struct {
        float position;
        float warp_amount;
        float frame_count;
        float source_mode;
    } uniforms{};
    uniforms.position = (ctx->param_count > 4) ? ctx->param_values[4] : 0.0f;
    uniforms.warp_amount = (ctx->param_count > 7) ? ctx->param_values[7] : 0.0f;
    uniforms.frame_count = static_cast<float>(thumb_wt_frames_);
    uniforms.source_mode = static_cast<float>(source);
    wgpuQueueWriteBuffer(ctx->queue, thumb_uniform_buf_, 0, &uniforms, sizeof(uniforms));

    vivid::thumbnail::run_pass(ctx, thumb_pipeline_, thumb_bind_group_, "WT Thumb Pass");
}
