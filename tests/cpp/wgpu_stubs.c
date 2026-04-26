// WGPU symbol stubs for dlopening operator dylibs in test processes.
//
// WavetableOsc and other GPU-aware operators import WebGPU functions for
// thumbnail rendering. The vivid runtime app links wgpu_native statically so
// these symbols resolve via the global flat namespace. Test processes that
// don't link WebGPU need stubs to satisfy the dynamic loader. The stubs
// return null/zero — they're never called on the audio path that the tests
// exercise (audio render lives in wavetable_osc_process.cpp, which has no
// WGPU calls).

#include <stddef.h>

void* wgpuBindGroupLayoutRelease(void* p)         { (void)p; return 0; }
void* wgpuBindGroupRelease(void* p)               { (void)p; return 0; }
void* wgpuBufferRelease(void* p)                  { (void)p; return 0; }
void* wgpuCommandEncoderBeginRenderPass(void* a, void* b)         { (void)a; (void)b; return 0; }
void* wgpuDeviceCreateBindGroup(void* a, void* b)                 { (void)a; (void)b; return 0; }
void* wgpuDeviceCreateBindGroupLayout(void* a, void* b)           { (void)a; (void)b; return 0; }
void* wgpuDeviceCreateBuffer(void* a, void* b)                    { (void)a; (void)b; return 0; }
void* wgpuDeviceCreatePipelineLayout(void* a, void* b)            { (void)a; (void)b; return 0; }
void* wgpuDeviceCreateRenderPipeline(void* a, void* b)            { (void)a; (void)b; return 0; }
void* wgpuDeviceCreateSampler(void* a, void* b)                   { (void)a; (void)b; return 0; }
void* wgpuDeviceCreateShaderModule(void* a, void* b)              { (void)a; (void)b; return 0; }
void* wgpuDeviceCreateTexture(void* a, void* b)                   { (void)a; (void)b; return 0; }
void* wgpuPipelineLayoutRelease(void* p)          { (void)p; return 0; }
void  wgpuQueueWriteBuffer(void* a, void* b, size_t c, const void* d, size_t e) { (void)a; (void)b; (void)c; (void)d; (void)e; }
void  wgpuQueueWriteTexture(void* a, const void* b, const void* c, size_t d, const void* e) { (void)a; (void)b; (void)c; (void)d; (void)e; }
void  wgpuRenderPassEncoderDraw(void* a, unsigned b, unsigned c, unsigned d, unsigned e) { (void)a; (void)b; (void)c; (void)d; (void)e; }
void  wgpuRenderPassEncoderEnd(void* a)           { (void)a; }
void* wgpuRenderPassEncoderRelease(void* p)       { (void)p; return 0; }
void  wgpuRenderPassEncoderSetBindGroup(void* a, unsigned b, void* c, size_t d, const unsigned* e) { (void)a; (void)b; (void)c; (void)d; (void)e; }
void  wgpuRenderPassEncoderSetPipeline(void* a, void* b) { (void)a; (void)b; }
void* wgpuRenderPipelineRelease(void* p)          { (void)p; return 0; }
void* wgpuSamplerRelease(void* p)                 { (void)p; return 0; }
void* wgpuShaderModuleRelease(void* p)            { (void)p; return 0; }
void* wgpuTextureCreateView(void* a, const void* b)               { (void)a; (void)b; return 0; }
void* wgpuTextureRelease(void* p)                 { (void)p; return 0; }
void* wgpuTextureViewRelease(void* p)             { (void)p; return 0; }
