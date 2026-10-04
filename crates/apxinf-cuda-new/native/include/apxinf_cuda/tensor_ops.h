#pragma once
#include <stdint.h>
// Contiguous F32 storage; operations are prepared before graph capture.
// Conv kind 1/2 p[14]: 0=context precision, 1=BF16 (requires BF16 context).
// Conv kind 1 p[15]: 1=explicit FP32 im2col Conv1d, 0=existing provider selection.
// Kinds 13/14: opt-in block LayerNorm/Softmax; legacy kinds 4/5 are unchanged.
// Kinds 15/16/17: isolated FP16 Linear/Conv1d/ConvTranspose1d (F32 outputs).
// Kind 18: isolated dynamically scaled E4M3 Linear (F32 accumulation/output).
// Kinds 19/20/21: FP16 high/low three-product Linear/Conv1d/ConvTranspose1d.
// Kind 22: strict F32 GEMM ConvTranspose1d. Immutable weights, prepared scratch.
// Kind 23: packed F32 QKV attention [B,S,3*H*64], scale 1/8, no mask.
// Kind 24: warp LayerNorm (width <=1024); kind 25: scaled residual.
// Kind 26: LayerNorm + kind 19; c=[linear bias N, norm weight K, norm bias K].
// Kind 27: exact-erf GELU(a) + kind 19; p=[M,N,K], optional c=linear bias N.
// Zero-initialized existing callers retain their original behavior.
struct apx_tensor_spec { int32_t kind; int32_t p[24]; float f[4]; };
extern "C" {
const char* apx_tensor_error();
int apx_tensor_context_create(int device, void** result);
int apx_tensor_math_mode(void* context, int tf32);
void apx_tensor_context_destroy(void* context);
int apx_tensor_sync(void* context);
int apx_tensor_prepare(void* context, const apx_tensor_spec* spec, const float* a,
                       const float* b, const float* c, float* y, void** result);
int apx_tensor_enqueue(void* execution);
int apx_tensor_tune(void* execution);
int apx_tensor_rgb_normalize(void* context,int pixels,const unsigned char* x,float* y,const float* mean,const float* std);
int apx_tensor_rgb_batch(void* context,int batch,int height,int width,int padded,const unsigned char* x,float* y,const float* mean,const float* std);
int apx_tensor_rgb_batch_device(void* context,int batch,int height,int width,int padded,const uintptr_t* frames,unsigned char* staging,float* y,const float* mean,const float* std);
void apx_tensor_destroy(void* execution);
int apx_tensor_capture_begin(void* context);
int apx_tensor_capture_end(void* context, void** graph);
int apx_tensor_graph_replay(void* context, void* graph);
void apx_tensor_graph_destroy(void* graph);
}
