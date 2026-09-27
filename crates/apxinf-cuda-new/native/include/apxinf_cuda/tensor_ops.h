#pragma once
#include <stdint.h>
// Version 1: contiguous F32 tensors; operations are prepared before graph capture.
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
void apx_tensor_destroy(void* execution);
int apx_tensor_capture_begin(void* context);
int apx_tensor_capture_end(void* context, void** graph);
int apx_tensor_graph_replay(void* context, void* graph);
void apx_tensor_graph_destroy(void* graph);
}
