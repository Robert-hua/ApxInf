#include "vendor/kernel_forward.h"
using K=AttentionKernel<float,cutlass::arch::Sm80,true,64,64,64,false,false>;
__global__ void apx_fmha_f32_kernel(K::Params p){if(p.advance_to_block())K::attention_kernel(p);}
extern "C" int apx_fmha_f32_prepare(){return int(cudaFuncSetAttribute(apx_fmha_f32_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(K::SharedStorage)));}
extern "C" int apx_fmha_f32_run(int batch,int seq,int heads,const float*qkv,float*y,cudaStream_t stream){
 K::Params p;p.query_ptr=const_cast<float*>(qkv);p.key_ptr=p.query_ptr+heads*64;p.value_ptr=p.key_ptr+heads*64;p.output_ptr=y;
 p.scale=.125f;p.head_dim=64;p.head_dim_value=64;p.num_queries=seq;p.num_keys=seq;p.num_keys_absolute=seq;p.num_batches=batch;p.num_heads=heads;
 p.q_strideM=p.k_strideM=p.v_strideM=heads*64*3;p.o_strideM=heads*64;
 p.q_strideH=p.k_strideH=p.v_strideH=64;p.q_strideB=p.k_strideB=p.v_strideB=(int64_t)seq*heads*64*3;
 apx_fmha_f32_kernel<<<p.getBlocksGrid(),p.getThreadsGrid(),sizeof(K::SharedStorage),stream>>>(p);
 return int(cudaPeekAtLastError());
}
