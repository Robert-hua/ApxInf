#pragma once
#include <cuda_runtime.h>
// Independent opt-in warp row reductions and scaled residual.
__device__ inline float apx_warp_sum(float v){for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);return __shfl_sync(0xffffffff,v,0);}
__global__ void apx_warp_norm(int rows,int width,const float*x,const float*w,const float*b,float*y,float eps){
 int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(row>=rows)return;
 float v[32],sum=0;
 
 #pragma unroll
 for(int j=0;j<32;j++){int c=j*32+lane;v[j]=c<width?x[row*width+c]:0.f;sum+=v[j];}
 float mean=apx_warp_sum(sum)/width;float var=0;
 
 #pragma unroll
 for(int j=0;j<32;j++){int c=j*32+lane;if(c<width){float d=v[j]-mean;var+=d*d;}}
 float inv=rsqrtf(apx_warp_sum(var)/width+eps);
 
 #pragma unroll
 for(int j=0;j<32;j++){int c=j*32+lane;if(c<width)y[row*width+c]=(v[j]-mean)*inv*w[c]+b[c];}
}
__global__ void apx_scaled_residual(int count,int width,const float*x,const float*branch,const float*gamma,float*y){
 int i=blockIdx.x*256+threadIdx.x;if(i<count)y[i]=__fadd_rn(x[i],__fmul_rn(branch[i],gamma[i%width]));
}
