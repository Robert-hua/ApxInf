#pragma once
#include <cuda_runtime.h>
#include <cmath>
// Explicit opt-in row reductions. Independent symbols and dispatch preserve
// legacy norm/softmax behavior; only Kart currently selects this provider.
__global__ void apx_block_layer_norm(int rows,int width,int channels,int spatial,const float*a,const float*w,const float*b,float*y,float eps) {
 int row=blockIdx.x; __shared__ float sums[256];
 float v=0;for(int i=threadIdx.x;i<width;i+=256)v+=a[row*width+i];
 sums[threadIdx.x]=v;__syncthreads();for(int s=128;s;s>>=1){if(threadIdx.x<s)sums[threadIdx.x]+=sums[threadIdx.x+s];__syncthreads();}
 // Every warp must consume the broadcast before warp 0 reuses sums[0].
 float mean=sums[0]/width;__syncthreads();v=0;for(int i=threadIdx.x;i<width;i+=256){float d=a[row*width+i]-mean;v+=d*d;}
 sums[threadIdx.x]=v;__syncthreads();for(int s=128;s;s>>=1){if(threadIdx.x<s)sums[threadIdx.x]+=sums[threadIdx.x+s];__syncthreads();}
 float inv=rsqrtf(sums[0]/width+eps);
 for(int i=threadIdx.x;i<width;i+=256){int idx=row*width+i;int ch=(idx/spatial)%channels;y[idx]=(a[idx]-mean)*inv*w[ch]+b[ch];}
}
__global__ void apx_block_softmax(int rows,int width,const float*a,float*y,float scale) {
 int row=blockIdx.x;__shared__ float s[256];float v=-INFINITY;
 for(int i=threadIdx.x;i<width;i+=256)v=fmaxf(v,a[row*width+i]*scale);
 s[threadIdx.x]=v;__syncthreads();for(int d=128;d;d>>=1){if(threadIdx.x<d)s[threadIdx.x]=fmaxf(s[threadIdx.x],s[threadIdx.x+d]);__syncthreads();}
 // The maximum and denominator reductions share this array.
 float mx=s[0];__syncthreads();v=0;for(int i=threadIdx.x;i<width;i+=256)v+=expf(a[row*width+i]*scale-mx);
 s[threadIdx.x]=v;__syncthreads();for(int d=128;d;d>>=1){if(threadIdx.x<d)s[threadIdx.x]+=s[threadIdx.x+d];__syncthreads();}
 for(int i=threadIdx.x;i<width;i+=256)y[row*width+i]=expf(a[row*width+i]*scale-mx)/s[0];
}
