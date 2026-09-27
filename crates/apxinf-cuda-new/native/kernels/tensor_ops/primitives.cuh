#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
// NCHW Conv1d lowering. Reused prepared scratch is [Cin * kernel, Lout].
__global__ void apx_im2col_1d(int n,int length,int output_length,int kernel,int stride,int pad,const float*x,float*col){
 int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;
 int position=i%output_length,k=(i/output_length)%kernel,c=i/(output_length*kernel);
 int source=position*stride-pad+k;
 col[i]=source>=0&&source<length?x[c*length+source]:0.f;
}
__global__ void apx_to_bf16(int n,const float*x,__nv_bfloat16*y){int i=blockIdx.x*256+threadIdx.x;if(i<n)y[i]=__float2bfloat16(x[i]);}
__global__ void apx_from_bf16(int n,const __nv_bfloat16*x,float*y){int i=blockIdx.x*256+threadIdx.x;if(i<n)y[i]=__bfloat162float(x[i]);}
__global__ void apx_round_bf16(int n,float*y,const float*b,int inner,int channels){int i=blockIdx.x*256+threadIdx.x;if(i<n){float v=y[i];if(b)v+=__bfloat162float(__float2bfloat16(b[(i/inner)%channels]));y[i]=__bfloat162float(__float2bfloat16(v));}}
// Fuse NCHW<->NHWC layout conversion with the explicitly selected BF16 cast.
// The filter conversion runs only during preparation, never once per diffusion step.
__global__ void apx_pack_nhwc(int n,int channels,int spatial,const float*x,__nv_bfloat16*y){int i=blockIdx.x*256+threadIdx.x;if(i<n){int c=i%channels;int v=i/channels;y[i]=__float2bfloat16(x[(v/spatial*channels+c)*spatial+v%spatial]);}}
__global__ void apx_unpack_nhwc(int n,int channels,int spatial,const __nv_bfloat16*x,float*y){int i=blockIdx.x*256+threadIdx.x;if(i<n){int c=(i/spatial)%channels;int v=i%spatial;y[i]=__bfloat162float(x[(i/(channels*spatial)*spatial+v)*channels+c]);}}
// p[] is lowered and bounds-checked by L2. All loops operate on device storage.
__global__ void apx_elementwise(int n,const float*a,const float*b,const float*c,float*y,int mode,int inner,int channels,float alpha,float beta) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float x=a[i]; int j=(i/inner)%channels;
 switch(mode){
 case 0:y[i]=x+b[i];break;
 case 1:y[i]=fmaxf(x,0.f);break;
 case 2:y[i]=x*tanhf(x>20.f?x:log1pf(expf(x)));break;
 case 3:y[i]=x/(1.f+expf(-x));break;
 case 4:y[i]=__fadd_rn(__fmul_rn(x,b[j]),c[j]);break;
 case 5:y[i]=x*alpha+beta;break;
 case 6:y[i]=x+b[j];break;
 case 7:y[i]=x;break;
 case 8:y[i]=__bfloat162float(__float2bfloat16(x));break;
 }
}
__global__ void apx_normalize(int rows,int width,int channels,int spatial,const float*a,const float*w,const float*b,float*y,float eps) {
 int row=blockIdx.x; __shared__ float sums[256];
 float v=0;for(int i=threadIdx.x;i<width;i+=256)v+=a[row*width+i];
 sums[threadIdx.x]=v;__syncthreads();for(int s=128;s;s>>=1){if(threadIdx.x<s)sums[threadIdx.x]+=sums[threadIdx.x+s];__syncthreads();}
 float mean=sums[0]/width;v=0;for(int i=threadIdx.x;i<width;i+=256){float d=a[row*width+i]-mean;v+=d*d;}
 sums[threadIdx.x]=v;__syncthreads();for(int s=128;s;s>>=1){if(threadIdx.x<s)sums[threadIdx.x]+=sums[threadIdx.x+s];__syncthreads();}
 float inv=rsqrtf(sums[0]/width+eps);
 for(int i=threadIdx.x;i<width;i+=256){int idx=row*width+i;int ch=(idx/spatial)%channels;y[idx]=(a[idx]-mean)*inv*w[ch]+b[ch];}
}
// Group statistics use Welford merges in warp/block order. Unlike LayerNorm,
// channel affine parameters are first folded using the computed row statistics.
struct ApxMoments { float mean,m2,count; };
__device__ ApxMoments apx_merge(ApxMoments a,ApxMoments b){
 if(a.count==0)return b;if(b.count==0)return a;
 float delta=b.mean-a.mean,total=a.count+b.count,ratio=b.count/total;
 return {a.mean+delta*ratio,a.m2+b.m2+delta*delta*a.count*ratio,total};
}
__device__ ApxMoments apx_warp_moments(ApxMoments v){
 for(int offset=16;offset;offset>>=1){
  ApxMoments other={__shfl_down_sync(0xffffffff,v.mean,offset),__shfl_down_sync(0xffffffff,v.m2,offset),__shfl_down_sync(0xffffffff,v.count,offset)};
  v=apx_merge(v,other);
 }return v;
}
__global__ void apx_group_normalize(int width,int channels,int spatial,const float*x,const float*w,const float*b,float*y,float eps){
 int row=blockIdx.x,tid=threadIdx.x;ApxMoments v={0,0,0};
 for(int i=tid;i<width;i+=blockDim.x){
  float value=x[row*width+i],delta=value-v.mean;v.count+=1;
  v.mean+=delta/v.count;v.m2+=delta*(value-v.mean);
 }
 v=apx_warp_moments(v);__shared__ ApxMoments warps[32];__shared__ float mean,inv;
 if((tid%32)==0)warps[tid/32]=v;__syncthreads();
 if(tid<32){
  v=tid<int(blockDim.x/32)?warps[tid]:ApxMoments{0,0,0};v=apx_warp_moments(v);
  if(tid==0){mean=v.mean;inv=rsqrtf(v.m2/v.count+eps);}
 }__syncthreads();
 for(int i=tid;i<width;i+=blockDim.x){int index=row*width+i,c=(index/spatial)%channels;
  float scale=inv*w[c],bias=-scale*mean+b[c];y[index]=scale*x[index]+bias;
 }
}
__global__ void apx_softmax(int rows,int width,const float*a,float*y,float scale) {
 int row=blockIdx.x;__shared__ float s[256];float v=-INFINITY;
 for(int i=threadIdx.x;i<width;i+=256)v=fmaxf(v,a[row*width+i]*scale);
 s[threadIdx.x]=v;__syncthreads();for(int d=128;d;d>>=1){if(threadIdx.x<d)s[threadIdx.x]=fmaxf(s[threadIdx.x],s[threadIdx.x+d]);__syncthreads();}
 float mx=s[0];v=0;for(int i=threadIdx.x;i<width;i+=256)v+=expf(a[row*width+i]*scale-mx);
 s[threadIdx.x]=v;__syncthreads();for(int d=128;d;d>>=1){if(threadIdx.x<d)s[threadIdx.x]+=s[threadIdx.x+d];__syncthreads();}
 for(int i=threadIdx.x;i<width;i+=256)y[row*width+i]=expf(a[row*width+i]*scale-mx)/s[0];
}
__global__ void apx_permute(int n,const float*a,float*y,int d0,int d1,int d2,int d3,int s0,int s1,int s2,int s3) {
 int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;int r=i;
 int c3=r%d3;r/=d3;int c2=r%d2;r/=d2;int c1=r%d1;int c0=r/d1;
 y[i]=a[c0*s0+c1*s1+c2*s2+c3*s3];
}
__global__ void apx_concat(int n,const float*a,const float*b,float*y,int left,int right,int inner) {
 int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;int block=i/((left+right)*inner);int j=i%((left+right)*inner);
 y[i]=j<left*inner?a[block*left*inner+j]:b[block*right*inner+j-left*inner];
}
__global__ void apx_slice(int n,const float*a,float*y,int original,int length,int start,int inner) {
 int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;y[i]=a[(i/(length*inner)*original+start)*inner+i%(length*inner)];
}
__global__ void apx_pool(int n,const float*a,float*y,int channels,int h,int w,int oh,int ow,int kernel,int stride,int pad) {
 int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;int x=i%ow,z=(i/ow)%oh,c=i/(ow*oh);float v=-INFINITY;
 for(int ky=0;ky<kernel;ky++)for(int kx=0;kx<kernel;kx++){int sy=z*stride-pad+ky,sx=x*stride-pad+kx;if(sy>=0&&sy<h&&sx>=0&&sx<w)v=fmaxf(v,a[(c*h+sy)*w+sx]);}y[i]=v;
}
// Generic linear combination with optional clamp. Used by a family-owned diffusion schedule.
__global__ void apx_axpby(int n,const float*a,const float*b,const float*c,float*y,float aa,float bb,float cc,float clip) {
 int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;float x=__fadd_rn(__fmul_rn(aa,a[i]),__fmul_rn(bb,b[i]));if(clip>0)x=fminf(clip,fmaxf(-clip,x));y[i]=__fadd_rn(x,c?__fmul_rn(cc,c[i]):0.f);
}

// Eval BatchNorm keeps the subtraction before scaling, matching the declared
// mixed precision semantics. Folding mean/bias on the host changes rounding.
__global__ void apx_batch_norm(int n,int channels,int spatial,const float*x,const float*p,float*y,float eps,int bf16,int frozen){
 int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;int c=(i/spatial)%channels;
 if(frozen){
  float scale=__fmul_rn(p[c],rsqrtf(__fadd_rn(p[3*channels+c],eps)));
  float bias=__fsub_rn(p[channels+c],__fmul_rn(p[2*channels+c],scale));
  y[i]=__fadd_rn(__fmul_rn(x[i],scale),bias);return;
 }
 // Match the separately computed FP32 inference invstd (Torch native CUDA).
 float inv=rsqrtf(__fadd_rn(p[3*channels+c],eps));
 float v=p[c]*(x[i]-p[2*channels+c])*inv+p[channels+c];
 y[i]=bf16?__bfloat162float(__float2bfloat16(v)):v;
}


__global__ void apx_rgb_normalize(int pixels,const unsigned char*x,float*y,float m0,float m1,float m2,float s0,float s1,float s2){
 int i=blockIdx.x*256+threadIdx.x;if(i>=pixels*3)return;int c=i/pixels;int p=i%pixels;
 float mean=c==0?m0:c==1?m1:m2;float std=c==0?s0:c==1?s1:s2;
 float value=__fmul_rn(float(x[p*3+c]),1.f/255.f);
 y[i]=__fdiv_rn(__fsub_rn(value,mean),std);
}

// Conv1d lowering with padded position rows, fused with the input BF16 cast.
__global__ void apx_im2row_bf16(int n,int length,int output_length,int kernel,int stride,int pad,int kdim,const float*x,__nv_bfloat16*col){
 int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;
 int position=i/kdim,k=i%kernel,c=(i%kdim)/kernel;
 int source=position*stride-pad+k;
 col[i]=__float2bfloat16(position<output_length&&source>=0&&source<length?x[c*length+source]:0.f);
}
// Preserve two BF16 rounding boundaries: convolution, then optional bias add.
__global__ void apx_unpack_conv1d_bf16(int n,int length,int padded,const __nv_bfloat16*x,const float*b,float*y){
 int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;int c=i/length;
 float v=__bfloat162float(x[c*padded+i%length]);
 y[i]=b?__bfloat162float(__float2bfloat16(v+__bfloat162float(__float2bfloat16(b[c])))):v;
}

// Per-tensor dynamic E4M3 quantization. Positive FP32 bits preserve max order.
__global__ void apx_bf16_absmax(int n,const float*x,float*maximum){
 int i=blockIdx.x*256+threadIdx.x;float v=0.f;
 for(;i<n;i+=gridDim.x*256)v=fmaxf(v,fabsf(__bfloat162float(__float2bfloat16(x[i]))));
 __shared__ float tmp[256];tmp[threadIdx.x]=v;__syncthreads();
 for(int d=128;d;d>>=1){if(threadIdx.x<d)tmp[threadIdx.x]=fmaxf(tmp[threadIdx.x],tmp[threadIdx.x+d]);__syncthreads();}
 if(threadIdx.x==0)atomicMax(reinterpret_cast<unsigned int*>(maximum),__float_as_uint(tmp[0]));
}
__global__ void apx_fp8_scale(float*scale){if(threadIdx.x==0)scale[0]=fmaxf(scale[0]/448.f,1e-12f);}
__global__ void apx_to_fp8(int n,const float*x,const float*scale,__nv_fp8_e4m3*y){
 int i=blockIdx.x*256+threadIdx.x;if(i<n)y[i]=__nv_fp8_e4m3(__bfloat162float(__float2bfloat16(x[i]))/scale[0]);
}
__global__ void apx_im2row_fp8(int n,int length,int output_length,int kernel,int stride,int pad,int kdim,const float*x,const float*scale,__nv_fp8_e4m3*col){
 int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;
 int position=i/kdim,k=i%kernel,c=(i%kdim)/kernel,source=position*stride-pad+k;
 float v=position<output_length&&source>=0&&source<length?__bfloat162float(__float2bfloat16(x[c*length+source])):0.f;
 col[i]=__nv_fp8_e4m3(v/scale[0]);
}
