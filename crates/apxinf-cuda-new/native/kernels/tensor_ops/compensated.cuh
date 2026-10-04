#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
namespace apx_compensated {
__device__ inline void parts(float v,__half&hi,__half&lo){hi=__float2half_rn(v);lo=__float2half_rn((v-__half2float(hi))*1024.f);}
__global__ void split(int n,const float*x,__half*h,__half*l){int i=blockIdx.x*256+threadIdx.x;if(i<n)parts(x[i],h[i],l[i]);}
// Preserve the standalone exact-erf GELU's FP32 operation order. In particular,
// do not contract 1+erf or replace this with the cuBLAS tanh epilogue.
__global__ void gelu_split(int n,const float*x,__half*h,__half*l){
 int i=blockIdx.x*256+threadIdx.x;
 if(i<n){float v=x[i];float g=__fmul_rn(__fmul_rn(v,0.5f),__fadd_rn(1.f,erff(__fmul_rn(v,0.7071067811865475244f))));parts(g,h[i],l[i]);}
}
__global__ void pack(int n,int co,int kernel,const float*w,__half*h,__half*l,float*f){
 int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;int ci=n/(co*kernel),ic=i%ci,oc=(i/ci)/kernel,k=(i/ci)%kernel;
 float v=w[(ic*co+oc)*kernel+k];if(f)f[i]=v;else parts(v,h[i],l[i]);
}
__global__ void lower(int n,int length,int out,int kernel,int stride,int pad,int channels,const float*x,__half*h,__half*l){
 int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;int t=i/(channels*kernel),k=i%kernel,c=(i/kernel)%channels,src=t*stride-pad+k;
 parts(src>=0&&src<length?x[c*length+src]:0.f,h[i],l[i]);
}
__global__ void bias(int n,int channels,int spatial,const float*b,float*y){int i=blockIdx.x*256+threadIdx.x;if(i<n)y[i]+=b[(i/spatial)%channels];}
__global__ void gather(int n,int length,int columns,int out,int kernel,int stride,int pad,int co,bool rowmajor,const float*col,const float*b,float*y){
 int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;int c=i/out,t=i%out;float v=0.f;
 for(int k=0;k<kernel;k++){int q=t+pad-k;if(q>=0&&q%stride==0&&q/stride<length)v+=col[rowmajor?(q/stride)*(co*kernel)+c*kernel+k:(c*kernel+k)*columns+q/stride];}
 y[i]=b?v+b[c]:v;
}
__global__ void unpad(int count,int out,int columns,int channels,const float*col,const float*b,float*y){
 int i=blockIdx.x*256+threadIdx.x;if(i<count){int c=i/out;y[i]=col[(i%out)*channels+c]+(b?b[c]:0.f);}
}
__device__ inline float norm_sum(float v){for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);return __shfl_sync(0xffffffff,v,0);}
template<int Chunks> __global__ void norm_split(int rows,int width,const float*x,const float*w,const float*b,__half*h,__half*l,float eps){
 int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(row>=rows)return;
 float v[Chunks],sum=0;
 #pragma unroll
 for(int j=0;j<Chunks;j++){int c=j*32+lane;v[j]=c<width?x[row*width+c]:0.f;sum+=v[j];}
 float mean=norm_sum(sum)/width,var=0;
 #pragma unroll
 for(int j=0;j<Chunks;j++){int c=j*32+lane;if(c<width){float d=v[j]-mean;var+=d*d;}}
 float inv=rsqrtf(norm_sum(var)/width+eps);
 #pragma unroll
 for(int j=0;j<Chunks;j++){int c=j*32+lane;if(c<width)parts((v[j]-mean)*inv*w[c]+b[c],h[row*width+c],l[row*width+c]);}
}

__global__ void rgb_padded(int count,int height,int width,int padded,const unsigned char*x,float*y,float m0,float m1,float m2,float s0,float s1,float s2){
 int i=blockIdx.x*256+threadIdx.x;if(i>=count)return;
 int col=i%padded,row=(i/padded)%height,c=(i/(padded*height))%3,batch=i/(padded*height*3);
 int src=max(0,min(width-1,col-(padded-width)/2));
 float value=__fdiv_rn(float(x[((batch*height+row)*width+src)*3+c]),255.f);
 float mean=c==0?m0:c==1?m1:m2,std=c==0?s0:c==1?s1:s2;
 y[i]=__fdiv_rn(__fsub_rn(value,mean),std);
}

}
