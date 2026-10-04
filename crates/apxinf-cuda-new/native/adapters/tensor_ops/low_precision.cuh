#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <stdexcept>
#include <string>
// Independent opt-in provider. Existing context math modes and kernels stay intact.
namespace apx_lowp {
inline void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
inline void check(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("FP16 cuBLAS status "+std::to_string(s));}
__global__ void cast_half(int n,const float*x,__half*y){int i=blockIdx.x*256+threadIdx.x;if(i<n)y[i]=__float2half_rn(x[i]);}
__global__ void pack_transpose(int n,int co,int kernel,const float*w,__half*y){
 int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;int ci=n/(co*kernel),ic=i%ci,oc=(i/ci)/kernel,k=(i/ci)%kernel;
 y[i]=__float2half_rn(w[(ic*co+oc)*kernel+k]);
}
__global__ void lower_half(int n,int length,int out,int kernel,int stride,int pad,const float*x,__half*y){
 int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;int t=i%out,k=(i/out)%kernel,c=i/(out*kernel),src=t*stride-pad+k;
 y[i]=__float2half_rn(src>=0&&src<length?x[c*length+src]:0.f);
}
__global__ void bias_linear(int n,int channels,const float*b,float*y){int i=blockIdx.x*256+threadIdx.x;if(i<n)y[i]+=b[i%channels];}
__global__ void bias_conv(int n,int length,const float*b,float*y){int i=blockIdx.x*256+threadIdx.x;if(i<n)y[i]+=b[i/length];}
__global__ void gather_transpose(int n,int length,int out,int kernel,int stride,int pad,const float*col,const float*b,float*y){
 int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;int c=i/out,t=i%out;float v=0.f;
 for(int k=0;k<kernel;k++){int q=t+pad-k;if(q>=0&&q%stride==0&&q/stride<length)v+=col[(c*kernel+k)*length+q/stride];}
 y[i]=b?v+b[c]:v;
}
struct Execution {
 cudaStream_t stream; cublasHandle_t blas{};__half*xh{},*wh{};float*col{};
 int kind,m=0,n=0,k=0,batch=0,ci=0,length=0,co=0,kernel=0,out=0,stride=0,pad=0;
 const float*x;const float*w;const float*b;float*y;
 Execution(cudaStream_t st,int op,const int*p,const float*a,const float*weight,const float*bias,float*output)
 :stream(st),kind(op),x(a),w(weight),b(bias),y(output){
  if(kind==15){m=p[0];n=p[1];k=p[2];}
  else{batch=p[0];ci=p[1];length=p[2];co=p[3];kernel=p[4];out=p[5];stride=p[6];pad=p[7];}
 }
 ~Execution(){if(xh)cudaFree(xh);if(wh)cudaFree(wh);if(col)cudaFree(col);if(blas)cublasDestroy(blas);}
 void prepare(){
  check(cublasCreate(&blas));check(cublasSetStream(blas,stream));check(cublasSetMathMode(blas,CUBLAS_DEFAULT_MATH));
  size_t nx=kind==15?size_t(m)*k:kind==16?size_t(ci)*kernel*out:size_t(ci)*length;
  size_t nw=kind==15?size_t(n)*k:size_t(ci)*co*kernel;
  size_t nc=kind==17?size_t(co)*kernel*length:0;
  if(nx>INT32_MAX||nw>INT32_MAX||nc>INT32_MAX||nx*2+nc*4>64*1024*1024)throw std::runtime_error("FP16 scratch/shape exceeds bound");
  check(cudaMalloc(&xh,nx*sizeof(__half)));check(cudaMalloc(&wh,nw*sizeof(__half)));
  if(nc)check(cudaMalloc(&col,nc*sizeof(float)));
  if(kind==17)pack_transpose<<<(nw+255)/256,256,0,stream>>>(int(nw),co,kernel,w,wh);
  else cast_half<<<(nw+255)/256,256,0,stream>>>(int(nw),w,wh);
  check(cudaPeekAtLastError());
 }
 void run(){
  float one=1.f,zero=0.f;
  if(kind==15){
   cast_half<<<(m*k+255)/256,256,0,stream>>>(m*k,x,xh);
   check(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&one,wh,CUDA_R_16F,k,xh,CUDA_R_16F,k,&zero,y,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
   if(b)bias_linear<<<(m*n+255)/256,256,0,stream>>>(m*n,n,b,y);
  }else for(int ib=0;ib<batch;ib++){
   const float*input=x+ib*ci*length;float*output=y+ib*co*out;
   if(kind==16){
    int count=ci*kernel*out;lower_half<<<(count+255)/256,256,0,stream>>>(count,length,out,kernel,stride,pad,input,xh);
    check(cublasGemmEx(blas,CUBLAS_OP_N,CUBLAS_OP_N,out,co,ci*kernel,&one,xh,CUDA_R_16F,out,wh,CUDA_R_16F,ci*kernel,&zero,output,CUDA_R_32F,out,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    if(b)bias_conv<<<(co*out+255)/256,256,0,stream>>>(co*out,out,b,output);
   }else{
    cast_half<<<(ci*length+255)/256,256,0,stream>>>(ci*length,input,xh);
    check(cublasGemmEx(blas,CUBLAS_OP_N,CUBLAS_OP_N,length,co*kernel,ci,&one,xh,CUDA_R_16F,length,wh,CUDA_R_16F,ci,&zero,col,CUDA_R_32F,length,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    gather_transpose<<<(co*out+255)/256,256,0,stream>>>(co*out,length,out,kernel,stride,pad,col,b,output);
   }
  }
  check(cudaPeekAtLastError());
 }
};
}

namespace apx_lowp {
__global__ void absmax(int n,const float*x,float*out){
 __shared__ float v[256];int i=blockIdx.x*256+threadIdx.x;v[threadIdx.x]=i<n?fabsf(x[i]):0.f;__syncthreads();
 for(int d=128;d;d>>=1){if(threadIdx.x<d)v[threadIdx.x]=fmaxf(v[threadIdx.x],v[threadIdx.x+d]);__syncthreads();}
 if(threadIdx.x==0)atomicMax(reinterpret_cast<int*>(out),__float_as_int(v[0]));
}
__global__ void to_scale(float*x){*x=fmaxf(*x/448.f,1e-12f);}
__global__ void pack_fp8(int n,int rows,int cols,int padded_cols,const float*x,const float*scale,__nv_fp8_e4m3*y){
 int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;int row=i/padded_cols,col=i%padded_cols;
 y[i]=__nv_fp8_e4m3(row<rows&&col<cols?x[row*cols+col]/(*scale):0.f);
}
__global__ void unpack_fp8(int count,int n,int npad,const float*x,const float*b,float*y){
 int i=blockIdx.x*256+threadIdx.x;if(i<count){float v=x[(i/n)*npad+i%n];y[i]=b?v+b[i%n]:v;}
}
struct Fp8Linear {
 cudaStream_t stream;cublasLtHandle_t handle{};cublasLtMatmulDesc_t desc{};
 cublasLtMatrixLayout_t wa{},xa{},ya{};cublasLtMatmulAlgo_t algo{};
 __nv_fp8_e4m3*xq{},*wq{};float*xs{},*ws{},*yp{};void*scratch{};size_t bytes{};
 int m,n,k,mp,np,kp;const float*x;const float*w;const float*b;float*y;
 Fp8Linear(cudaStream_t st,const int*p,const float*a,const float*weight,const float*bias,float*out)
 :stream(st),m(p[0]),n(p[1]),k(p[2]),mp((m+15)/16*16),np((n+15)/16*16),kp((k+15)/16*16),x(a),w(weight),b(bias),y(out){}
 ~Fp8Linear(){if(xq)cudaFree(xq);if(wq)cudaFree(wq);if(xs)cudaFree(xs);if(ws)cudaFree(ws);if(yp)cudaFree(yp);if(scratch)cudaFree(scratch);if(wa)cublasLtMatrixLayoutDestroy(wa);if(xa)cublasLtMatrixLayoutDestroy(xa);if(ya)cublasLtMatrixLayoutDestroy(ya);if(desc)cublasLtMatmulDescDestroy(desc);if(handle)cublasLtDestroy(handle);}
 void prepare(){
  int device=0;cudaDeviceProp prop{};check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&prop,device));
  if(prop.major!=11||prop.minor!=0)throw std::runtime_error("FP8 Linear candidate qualified on Thor sm110 only");
  if(size_t(mp)*kp+size_t(mp)*np*4>64*1024*1024||size_t(np)*kp>INT32_MAX)throw std::runtime_error("FP8 scratch/weights exceed bound");
  check(cudaMalloc(&xq,size_t(mp)*kp));check(cudaMalloc(&wq,size_t(np)*kp));check(cudaMalloc(&xs,4));check(cudaMalloc(&ws,4));check(cudaMalloc(&yp,size_t(mp)*np*4));
  check(cudaMemsetAsync(ws,0,4,stream));absmax<<<(n*k+255)/256,256,0,stream>>>(n*k,w,ws);to_scale<<<1,1,0,stream>>>(ws);
  pack_fp8<<<(np*kp+255)/256,256,0,stream>>>(np*kp,n,k,kp,w,ws,wq);
  check(cublasLtCreate(&handle));check(cublasLtMatmulDescCreate(&desc,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t trans=CUBLAS_OP_T;check(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
  check(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&ws,sizeof(ws)));
  check(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&xs,sizeof(xs)));
  check(cublasLtMatrixLayoutCreate(&wa,CUDA_R_8F_E4M3,kp,np,kp));check(cublasLtMatrixLayoutCreate(&xa,CUDA_R_8F_E4M3,kp,mp,kp));check(cublasLtMatrixLayoutCreate(&ya,CUDA_R_32F,np,mp,np));
  cublasLtMatmulPreference_t pref{};check(cublasLtMatmulPreferenceCreate(&pref));size_t limit=4*1024*1024;
  auto status=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&limit,sizeof(limit));
  if(status!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);check(status);}
  cublasLtMatmulHeuristicResult_t result{};int count=0;
  status=cublasLtMatmulAlgoGetHeuristic(handle,desc,wa,xa,ya,ya,pref,1,&result,&count);cublasLtMatmulPreferenceDestroy(pref);check(status);
  if(count!=1||result.state!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("no FP8 Linear algorithm");
  algo=result.algo;bytes=result.workspaceSize;if(bytes>limit)throw std::runtime_error("FP8 workspace exceeds bound");if(bytes)check(cudaMalloc(&scratch,bytes));check(cudaPeekAtLastError());
 }
 void run(){
  check(cudaMemsetAsync(xs,0,4,stream));absmax<<<(m*k+255)/256,256,0,stream>>>(m*k,x,xs);to_scale<<<1,1,0,stream>>>(xs);
  pack_fp8<<<(mp*kp+255)/256,256,0,stream>>>(mp*kp,m,k,kp,x,xs,xq);
  float one=1.f,zero=0.f;check(cublasLtMatmul(handle,desc,&one,wq,wa,xq,xa,&zero,yp,ya,yp,ya,&algo,scratch,bytes,stream));
  unpack_fp8<<<(m*n+255)/256,256,0,stream>>>(m*n,n,np,yp,b,y);check(cudaPeekAtLastError());
 }
};
}
