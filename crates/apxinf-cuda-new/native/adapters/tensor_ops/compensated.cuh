#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <stdexcept>
#include <string>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include "../../kernels/tensor_ops/compensated.cuh"
// Independent prepared provider: no change to existing F32/FP16 dispatch.
// High/low decomposition follows the three-product idea in CUTLASS example 27,
// but uses FP16 hi/lo (residual scaled by 1024) and cuBLAS FP32 output GEMMs.
namespace apx_compensated {
inline void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
inline void check(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("compensated cuBLAS status "+std::to_string(s));}
// Optional offline tactic search, owned by one prepared operation. No persistent
// recipes, shared handles, environment switches or enqueue-time selection.
struct TunedMatmul {
 cublasLtHandle_t lt{};cublasLtMatmulDesc_t desc{};
 cublasLtMatrixLayout_t a{},b{},c{};cublasLtMatmulAlgo_t algo{};
 void*workspace{};static constexpr size_t bytes=4*1024*1024;size_t workspace_bytes=bytes;bool enabled=false;
 TunedMatmul(int m,int n,int k){
  try{
   check(cublasLtCreate(&lt));check(cublasLtMatmulDescCreate(&desc,CUBLAS_COMPUTE_32F,CUDA_R_32F));
   cublasOperation_t tr=CUBLAS_OP_T;check(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&tr,sizeof(tr)));
   check(cublasLtMatrixLayoutCreate(&a,CUDA_R_16F,k,n,k));check(cublasLtMatrixLayoutCreate(&b,CUDA_R_16F,k,m,k));check(cublasLtMatrixLayoutCreate(&c,CUDA_R_32F,n,m,n));check(cudaMalloc(&workspace,bytes));
  }catch(...){clear();throw;}
 }
 void clear(){if(workspace)cudaFree(workspace);if(c)cublasLtMatrixLayoutDestroy(c);if(b)cublasLtMatrixLayoutDestroy(b);if(a)cublasLtMatrixLayoutDestroy(a);if(desc)cublasLtMatmulDescDestroy(desc);if(lt)cublasLtDestroy(lt);}
 ~TunedMatmul(){clear();}
 void multiply(cudaStream_t stream,const __half*w,const __half*x,float*y,float alpha,float beta){
  check(cublasLtMatmul(lt,desc,&alpha,w,a,x,b,&beta,y,c,y,c,&algo,workspace,workspace_bytes,stream));
 }
};
struct Execution {
 std::unique_ptr<TunedMatmul> tuned;
 cublasLtHandle_t lt{};cublasLtMatmulDesc_t desc{};cublasLtMatrixLayout_t aw{},bx{},cy{};
 cublasLtMatmulAlgo_t algo{};void*lt_workspace{};size_t lt_bytes=0;bool fused_bias=false;bool norm_input=false;bool gelu_input=false;float norm_eps=1e-6f;
 cudaStream_t stream;cublasHandle_t blas{};__half*xh{},*xl{},*wh{},*wl{};float*col{},*wf{};
 int columns=0;
 int kind,m=0,n=0,k=0,batch=0,ci=0,length=0,co=0,kernel=0,out=0,stride=0,pad=0;
 const float*x;const float*w;const float*b;float*y;
 Execution(cudaStream_t st,int op,const int*p,const float*a,const float*weight,const float*bias,float*output,float eps=1e-6f)
 :stream(st),kind(op==26||op==27?19:op),x(a),w(weight),b(bias),y(output){
  norm_input=op==26;gelu_input=op==27;norm_eps=eps;
  if(kind==19){m=p[0];n=p[1];k=p[2];}
  else{batch=p[0];ci=p[1];length=p[2];co=p[3];kernel=p[4];out=p[5];stride=p[6];pad=p[7];columns=kind==20?(out+15)/16*16:kind==21?(length+15)/16*16:length;}
 }
 ~Execution(){if(lt_workspace)cudaFree(lt_workspace);if(aw)cublasLtMatrixLayoutDestroy(aw);if(bx)cublasLtMatrixLayoutDestroy(bx);if(cy)cublasLtMatrixLayoutDestroy(cy);if(desc)cublasLtMatmulDescDestroy(desc);if(lt)cublasLtDestroy(lt);if(xh)cudaFree(xh);if(xl)cudaFree(xl);if(wh)cudaFree(wh);if(wl)cudaFree(wl);if(col)cudaFree(col);if(wf)cudaFree(wf);if(blas)cublasDestroy(blas);}
 void prepare(){
  check(cublasCreate(&blas));check(cublasSetStream(blas,stream));
  check(cublasSetMathMode(blas,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
  size_t nx=kind==19?size_t(m)*k:kind==20?size_t(ci)*kernel*columns:size_t(ci)*columns;
  size_t nw=kind==19?size_t(n)*k:size_t(ci)*co*kernel;
  size_t nc=kind==20?size_t(co)*columns:kind>=21?size_t(co)*kernel*columns:0;
  if(nx>INT32_MAX||nw>INT32_MAX||nc>INT32_MAX||(nx+nc)*4>64*1024*1024)throw std::runtime_error("compensated scratch/shape exceeds bound");
  if(nc)check(cudaMalloc(&col,nc*4));
  if(kind==22){check(cudaMalloc(&wf,nw*4));pack<<<(nw+255)/256,256,0,stream>>>(int(nw),co,kernel,w,nullptr,nullptr,wf);}
  else{
   check(cudaMalloc(&xh,nx*2));check(cudaMalloc(&xl,nx*2));check(cudaMalloc(&wh,nw*2));check(cudaMalloc(&wl,nw*2));
   if(kind==21)pack<<<(nw+255)/256,256,0,stream>>>(int(nw),co,kernel,w,wh,wl,nullptr);
   else split<<<(nw+255)/256,256,0,stream>>>(int(nw),w,wh,wl);
  }
  if(kind==19&&b){
   check(cublasLtCreate(&lt));check(cublasLtMatmulDescCreate(&desc,CUBLAS_COMPUTE_32F,CUDA_R_32F));
   cublasOperation_t tr=CUBLAS_OP_T;check(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&tr,sizeof(tr)));
   cublasLtEpilogue_t ep=CUBLASLT_EPILOGUE_BIAS;check(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_EPILOGUE,&ep,sizeof(ep)));
   cudaDataType_t bias_type=CUDA_R_32F;check(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,&bias_type,sizeof(bias_type)));
   check(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&b,sizeof(b)));
   check(cublasLtMatrixLayoutCreate(&aw,CUDA_R_16F,k,n,k));check(cublasLtMatrixLayoutCreate(&bx,CUDA_R_16F,k,m,k));check(cublasLtMatrixLayoutCreate(&cy,CUDA_R_32F,n,m,n));
   cublasLtMatmulPreference_t pref;check(cublasLtMatmulPreferenceCreate(&pref));size_t limit=4*1024*1024;
   auto rc=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&limit,sizeof(limit));
   cublasLtMatmulHeuristicResult_t h{};int found=0;
   if(rc==CUBLAS_STATUS_SUCCESS)rc=cublasLtMatmulAlgoGetHeuristic(lt,desc,aw,bx,cy,cy,pref,1,&h,&found);
   cublasLtMatmulPreferenceDestroy(pref);
   if(rc==CUBLAS_STATUS_SUCCESS&&found&&h.state==CUBLAS_STATUS_SUCCESS){algo=h.algo;lt_bytes=h.workspaceSize;if(lt_bytes)check(cudaMalloc(&lt_workspace,lt_bytes));fused_bias=true;}
  }
  check(cudaPeekAtLastError());
 }
 void gemm(cublasOperation_t ta,int gm,int gn,int gk,const __half*ah,const __half*al,int lda,const __half*bh,const __half*bl,int ldb,float*z,int ldc){
  float one=1.f,zero=0.f,small=1.f/1024;
  if(tuned&&tuned->enabled){
   tuned->multiply(stream,ah,bh,z,one,zero);tuned->multiply(stream,al,bh,z,small,one);
   if(fused_bias)check(cublasLtMatmul(lt,desc,&small,ah,aw,bl,bx,&one,z,cy,z,cy,&algo,lt_workspace,lt_bytes,stream));
   else tuned->multiply(stream,ah,bl,z,small,one);
   return;
  }
  check(cublasGemmEx(blas,ta,CUBLAS_OP_N,gm,gn,gk,&one,ah,CUDA_R_16F,lda,bh,CUDA_R_16F,ldb,&zero,z,CUDA_R_32F,ldc,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  check(cublasGemmEx(blas,ta,CUBLAS_OP_N,gm,gn,gk,&small,al,CUDA_R_16F,lda,bh,CUDA_R_16F,ldb,&one,z,CUDA_R_32F,ldc,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  if(fused_bias)check(cublasLtMatmul(lt,desc,&small,ah,aw,bl,bx,&one,z,cy,z,cy,&algo,lt_workspace,lt_bytes,stream));
  else check(cublasGemmEx(blas,ta,CUBLAS_OP_N,gm,gn,gk,&small,ah,CUDA_R_16F,lda,bl,CUDA_R_16F,ldb,&one,z,CUDA_R_32F,ldc,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
 }
 void run(){
  if(kind==19){
   if(norm_input){
    if(k==384)norm_split<12><<<(m+3)/4,128,0,stream>>>(m,k,x,b+n,b+n+k,xh,xl,norm_eps);
    else norm_split<32><<<(m+3)/4,128,0,stream>>>(m,k,x,b+n,b+n+k,xh,xl,norm_eps);
   }
   else if(gelu_input)gelu_split<<<(m*k+255)/256,256,0,stream>>>(m*k,x,xh,xl);
   else split<<<(m*k+255)/256,256,0,stream>>>(m*k,x,xh,xl);
   gemm(CUBLAS_OP_T,n,m,k,wh,wl,k,xh,xl,k,y,n);
   if(b&&!fused_bias)bias<<<(m*n+255)/256,256,0,stream>>>(m*n,n,1,b,y);
  }else for(int ib=0;ib<batch;ib++){
   const float*input=x+ib*ci*length;float*output=y+ib*co*out;
   if(kind==20){
    int count=ci*kernel*columns;lower<<<(count+255)/256,256,0,stream>>>(count,length,columns,kernel,stride,pad,ci,input,xh,xl);
    gemm(CUBLAS_OP_T,co,columns,ci*kernel,wh,wl,ci*kernel,xh,xl,ci*kernel,col,co);
    unpad<<<(co*out+255)/256,256,0,stream>>>(co*out,out,columns,co,col,b,output);
   }else{
    if(kind==22){float one=1,zero=0;check(cublasSgemm(blas,CUBLAS_OP_N,CUBLAS_OP_N,length,co*kernel,ci,&one,input,length,wf,ci,&zero,col,length));}
    else{int count=ci*columns;lower<<<(count+255)/256,256,0,stream>>>(count,length,columns,1,1,0,ci,input,xh,xl);gemm(CUBLAS_OP_T,co*kernel,columns,ci,wh,wl,ci,xh,xl,ci,col,co*kernel);}
    gather<<<(co*out+255)/256,256,0,stream>>>(co*out,length,columns,out,kernel,stride,pad,co,kind!=22,col,b,output);
   }
  }
  check(cudaPeekAtLastError());
 }
 void tune(){
  if(kind==22||tuned)return;
  check(cudaStreamSynchronize(stream));
  auto next=std::make_unique<TunedMatmul>(kind==19?m:columns,kind==19?n:kind==20?co:co*kernel,kind==19?k:kind==20?ci*kernel:ci);
  cublasLtMatmulPreference_t pref{};check(cublasLtMatmulPreferenceCreate(&pref));
  cublasLtMatmulHeuristicResult_t choices[32]{};int found=0;
  auto rc=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&TunedMatmul::bytes,sizeof(size_t));
  if(rc==CUBLAS_STATUS_SUCCESS)rc=cublasLtMatmulAlgoGetHeuristic(next->lt,next->desc,next->a,next->b,next->c,next->c,pref,32,choices,&found);
  cublasLtMatmulPreferenceDestroy(pref);check(rc);
  if(!found)return;
  tuned=std::move(next);
  cudaEvent_t start{},end{};
  try{
   check(cudaEventCreate(&start));check(cudaEventCreate(&end));
   auto measure=[&](){
    for(int i=0;i<3;i++)run();check(cudaStreamSynchronize(stream));
    cudaGraph_t graph{};cudaGraphExec_t exec{};
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    try{for(int i=0;i<8;i++)run();}
    catch(...){cudaStreamEndCapture(stream,&graph);if(graph)cudaGraphDestroy(graph);throw;}
    check(cudaStreamEndCapture(stream,&graph));
    try{
     check(cudaGraphInstantiate(&exec,graph,0));
     for(int i=0;i<3;i++)check(cudaGraphLaunch(exec,stream));
     check(cudaEventRecord(start,stream));for(int i=0;i<10;i++)check(cudaGraphLaunch(exec,stream));
     check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
     float ms;check(cudaEventElapsedTime(&ms,start,end));cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);return ms;
    }catch(...){cudaStreamSynchronize(stream);if(exec)cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);throw;}
   };
   float best=measure();int winner=-1;
   for(int i=0;i<found;i++)if(choices[i].state==CUBLAS_STATUS_SUCCESS){
    tuned->enabled=true;tuned->algo=choices[i].algo;float ms=measure();
    if(ms<best*.98f){best=ms;winner=i;}
   }
   tuned->enabled=winner>=0;if(winner>=0)tuned->algo=choices[winner].algo;
   if(winner>=0&&choices[winner].workspaceSize<TunedMatmul::bytes){
    check(cudaFree(tuned->workspace));tuned->workspace=nullptr;tuned->workspace_bytes=choices[winner].workspaceSize;
    if(tuned->workspace_bytes)check(cudaMalloc(&tuned->workspace,tuned->workspace_bytes));
   }
   if(std::getenv("APXINF_TENSOR_TUNING_LOG")){
    int id=-1;size_t size=0;if(winner>=0)check(cublasLtMatmulAlgoConfigGetAttribute(&tuned->algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&size));
    std::fprintf(stderr,"compensated_tune kind=%d M=%d N=%d K=%d candidates=%d winner=%d algo=%d workspace=%zu\n",kind,kind==19?m:columns,kind==19?n:kind==20?co:co*kernel,kind==19?k:kind==20?ci*kernel:ci,found,winner,id,winner>=0?tuned->workspace_bytes:0);
   }
   run();check(cudaStreamSynchronize(stream));
  }catch(...){if(start)cudaEventDestroy(start);if(end)cudaEventDestroy(end);tuned.reset();throw;}
  cudaEventDestroy(start);cudaEventDestroy(end);
  if(!tuned->enabled)tuned.reset();
 }
};
}
