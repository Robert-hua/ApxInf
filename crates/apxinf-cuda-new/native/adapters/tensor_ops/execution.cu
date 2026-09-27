#include "apxinf_cuda/tensor_ops.h"
#include "../../kernels/tensor_ops/primitives.cuh"
#include <cudnn.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
namespace {
thread_local std::string error;
void ck(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
void ck(cudnnStatus_t s){if(s!=CUDNN_STATUS_SUCCESS)throw std::runtime_error(cudnnGetErrorString(s));}
void ck(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS status "+std::to_string(s));}
template<class F> int guard(F f){try{f();error.clear();return 0;}catch(const std::exception&e){error=e.what();return 1;}}
struct Context{cudaStream_t stream{};cudnnHandle_t dn{};cublasHandle_t bl{};bool tf32=false;bool bf16=false;bool fp8_conv=false;
 ~Context(){if(bl)cublasDestroy(bl);if(dn)cudnnDestroy(dn);if(stream)cudaStreamDestroy(stream);}};
struct Fp8Conv {
 cublasLtHandle_t handle{};cublasLtMatmulDesc_t desc{};
 cublasLtMatrixLayout_t a{},b{},y{};cublasLtMatmulAlgo_t algo{};
 __nv_fp8_e4m3*activation{};__nv_fp8_e4m3*weight{};float*scale_a{};float*scale_b{};
 void*workspace{};size_t bytes{};
 ~Fp8Conv(){if(activation)cudaFree(activation);if(weight)cudaFree(weight);if(scale_a)cudaFree(scale_a);if(scale_b)cudaFree(scale_b);if(workspace)cudaFree(workspace);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(y)cublasLtMatrixLayoutDestroy(y);if(desc)cublasLtMatmulDescDestroy(desc);if(handle)cublasLtDestroy(handle);}
};
struct Bf16Linear {
 cublasLtHandle_t handle{};cublasLtMatmulDesc_t desc{};
 cublasLtMatrixLayout_t w{},x{},y{};cublasLtMatmulAlgo_t algo{};
 __nv_bfloat16*bias{};void*workspace{};size_t bytes{};
 ~Bf16Linear(){if(bias)cudaFree(bias);if(workspace)cudaFree(workspace);if(w)cublasLtMatrixLayoutDestroy(w);if(x)cublasLtMatrixLayoutDestroy(x);if(y)cublasLtMatrixLayoutDestroy(y);if(desc)cublasLtMatmulDescDestroy(desc);if(handle)cublasLtDestroy(handle);}
};
struct Execution{Context*ctx{};apx_tensor_spec s{};const float*a{};const float*b{};const float*c{};float*y{};
 cudnnTensorDescriptor_t xdesc{},ydesc{};cudnnFilterDescriptor_t wdesc{};cudnnConvolutionDescriptor_t conv{};
 void*workspace{};size_t workspace_bytes{};
 std::unique_ptr<Bf16Linear> linear;std::unique_ptr<Fp8Conv> fp8;bool nchw=false;bool im2col_1d=false;bool bf16_im2row=false;int padded_columns=0;
 __nv_bfloat16*abf{};__nv_bfloat16*bbf{};__nv_bfloat16*ybf{};int an{},bn{},yn{};
 cudnnConvolutionFwdAlgo_t forward_algo=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
 cudnnConvolutionBwdDataAlgo_t backward_algo=CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
 ~Execution(){if(abf)cudaFree(abf);if(bbf)cudaFree(bbf);if(ybf)cudaFree(ybf);if(workspace)cudaFree(workspace);if(xdesc)cudnnDestroyTensorDescriptor(xdesc);if(ydesc)cudnnDestroyTensorDescriptor(ydesc);if(wdesc)cudnnDestroyFilterDescriptor(wdesc);if(conv)cudnnDestroyConvolutionDescriptor(conv);}};
void enqueue(Execution&e){auto&p=e.s.p;auto&f=e.s.f;auto st=e.ctx->stream;float one=1.f,zero=0.f;
 if(e.abf&&!e.bf16_im2row){if((e.s.kind==1||e.s.kind==2)&&!e.nchw)apx_pack_nhwc<<<(e.an+255)/256,256,0,st>>>(e.an,p[1],p[2]*p[3],e.a,e.abf);else apx_to_bf16<<<(e.an+255)/256,256,0,st>>>(e.an,e.a,e.abf);if(e.s.kind==10)apx_to_bf16<<<(e.bn+255)/256,256,0,st>>>(e.bn,e.b,e.bbf);}
 switch(e.s.kind){
 case 0: // row-major Y[M,N] = A[M,K] W[N,K]^T
  if(e.linear){
   auto&q=*e.linear;
   ck(cublasLtMatmul(q.handle,q.desc,&one,e.bbf,q.w,e.abf,q.x,&zero,e.ybf,q.y,e.ybf,q.y,&q.algo,q.workspace,q.bytes,st));
   apx_from_bf16<<<(e.yn+255)/256,256,0,st>>>(e.yn,e.ybf,e.y);break;
  }
  if(e.ctx->bf16){
   ck(cublasGemmEx(e.ctx->bl,CUBLAS_OP_T,CUBLAS_OP_N,p[1],p[0],p[2],&one,e.bbf,CUDA_R_16BF,p[2],e.abf,CUDA_R_16BF,p[2],&zero,e.y,CUDA_R_32F,p[1],CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
   apx_round_bf16<<<(e.yn+255)/256,256,0,st>>>(e.yn,e.y,e.c,1,p[1]);break;
  }
  ck(cublasSgemm(e.ctx->bl,CUBLAS_OP_T,CUBLAS_OP_N,p[1],p[0],p[2],&one,e.b,p[2],e.a,p[2],&zero,e.y,p[1]));
  if(e.c)apx_elementwise<<<(p[0]*p[1]+255)/256,256,0,st>>>(p[0]*p[1],e.y,e.c,nullptr,e.y,6,1,p[1],0,0);break;
 case 1: case 2:
  if(e.fp8){
   auto&q=*e.fp8;int k=p[1]*p[6],columns=e.padded_columns;
   for(int batch=0;batch<p[0];batch++){
    const float*x=e.a+batch*p[1]*p[3];int n=p[1]*p[3];
    ck(cudaMemsetAsync(q.scale_a,0,sizeof(float),st));
    apx_bf16_absmax<<<(n+255)/256,256,0,st>>>(n,x,q.scale_a);apx_fp8_scale<<<1,1,0,st>>>(q.scale_a);
    apx_im2row_fp8<<<(k*columns+255)/256,256,0,st>>>(k*columns,p[3],p[9],p[6],p[13],p[11],k,x,q.scale_a,q.activation);
    ck(cublasLtMatmul(q.handle,q.desc,&one,q.activation,q.a,q.weight,q.b,&zero,e.ybf,q.y,e.ybf,q.y,&q.algo,q.workspace,q.bytes,st));
    apx_unpack_conv1d_bf16<<<(p[4]*p[9]+255)/256,256,0,st>>>(p[4]*p[9],p[9],columns,e.ybf,e.c,e.y+batch*p[4]*p[9]);
   }break;
  }
  if(e.bf16_im2row){
   int k=p[1]*p[6],columns=e.padded_columns;
   for(int batch=0;batch<p[0];batch++){
    apx_im2row_bf16<<<(k*columns+255)/256,256,0,st>>>(k*columns,p[3],p[9],p[6],p[13],p[11],k,e.a+batch*p[1]*p[3],e.abf);
    ck(cublasGemmEx(e.ctx->bl,CUBLAS_OP_T,CUBLAS_OP_N,columns,p[4],k,&one,e.abf,CUDA_R_16BF,k,e.bbf,CUDA_R_16BF,k,&zero,e.ybf,CUDA_R_16BF,columns,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    apx_unpack_conv1d_bf16<<<(p[4]*p[9]+255)/256,256,0,st>>>(p[4]*p[9],p[9],columns,e.ybf,e.c,e.y+batch*p[4]*p[9]);
   }break;
  }
  if(e.im2col_1d){
   int k=p[1]*p[6],columns=p[9],elements=k*columns;
   for(int batch=0;batch<p[0];batch++){
    auto*col=static_cast<float*>(e.workspace);
    apx_im2col_1d<<<(elements+255)/256,256,0,st>>>(elements,p[3],columns,p[6],p[13],p[11],e.a+batch*p[1]*p[3],col);
    ck(cublasSgemm(e.ctx->bl,CUBLAS_OP_N,CUBLAS_OP_N,columns,p[4],k,&one,col,columns,e.b,k,&zero,e.y+batch*p[4]*columns,columns));
   }
   if(e.c)apx_elementwise<<<(p[0]*p[4]*p[9]+255)/256,256,0,st>>>(p[0]*p[4]*p[9],e.y,e.c,nullptr,e.y,6,p[9],p[4],0,0);
   break;
  }
  if(e.s.kind==1)ck(cudnnConvolutionForward(e.ctx->dn,&one,e.xdesc,e.abf?static_cast<void*>(e.abf):const_cast<float*>(e.a),e.wdesc,e.bbf?static_cast<void*>(e.bbf):const_cast<float*>(e.b),e.conv,e.forward_algo,e.workspace,e.workspace_bytes,&zero,e.ydesc,e.ybf?static_cast<void*>(e.ybf):e.y));
  else ck(cudnnConvolutionBackwardData(e.ctx->dn,&one,e.wdesc,e.bbf?static_cast<void*>(e.bbf):const_cast<float*>(e.b),e.xdesc,e.abf?static_cast<void*>(e.abf):const_cast<float*>(e.a),e.conv,e.backward_algo,e.workspace,e.workspace_bytes,&zero,e.ydesc,e.ybf?static_cast<void*>(e.ybf):e.y));
  if(e.ybf){if(e.nchw)apx_from_bf16<<<(e.yn+255)/256,256,0,st>>>(e.yn,e.ybf,e.y);else apx_unpack_nhwc<<<(e.yn+255)/256,256,0,st>>>(e.yn,p[4],p[8]*p[9],e.ybf,e.y);if(e.c)apx_round_bf16<<<(e.yn+255)/256,256,0,st>>>(e.yn,e.y,e.c,p[8]*p[9],p[4]);break;}
  if(e.c)apx_elementwise<<<(p[0]*p[4]*p[8]*p[9]+255)/256,256,0,st>>>(p[0]*p[4]*p[8]*p[9],e.y,e.c,nullptr,e.y,6,p[8]*p[9],p[4],0,0);break;
 case 3:apx_elementwise<<<(p[0]+255)/256,256,0,st>>>(p[0],e.a,e.b,e.c,e.y,p[1],p[2],p[3],f[0],f[1]);break;
 case 4:if(p[3]>1)apx_group_normalize<<<p[0],p[1]<512?32:512,0,st>>>(p[1],p[2],p[3],e.a,e.b,e.c,e.y,f[0]);else apx_normalize<<<p[0],256,0,st>>>(p[0],p[1],p[2],p[3],e.a,e.b,e.c,e.y,f[0]);break;
 case 5:apx_softmax<<<p[0],256,0,st>>>(p[0],p[1],e.a,e.y,f[0]);break;
 case 6:apx_permute<<<(p[0]+255)/256,256,0,st>>>(p[0],e.a,e.y,p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8]);break;
 case 7:apx_concat<<<(p[0]+255)/256,256,0,st>>>(p[0],e.a,e.b,e.y,p[1],p[2],p[3]);break;
 case 8:apx_slice<<<(p[0]+255)/256,256,0,st>>>(p[0],e.a,e.y,p[1],p[2],p[3],p[4]);break;
 case 9:apx_pool<<<(p[0]+255)/256,256,0,st>>>(p[0],e.a,e.y,p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8]);break;
 case 10: // batched row-major A[B,M,K] times B[B,K,N] or B[B,N,K]
  if(e.ctx->bf16){
   ck(cublasGemmStridedBatchedEx(e.ctx->bl,p[4]?CUBLAS_OP_T:CUBLAS_OP_N,CUBLAS_OP_N,p[2],p[1],p[3],&f[0],e.bbf,CUDA_R_16BF,p[4]?p[3]:p[2],(long long)p[2]*p[3],e.abf,CUDA_R_16BF,p[3],(long long)p[1]*p[3],&zero,e.y,CUDA_R_32F,p[2],(long long)p[1]*p[2],p[0],CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
   apx_round_bf16<<<(e.yn+255)/256,256,0,st>>>(e.yn,e.y,nullptr,1,1);break;
  }
  ck(cublasSgemmStridedBatched(e.ctx->bl,p[4]?CUBLAS_OP_T:CUBLAS_OP_N,CUBLAS_OP_N,p[2],p[1],p[3],&f[0],e.b,p[4]?p[3]:p[2],(long long)p[2]*p[3],e.a,p[3],(long long)p[1]*p[3],&zero,e.y,p[2],(long long)p[1]*p[2],p[0]));break;
 case 11:apx_axpby<<<(p[0]+255)/256,256,0,st>>>(p[0],e.a,e.b,e.c,e.y,f[0],f[1],f[2],f[3]);break;
 case 12:apx_batch_norm<<<(p[0]+255)/256,256,0,st>>>(p[0],p[1],p[2],e.a,e.b,e.y,f[0],p[3],p[4]);break;
 default:throw std::runtime_error("unsupported tensor semantic");
 }ck(cudaPeekAtLastError());
}
}
extern "C" const char* apx_tensor_error(){return error.c_str();}
extern "C" int apx_tensor_context_create(int device,void**out){*out=nullptr;return guard([&]{ck(cudaSetDevice(device));auto c=std::make_unique<Context>();ck(cudaStreamCreateWithFlags(&c->stream,cudaStreamNonBlocking));ck(cudnnCreate(&c->dn));ck(cudnnSetStream(c->dn,c->stream));ck(cublasCreate(&c->bl));ck(cublasSetStream(c->bl,c->stream));ck(cublasSetMathMode(c->bl,CUBLAS_PEDANTIC_MATH));*out=c.release();});}
extern "C" void apx_tensor_context_destroy(void*c){delete static_cast<Context*>(c);}
extern "C" int apx_tensor_math_mode(void*c,int mode){return guard([&]{if(mode<0||mode>3)throw std::runtime_error("unsupported precision");auto*ctx=static_cast<Context*>(c);ck(cublasSetMathMode(ctx->bl,mode==1?CUBLAS_TF32_TENSOR_OP_MATH:mode>=2?CUBLAS_DEFAULT_MATH:CUBLAS_PEDANTIC_MATH));ctx->tf32=mode==1;ctx->bf16=mode>=2;ctx->fp8_conv=mode==3;
 if(ctx->fp8_conv){int device=0;cudaDeviceProp prop{};ck(cudaGetDevice(&device));ck(cudaGetDeviceProperties(&prop,device));if(prop.major!=11||prop.minor!=0)throw std::runtime_error("experimental FP8 Conv1d is qualified for Thor sm110 only");}});}
extern "C" int apx_tensor_sync(void*c){return guard([&]{ck(cudaStreamSynchronize(static_cast<Context*>(c)->stream));});}
extern "C" int apx_tensor_prepare(void*c,const apx_tensor_spec*s,const float*a,const float*b,const float*bias,float*y,void**out){*out=nullptr;return guard([&]{
 auto e=std::make_unique<Execution>();e->ctx=static_cast<Context*>(c);e->s=*s;e->a=a;e->b=b;e->c=bias;e->y=y;auto&p=e->s.p;
 const char*layout=std::getenv("APXINF_TENSOR_BF16_LAYOUT");
 const bool reference_layout=layout&&(std::strcmp(layout,"reference")==0||std::strcmp(layout,"reference_unet")==0);
 e->nchw=reference_layout&&(p[1]==3||(p[5]==1&&p[6]==1));
 // Contraction Conv1d is sensitive to cuDNN layout-dependent reduction order.
 if(layout&&std::strcmp(layout,"reference_unet")==0&&s->kind==1&&p[2]==1&&(p[1]<=3||(p[1]>p[4]&&p[4]<=512)))e->nchw=true;
 // Experimental provider choice is frozen during prepare; enqueue does not read env,
 // allocate, or select algorithms. Other shapes/precisions retain the cuDNN path.
 const char*conv1d=std::getenv("APXINF_TENSOR_FP32_CONV1D");
 if(conv1d&&std::strcmp(conv1d,"im2col")==0&&s->kind==1&&!e->ctx->bf16&&!e->ctx->tf32&&p[2]==1&&p[5]==1&&p[8]==1&&p[10]==0&&p[12]==1){
  size_t elements=size_t(p[1])*p[6]*p[9];
  if(elements<=size_t(64*1024*1024)/sizeof(float)){
   e->im2col_1d=true;e->workspace_bytes=elements*sizeof(float);ck(cudaMalloc(&e->workspace,e->workspace_bytes));
   *out=e.release();return;
  }
 }
 // FP8 is an explicit mixed-precision context, never selected for BF16 silently.
 // Narrow input/output projections and transposed/2D convolutions remain BF16.
 if(e->ctx->fp8_conv&&s->kind==1&&p[1]>=128&&p[4]>=128&&p[1]%16==0&&p[4]%16==0&&p[2]==1&&p[5]==1&&p[8]==1&&p[10]==0&&p[12]==1){
  size_t columns=(size_t(p[9])+15)/16*16,k=size_t(p[1])*p[6];
  if(k*columns+2*size_t(p[4])*columns>64*1024*1024)throw std::runtime_error("FP8 Conv1d scratch exceeds budget");
  e->padded_columns=int(columns);e->fp8=std::make_unique<Fp8Conv>();auto&q=*e->fp8;
  ck(cudaMalloc(&q.activation,k*columns));ck(cudaMalloc(&q.weight,k*p[4]));ck(cudaMalloc(&q.scale_a,sizeof(float)));ck(cudaMalloc(&q.scale_b,sizeof(float)));ck(cudaMalloc(&e->ybf,columns*p[4]*sizeof(__nv_bfloat16)));
  ck(cudaMemsetAsync(q.scale_b,0,sizeof(float),e->ctx->stream));
  apx_bf16_absmax<<<(k*p[4]+255)/256,256,0,e->ctx->stream>>>(int(k*p[4]),b,q.scale_b);apx_fp8_scale<<<1,1,0,e->ctx->stream>>>(q.scale_b);
  apx_to_fp8<<<(k*p[4]+255)/256,256,0,e->ctx->stream>>>(int(k*p[4]),b,q.scale_b,q.weight);
  ck(cublasLtCreate(&q.handle));ck(cublasLtMatmulDescCreate(&q.desc,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t trans=CUBLAS_OP_T;ck(cublasLtMatmulDescSetAttribute(q.desc,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
  ck(cublasLtMatmulDescSetAttribute(q.desc,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&q.scale_a,sizeof(q.scale_a)));ck(cublasLtMatmulDescSetAttribute(q.desc,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&q.scale_b,sizeof(q.scale_b)));
  ck(cublasLtMatrixLayoutCreate(&q.a,CUDA_R_8F_E4M3,k,columns,k));ck(cublasLtMatrixLayoutCreate(&q.b,CUDA_R_8F_E4M3,k,p[4],k));ck(cublasLtMatrixLayoutCreate(&q.y,CUDA_R_16BF,columns,p[4],columns));
  cublasLtMatmulPreference_t pref{};ck(cublasLtMatmulPreferenceCreate(&pref));size_t limit=4*1024*1024;
  auto status=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&limit,sizeof(limit));
  if(status!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);ck(status);}
  cublasLtMatmulHeuristicResult_t result{};int count=0;status=cublasLtMatmulAlgoGetHeuristic(q.handle,q.desc,q.a,q.b,q.y,q.y,pref,1,&result,&count);cublasLtMatmulPreferenceDestroy(pref);ck(status);
  if(count!=1||result.state!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("no native FP8 Conv1d GEMM candidate");
  q.algo=result.algo;q.bytes=result.workspaceSize;if(q.bytes>limit)throw std::runtime_error("FP8 workspace exceeds budget");if(q.bytes)ck(cudaMalloc(&q.workspace,q.bytes));
  *out=e.release();return;
 }
 const char*bfconv=std::getenv("APXINF_TENSOR_BF16_CONV1D");
 if(bfconv&&std::strcmp(bfconv,"im2row")==0&&s->kind==1&&e->ctx->bf16&&!e->nchw&&p[2]==1&&p[5]==1&&p[8]==1&&p[10]==0&&p[12]==1){
  size_t columns=(size_t(p[9])+15)/16*16,k=size_t(p[1])*p[6];
  if((k*columns+size_t(p[4])*columns)*sizeof(__nv_bfloat16)<=64*1024*1024){
   e->bf16_im2row=true;e->padded_columns=int(columns);e->an=int(k*columns);e->bn=int(k*p[4]);e->yn=int(columns*p[4]);
   ck(cudaMalloc(&e->abf,e->an*sizeof(__nv_bfloat16)));ck(cudaMalloc(&e->bbf,e->bn*sizeof(__nv_bfloat16)));ck(cudaMalloc(&e->ybf,e->yn*sizeof(__nv_bfloat16)));
   apx_to_bf16<<<(e->bn+255)/256,256,0,e->ctx->stream>>>(e->bn,b,e->bbf);
   *out=e.release();return;
  }
 }
 if(e->ctx->bf16&&(s->kind==0||s->kind==1||s->kind==2||s->kind==10)){
  if(s->kind==0){e->an=p[0]*p[2];e->bn=p[1]*p[2];e->yn=p[0]*p[1];}
  else if(s->kind==10){e->an=p[0]*p[1]*p[3];e->bn=p[0]*p[2]*p[3];e->yn=p[0]*p[1]*p[2];}
  else{e->an=p[0]*p[1]*p[2]*p[3];e->bn=p[4]*p[1]*p[5]*p[6];e->yn=p[0]*p[4]*p[8]*p[9];}
  ck(cudaMalloc(&e->abf,e->an*sizeof(__nv_bfloat16)));ck(cudaMalloc(&e->bbf,e->bn*sizeof(__nv_bfloat16)));
  if(s->kind==1||s->kind==2)ck(cudaMalloc(&e->ybf,e->yn*sizeof(__nv_bfloat16)));
  if((s->kind==1||s->kind==2)&&!e->nchw)apx_pack_nhwc<<<(e->bn+255)/256,256,0,e->ctx->stream>>>(e->bn,s->kind==1?p[1]:p[4],p[5]*p[6],b,e->bbf);
  else if(s->kind!=10)apx_to_bf16<<<(e->bn+255)/256,256,0,e->ctx->stream>>>(e->bn,b,e->bbf);
 }
 if(e->ctx->bf16&&s->kind==0&&bias&&p[1]>1&&p[2]>1){
  // Match autocast Linear: BF16 operands and bias, FP32 accumulation,
  // fused bias epilogue, then one BF16 output rounding. Prepare owns all resources.
  e->linear=std::make_unique<Bf16Linear>();auto&q=*e->linear;
  ck(cudaMalloc(&q.bias,p[1]*sizeof(__nv_bfloat16)));ck(cudaMalloc(&e->ybf,e->yn*sizeof(__nv_bfloat16)));
  apx_to_bf16<<<(p[1]+255)/256,256,0,e->ctx->stream>>>(p[1],bias,q.bias);
  ck(cublasLtCreate(&q.handle));ck(cublasLtMatmulDescCreate(&q.desc,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t trans=CUBLAS_OP_T;ck(cublasLtMatmulDescSetAttribute(q.desc,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
  cublasLtEpilogue_t epilogue=CUBLASLT_EPILOGUE_BIAS;ck(cublasLtMatmulDescSetAttribute(q.desc,CUBLASLT_MATMUL_DESC_EPILOGUE,&epilogue,sizeof(epilogue)));
  ck(cublasLtMatmulDescSetAttribute(q.desc,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&q.bias,sizeof(q.bias)));
  ck(cublasLtMatrixLayoutCreate(&q.w,CUDA_R_16BF,p[2],p[1],p[2]));ck(cublasLtMatrixLayoutCreate(&q.x,CUDA_R_16BF,p[2],p[0],p[2]));ck(cublasLtMatrixLayoutCreate(&q.y,CUDA_R_16BF,p[1],p[0],p[1]));
  cublasLtMatmulPreference_t pref{};ck(cublasLtMatmulPreferenceCreate(&pref));size_t limit=1024*1024;
  auto status=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&limit,sizeof(limit));
  if(status!=CUBLAS_STATUS_SUCCESS){cublasLtMatmulPreferenceDestroy(pref);ck(status);}
  cublasLtMatmulHeuristicResult_t result{};int count=0;status=cublasLtMatmulAlgoGetHeuristic(q.handle,q.desc,q.w,q.x,q.y,q.y,pref,1,&result,&count);cublasLtMatmulPreferenceDestroy(pref);ck(status);
  if(count!=1||result.state!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("no BF16 Linear GEMM candidate");
  q.algo=result.algo;q.bytes=result.workspaceSize;if(q.bytes>limit)throw std::runtime_error("Linear workspace exceeds budget");if(q.bytes)ck(cudaMalloc(&q.workspace,q.bytes));
 }
 if(s->kind==1||s->kind==2){
  ck(cudnnCreateTensorDescriptor(&e->xdesc));ck(cudnnCreateTensorDescriptor(&e->ydesc));ck(cudnnCreateFilterDescriptor(&e->wdesc));ck(cudnnCreateConvolutionDescriptor(&e->conv));
  auto dtype=e->ctx->bf16?CUDNN_DATA_BFLOAT16:CUDNN_DATA_FLOAT;
  auto format=e->ctx->bf16&&!e->nchw?CUDNN_TENSOR_NHWC:CUDNN_TENSOR_NCHW;
  ck(cudnnSetTensor4dDescriptor(e->xdesc,format,dtype,p[0],p[1],p[2],p[3]));
  ck(cudnnSetTensor4dDescriptor(e->ydesc,format,dtype,p[0],p[4],p[8],p[9]));
  ck(cudnnSetFilter4dDescriptor(e->wdesc,dtype,format,s->kind==1?p[4]:p[1],s->kind==1?p[1]:p[4],p[5],p[6]));
  ck(cudnnSetConvolution2dDescriptor(e->conv,p[10],p[11],p[12],p[13],1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT));
  ck(cudnnSetConvolutionMathType(e->conv,e->ctx->bf16?CUDNN_TENSOR_OP_MATH:e->ctx->tf32?CUDNN_DEFAULT_MATH:CUDNN_FMA_MATH));
  if(s->kind==1){
   int count=0;cudnnConvolutionFwdAlgoPerf_t candidates[16]{};
   ck(cudnnGetConvolutionForwardAlgorithm_v7(e->ctx->dn,e->xdesc,e->wdesc,e->conv,e->ydesc,16,&count,candidates));
   for(int i=0;i<count;i++)if(candidates[i].status==CUDNN_STATUS_SUCCESS&&candidates[i].memory<=64*1024*1024&&candidates[i].determinism==CUDNN_DETERMINISTIC&&(e->ctx->tf32||e->ctx->bf16||candidates[i].mathType==CUDNN_FMA_MATH)){
    e->forward_algo=candidates[i].algo;ck(cudnnSetConvolutionMathType(e->conv,candidates[i].mathType));break;
   }
  }else{
   int count=0;cudnnConvolutionBwdDataAlgoPerf_t candidates[16]{};
   ck(cudnnGetConvolutionBackwardDataAlgorithm_v7(e->ctx->dn,e->wdesc,e->xdesc,e->conv,e->ydesc,16,&count,candidates));
   for(int i=0;i<count;i++)if(candidates[i].status==CUDNN_STATUS_SUCCESS&&candidates[i].memory<=64*1024*1024&&candidates[i].determinism==CUDNN_DETERMINISTIC&&(e->ctx->tf32||e->ctx->bf16||candidates[i].mathType==CUDNN_FMA_MATH)){
    e->backward_algo=candidates[i].algo;ck(cudnnSetConvolutionMathType(e->conv,candidates[i].mathType));break;
   }
  }
  if(s->kind==1)ck(cudnnGetConvolutionForwardWorkspaceSize(e->ctx->dn,e->xdesc,e->wdesc,e->conv,e->ydesc,e->forward_algo,&e->workspace_bytes));
  else ck(cudnnGetConvolutionBackwardDataWorkspaceSize(e->ctx->dn,e->wdesc,e->xdesc,e->conv,e->ydesc,e->backward_algo,&e->workspace_bytes));
  if(e->workspace_bytes)ck(cudaMalloc(&e->workspace,e->workspace_bytes));
 }*out=e.release();});}
extern "C" int apx_tensor_enqueue(void*e){return guard([&]{enqueue(*static_cast<Execution*>(e));});}
extern "C" void apx_tensor_destroy(void*e){delete static_cast<Execution*>(e);}
extern "C" int apx_tensor_capture_begin(void*c){return guard([&]{ck(cudaStreamBeginCapture(static_cast<Context*>(c)->stream,cudaStreamCaptureModeThreadLocal));});}
extern "C" int apx_tensor_capture_end(void*c,void**out){*out=nullptr;return guard([&]{cudaGraph_t g{};ck(cudaStreamEndCapture(static_cast<Context*>(c)->stream,&g));cudaGraphExec_t x{};auto s=cudaGraphInstantiate(&x,g,nullptr,nullptr,0);cudaGraphDestroy(g);ck(s);*out=x;});}
extern "C" int apx_tensor_graph_replay(void*c,void*g){return guard([&]{ck(cudaGraphLaunch(static_cast<cudaGraphExec_t>(g),static_cast<Context*>(c)->stream));});}
extern "C" void apx_tensor_graph_destroy(void*g){if(g)cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(g));}

// Explicit offline preparation using the most recent real activation buffers.
// This is never called by enqueue or inside graph capture.
extern "C" int apx_tensor_tune(void*raw){return guard([&]{
 auto&e=*static_cast<Execution*>(raw);if(e.s.kind!=1||e.im2col_1d||e.bf16_im2row||e.fp8)return;
 auto&p=e.s.p;ck(cudaStreamSynchronize(e.ctx->stream));
 if(e.abf&&!e.nchw)apx_pack_nhwc<<<(e.an+255)/256,256,0,e.ctx->stream>>>(e.an,p[1],p[2]*p[3],e.a,e.abf);
 void*scratch=nullptr;constexpr size_t limit=64*1024*1024;ck(cudaMalloc(&scratch,limit));
 int count=0;cudnnConvolutionFwdAlgoPerf_t candidates[16]{};
 auto status=cudnnFindConvolutionForwardAlgorithmEx(e.ctx->dn,e.xdesc,e.abf?static_cast<void*>(e.abf):const_cast<float*>(e.a),e.wdesc,e.bbf?static_cast<void*>(e.bbf):const_cast<float*>(e.b),e.conv,e.ydesc,e.ybf?static_cast<void*>(e.ybf):e.y,16,&count,candidates,scratch,limit);
 cudaFree(scratch);ck(status);
 for(int i=0;i<count;i++)if(candidates[i].status==CUDNN_STATUS_SUCCESS&&candidates[i].memory<=limit&&candidates[i].determinism==CUDNN_DETERMINISTIC&&(e.ctx->bf16||e.ctx->tf32||candidates[i].mathType==CUDNN_FMA_MATH)){
  size_t bytes=0;ck(cudnnSetConvolutionMathType(e.conv,candidates[i].mathType));
  ck(cudnnGetConvolutionForwardWorkspaceSize(e.ctx->dn,e.xdesc,e.wdesc,e.conv,e.ydesc,candidates[i].algo,&bytes));
  void*next=nullptr;if(bytes)ck(cudaMalloc(&next,bytes));if(e.workspace)cudaFree(e.workspace);
  e.workspace=next;e.workspace_bytes=bytes;e.forward_algo=candidates[i].algo;return;
 }
 throw std::runtime_error("no deterministic convolution tuning candidate within workspace budget");
});}

extern "C" int apx_tensor_rgb_normalize(void*c,int pixels,const unsigned char*x,float*y,const float*mean,const float*std){return guard([&]{
 auto*ctx=static_cast<Context*>(c);apx_rgb_normalize<<<(pixels*3+255)/256,256,0,ctx->stream>>>(pixels,x,y,mean[0],mean[1],mean[2],std[0],std[1],std[2]);ck(cudaPeekAtLastError());
});}
