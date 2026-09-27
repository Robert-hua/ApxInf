#include "apxinf_cuda/tensor_ops.h"
#include "../../kernels/tensor_ops/primitives.cuh"
#include <cudnn.h>
#include <cublas_v2.h>
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
struct Context{cudaStream_t stream{};cudnnHandle_t dn{};cublasHandle_t bl{};bool tf32=false;bool bf16=false;
 ~Context(){if(bl)cublasDestroy(bl);if(dn)cudnnDestroy(dn);if(stream)cudaStreamDestroy(stream);}};
struct Execution{Context*ctx{};apx_tensor_spec s{};const float*a{};const float*b{};const float*c{};float*y{};
 cudnnTensorDescriptor_t xdesc{},ydesc{};cudnnFilterDescriptor_t wdesc{};cudnnConvolutionDescriptor_t conv{};
 void*workspace{};size_t workspace_bytes{};
 bool im2col_1d=false;
 __nv_bfloat16*abf{};__nv_bfloat16*bbf{};__nv_bfloat16*ybf{};int an{},bn{},yn{};
 cudnnConvolutionFwdAlgo_t forward_algo=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
 cudnnConvolutionBwdDataAlgo_t backward_algo=CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
 ~Execution(){if(abf)cudaFree(abf);if(bbf)cudaFree(bbf);if(ybf)cudaFree(ybf);if(workspace)cudaFree(workspace);if(xdesc)cudnnDestroyTensorDescriptor(xdesc);if(ydesc)cudnnDestroyTensorDescriptor(ydesc);if(wdesc)cudnnDestroyFilterDescriptor(wdesc);if(conv)cudnnDestroyConvolutionDescriptor(conv);}};
void enqueue(Execution&e){auto&p=e.s.p;auto&f=e.s.f;auto st=e.ctx->stream;float one=1.f,zero=0.f;
 if(e.abf){if(e.s.kind==1||e.s.kind==2)apx_pack_nhwc<<<(e.an+255)/256,256,0,st>>>(e.an,p[1],p[2]*p[3],e.a,e.abf);else apx_to_bf16<<<(e.an+255)/256,256,0,st>>>(e.an,e.a,e.abf);if(e.s.kind==10)apx_to_bf16<<<(e.bn+255)/256,256,0,st>>>(e.bn,e.b,e.bbf);}
 switch(e.s.kind){
 case 0: // row-major Y[M,N] = A[M,K] W[N,K]^T
  if(e.ctx->bf16){
   ck(cublasGemmEx(e.ctx->bl,CUBLAS_OP_T,CUBLAS_OP_N,p[1],p[0],p[2],&one,e.bbf,CUDA_R_16BF,p[2],e.abf,CUDA_R_16BF,p[2],&zero,e.y,CUDA_R_32F,p[1],CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
   apx_round_bf16<<<(e.yn+255)/256,256,0,st>>>(e.yn,e.y,e.c,1,p[1]);break;
  }
  ck(cublasSgemm(e.ctx->bl,CUBLAS_OP_T,CUBLAS_OP_N,p[1],p[0],p[2],&one,e.b,p[2],e.a,p[2],&zero,e.y,p[1]));
  if(e.c)apx_elementwise<<<(p[0]*p[1]+255)/256,256,0,st>>>(p[0]*p[1],e.y,e.c,nullptr,e.y,6,1,p[1],0,0);break;
 case 1: case 2:
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
  if(e.ybf){apx_unpack_nhwc<<<(e.yn+255)/256,256,0,st>>>(e.yn,p[4],p[8]*p[9],e.ybf,e.y);if(e.c)apx_round_bf16<<<(e.yn+255)/256,256,0,st>>>(e.yn,e.y,e.c,p[8]*p[9],p[4]);break;}
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
extern "C" int apx_tensor_math_mode(void*c,int mode){return guard([&]{if(mode<0||mode>2)throw std::runtime_error("unsupported precision");auto*ctx=static_cast<Context*>(c);ck(cublasSetMathMode(ctx->bl,mode==1?CUBLAS_TF32_TENSOR_OP_MATH:mode==2?CUBLAS_DEFAULT_MATH:CUBLAS_PEDANTIC_MATH));ctx->tf32=mode==1;ctx->bf16=mode==2;});}
extern "C" int apx_tensor_sync(void*c){return guard([&]{ck(cudaStreamSynchronize(static_cast<Context*>(c)->stream));});}
extern "C" int apx_tensor_prepare(void*c,const apx_tensor_spec*s,const float*a,const float*b,const float*bias,float*y,void**out){*out=nullptr;return guard([&]{
 auto e=std::make_unique<Execution>();e->ctx=static_cast<Context*>(c);e->s=*s;e->a=a;e->b=b;e->c=bias;e->y=y;auto&p=e->s.p;
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
 if(e->ctx->bf16&&(s->kind==0||s->kind==1||s->kind==2||s->kind==10)){
  if(s->kind==0){e->an=p[0]*p[2];e->bn=p[1]*p[2];e->yn=p[0]*p[1];}
  else if(s->kind==10){e->an=p[0]*p[1]*p[3];e->bn=p[0]*p[2]*p[3];e->yn=p[0]*p[1]*p[2];}
  else{e->an=p[0]*p[1]*p[2]*p[3];e->bn=p[4]*p[1]*p[5]*p[6];e->yn=p[0]*p[4]*p[8]*p[9];}
  ck(cudaMalloc(&e->abf,e->an*sizeof(__nv_bfloat16)));ck(cudaMalloc(&e->bbf,e->bn*sizeof(__nv_bfloat16)));
  if(s->kind==1||s->kind==2)ck(cudaMalloc(&e->ybf,e->yn*sizeof(__nv_bfloat16)));
  if(s->kind==1||s->kind==2)apx_pack_nhwc<<<(e->bn+255)/256,256,0,e->ctx->stream>>>(e->bn,s->kind==1?p[1]:p[4],p[5]*p[6],b,e->bbf);
  else if(s->kind!=10)apx_to_bf16<<<(e->bn+255)/256,256,0,e->ctx->stream>>>(e->bn,b,e->bbf);
 }
 if(s->kind==1||s->kind==2){
  ck(cudnnCreateTensorDescriptor(&e->xdesc));ck(cudnnCreateTensorDescriptor(&e->ydesc));ck(cudnnCreateFilterDescriptor(&e->wdesc));ck(cudnnCreateConvolutionDescriptor(&e->conv));
  auto dtype=e->ctx->bf16?CUDNN_DATA_BFLOAT16:CUDNN_DATA_FLOAT;
  auto format=e->ctx->bf16?CUDNN_TENSOR_NHWC:CUDNN_TENSOR_NCHW;
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
 auto&e=*static_cast<Execution*>(raw);if(e.s.kind!=1||e.im2col_1d)return;
 auto&p=e.s.p;ck(cudaStreamSynchronize(e.ctx->stream));
 if(e.abf)apx_pack_nhwc<<<(e.an+255)/256,256,0,e.ctx->stream>>>(e.an,p[1],p[2]*p[3],e.a,e.abf);
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
