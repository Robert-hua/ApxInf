"""Explicit Kart TensorRT vision candidate; native policy math remains separate.

Uses CUDA runtime directly, no Torch dependency. The feature handoff is through
host memory (D2H then native H2D); it is not a zero-copy or all-native model.
"""
from pathlib import Path
import ctypes as C
import ctypes.util
import hashlib,json
import numpy as np


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


class KartTrtVision:
    def __init__(self,engine_path,model_dir,device=0):
        import tensorrt as trt
        path=Path(engine_path);self.manifest_path=path.with_suffix('.json')
        meta=json.loads(self.manifest_path.read_text())
        if (meta.get('input_contract')!='kart_dp_vision_features_v1'
            or meta.get('engine_sha256')!=_sha(path)
            or meta.get('weights_sha256')!=_sha(Path(model_dir)/'model.safetensors')):
            raise ValueError('TensorRT vision artifact/weights identity mismatch')
        self.metadata=meta;self.device=device;self.stream=C.c_void_p();self.dx=C.c_void_p();self.dy=C.c_void_p()
        self.context=None;self.engine=None;self.runtime=None;self.closed=False
        self.cuda=C.CDLL(ctypes.util.find_library('cudart') or 'libcudart.so.13')
        for name,args in {
            'cudaSetDevice':[C.c_int], 'cudaStreamCreate':[C.POINTER(C.c_void_p)],
            'cudaStreamSynchronize':[C.c_void_p],'cudaStreamDestroy':[C.c_void_p],
            'cudaMalloc':[C.POINTER(C.c_void_p),C.c_size_t],'cudaFree':[C.c_void_p],
            'cudaMemcpyAsync':[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int,C.c_void_p],
        }.items():
            fn=getattr(self.cuda,name);fn.argtypes=args;fn.restype=C.c_int
        self.logger=trt.Logger(trt.Logger.WARNING)
        try:
            self._call('cudaSetDevice',device)
            self.runtime=trt.Runtime(self.logger);self.engine=self.runtime.deserialize_cuda_engine(path.read_bytes())
            if self.engine is None:raise RuntimeError('TensorRT vision deserialization failed')
            self.context=self.engine.create_execution_context()
            specs={'images':((1,4,3,224,288),trt.TensorIOMode.INPUT),'features':((1,64,384),trt.TensorIOMode.OUTPUT)}
            if self.engine.num_io_tensors!=2:raise ValueError('Unexpected vision engine I/O count')
            for name,(shape,mode) in specs.items():
                if (tuple(self.engine.get_tensor_shape(name))!=shape or self.engine.get_tensor_dtype(name)!=trt.float32
                    or self.engine.get_tensor_mode(name)!=mode):raise ValueError('Invalid vision engine profile')
            self._call('cudaStreamCreate',C.byref(self.stream))
            self._call('cudaMalloc',C.byref(self.dx),1*4*3*224*288*4)
            self._call('cudaMalloc',C.byref(self.dy),1*64*384*4)
            self.context.set_tensor_address('images',self.dx.value);self.context.set_tensor_address('features',self.dy.value)
        except BaseException:
            self.close();raise
    def _call(self,name,*args):
        status=getattr(self.cuda,name)(*args)
        if status:raise RuntimeError(f'{name} failed: CUDA {status}')
    def infer(self,image):
        if self.closed:raise RuntimeError('TensorRT vision is closed')
        x=np.ascontiguousarray(image,dtype=np.float32)
        if x.shape!=(1,4,3,224,288) or not np.isfinite(x).all():raise ValueError('Vision expects finite FP32 RGB [1,4,3,224,288] in [0,1]')
        self._call('cudaSetDevice',self.device)
        y=np.empty((1,24576),np.float32)
        self._call('cudaMemcpyAsync',self.dx,C.c_void_p(x.ctypes.data),x.nbytes,1,self.stream)
        if not self.context.execute_async_v3(self.stream.value):raise RuntimeError('TensorRT vision execution failed')
        self._call('cudaMemcpyAsync',C.c_void_p(y.ctypes.data),self.dy,y.nbytes,2,self.stream)
        self._call('cudaStreamSynchronize',self.stream)
        return y
    def close(self):
        if self.closed:return
        self.closed=True
        self.cuda.cudaSetDevice(self.device)
        if self.stream.value:self.cuda.cudaStreamSynchronize(self.stream)
        self.context=None;self.engine=None;self.runtime=None
        if self.dx.value:self.cuda.cudaFree(self.dx);self.dx=C.c_void_p()
        if self.dy.value:self.cuda.cudaFree(self.dy);self.dy=C.c_void_p()
        if self.stream.value:self.cuda.cudaStreamDestroy(self.stream);self.stream=C.c_void_p()

    def __del__(self):
        if hasattr(self,"cuda"):
            try:self.close()
            except Exception:pass
