//! Prepared, contiguous F32 tensor operators. No model semantics or host fallback.
//! Preparation owns descriptors/workspace; enqueue performs device work only.
use crate::CudaBuffer;
use std::{
    cell::Cell,
    ffi::{c_void, CStr},
    rc::Rc,
};
type Result<T> = std::result::Result<T, String>;
#[repr(C)]
#[derive(Clone, Default)]
struct Spec {
    kind: i32,
    p: [i32; 24],
    f: [f32; 4],
}
unsafe extern "C" {
    fn apx_tensor_rgb_batch_device(c:*mut c_void,batch:i32,height:i32,width:i32,padded:i32,frames:*const usize,staging:*mut u8,y:*mut f32,mean:*const f32,std:*const f32)->i32;
    fn apx_tensor_rgb_batch(c:*mut c_void,batch:i32,height:i32,width:i32,padded:i32,x:*const u8,y:*mut f32,mean:*const f32,std:*const f32)->i32;
    fn apx_tensor_error() -> *const std::ffi::c_char;
    fn apx_tensor_context_create(device: i32, out: *mut *mut c_void) -> i32;
    fn apx_tensor_math_mode(c: *mut c_void, tf32: i32) -> i32;
    fn apx_tensor_context_destroy(c: *mut c_void);
    fn apx_tensor_sync(c: *mut c_void) -> i32;
    fn apx_tensor_prepare(
        c: *mut c_void,
        s: *const Spec,
        a: *const f32,
        b: *const f32,
        bias: *const f32,
        y: *mut f32,
        out: *mut *mut c_void,
    ) -> i32;
    fn apx_tensor_enqueue(e: *mut c_void) -> i32;
    fn apx_tensor_tune(e: *mut c_void) -> i32;
    fn apx_tensor_rgb_normalize(c:*mut c_void,pixels:i32,x:*const u8,y:*mut f32,mean:*const f32,std:*const f32)->i32;
    fn apx_tensor_destroy(e: *mut c_void);
    fn apx_tensor_capture_begin(c: *mut c_void) -> i32;
    fn apx_tensor_capture_end(c: *mut c_void, out: *mut *mut c_void) -> i32;
    fn apx_tensor_graph_replay(c: *mut c_void, g: *mut c_void) -> i32;
    fn apx_tensor_graph_destroy(g: *mut c_void);
}
fn check(code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(unsafe { CStr::from_ptr(apx_tensor_error()) }
            .to_string_lossy()
            .into_owned())
    }
}
fn count(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() || shape.contains(&0) {
        return Err("empty tensor shape".into());
    }
    shape.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d)
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or("tensor size overflow".into())
    })
}
struct ContextInner {
    raw: *mut c_void,
    device: usize,
    bf16: Cell<bool>,
}
impl Drop for ContextInner {
    fn drop(&mut self) {
        unsafe {
            let _ = apx_tensor_sync(self.raw);
            apx_tensor_context_destroy(self.raw)
        }
    }
}
/// Per-operation precision constraint, fixed before capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvPrecision { ContextDefault, Bf16 }
#[derive(Clone)]
pub struct Context(Rc<ContextInner>);
impl Context {
    pub fn with_bf16(device: usize) -> Result<Self> {
        let ctx = Self::new(device)?;
        check(unsafe { apx_tensor_math_mode(ctx.0.raw, 2) })?;
        ctx.0.bf16.set(true);
        Ok(ctx)
    }
    /// Experimental E4M3 dynamic Conv1d with BF16 for all other operations.
    /// Forward 1D convolutions with aligned Cin/Cout >=128 use per-tensor scales.
    /// Currently qualified only for Thor sm110; this is not an accepted model variant.
    pub fn with_fp8_conv1d(device: usize) -> Result<Self> {
        let ctx = Self::new(device)?;
        check(unsafe { apx_tensor_math_mode(ctx.0.raw, 3) })?;
        ctx.0.bf16.set(true);
        Ok(ctx)
    }
    pub fn with_tf32(device: usize) -> Result<Self> {
        let ctx = Self::new(device)?;
        check(unsafe { apx_tensor_math_mode(ctx.0.raw, 1) })?;
        Ok(ctx)
    }
    pub fn new(device: usize) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        check(unsafe {
            apx_tensor_context_create(
                i32::try_from(device).map_err(|_| "device overflow")?,
                &mut raw,
            )
        })?;
        Ok(Self(Rc::new(ContextInner {
            raw,
            device,
            bf16: Cell::new(false),
        })))
    }
    pub fn is_bf16(&self) -> bool {
        self.0.bf16.get()
    }
    pub fn round_bf16(&self, x: &Tensor) -> Result<(Tensor, Operation)> {
        self.element(x, None, None, 8, 1, 1, &[])
    }
    pub fn synchronize(&self) -> Result<()> {
        check(unsafe { apx_tensor_sync(self.0.raw) })
    }
    pub fn zeros(&self, shape: &[usize]) -> Result<Tensor> {
        let n = count(shape)?;
        Ok(Tensor {
            buffer: CudaBuffer::alloc_zeros(n * 4, self.0.device)?,
            shape: shape.to_vec(),
            context: self.clone(),
            immutable: Rc::new(Cell::new(false)),
        })
    }
    pub fn tensor(&self, shape: &[usize], values: &[f32]) -> Result<Tensor> {
        let t = self.zeros(shape)?;
        t.write(values)?;
        t.immutable.set(true);
        Ok(t)
    }
    fn prepare(
        &self,
        kind: i32,
        params: &[usize],
        floats: &[f32],
        inputs: &[&Tensor],
        output: &Tensor,
    ) -> Result<Operation> {
        if inputs.is_empty() || inputs.len() > 3 || params.len() > 24 || floats.len() > 4 {
            return Err("invalid operator arity".into());
        }
        for t in inputs.iter().copied().chain(std::iter::once(output)) {
            if !Rc::ptr_eq(&t.context.0, &self.0) {
                return Err("operator tensors belong to different contexts".into());
            }
        }
        if output.immutable.get() {
            return Err("operator cannot write immutable tensor".into());
        }
        for input in inputs {
            let a = input.buffer.ptr() as usize;
            let y = output.buffer.ptr() as usize;
            if a < y + output.len() * 4
                && y < a + input.len() * 4
                && !(a == y
                    && input.len() == output.len()
                    && (kind == 11 || kind == 3 && params.get(1) == Some(&7)))
            {
                return Err("unsupported overlapping tensor views".into());
            }
        }
        let mut s = Spec {
            kind,
            ..Spec::default()
        };
        for (i, &v) in params.iter().enumerate() {
            s.p[i] = i32::try_from(v).map_err(|_| "operator dimension overflow")?;
        }
        s.f[..floats.len()].copy_from_slice(floats);
        let mut ptrs = [std::ptr::null(); 3];
        for (i, t) in inputs.iter().enumerate() {
            ptrs[i] = t.buffer.ptr().cast();
        }
        let mut raw = std::ptr::null_mut();
        check(unsafe {
            apx_tensor_prepare(
                self.0.raw,
                &s,
                ptrs[0],
                ptrs[1],
                ptrs[2],
                output.buffer.ptr().cast(),
                &mut raw,
            )
        })?;
        Ok(Operation(Rc::new(OperationInner {
            raw,
            tuned:Cell::new(false),captured:Cell::new(false),
            _context: self.clone(),
            _tensors: inputs
                .iter()
                .map(|t| (*t).clone())
                .chain(std::iter::once(output.clone()))
                .collect(),
        })))
    }
    pub fn linear(
        &self,
        x: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<(Tensor, Operation)> {
        if !w.immutable.get() {
            return Err("linear requires immutable weights".into());
        }
        if w.shape.len() != 2 || x.shape.last() != Some(&w.shape[1]) {
            return Err("linear shape mismatch".into());
        }
        let n = w.shape[0];
        let k = w.shape[1];
        let m = x.len() / k;
        if bias.is_some_and(|b| b.shape != [n]) {
            return Err("linear bias shape mismatch".into());
        }
        let mut shape = x.shape.clone();
        *shape.last_mut().unwrap() = n;
        let y = self.zeros(&shape)?;
        let mut args = vec![x, w];
        if let Some(b) = bias {
            args.push(b)
        }
        let op = self.prepare(0, &[m, n, k], &[], &args, &y)?;
        Ok((y, op))
    }
    /// Opt-in FP16 operands with FP32 GEMM accumulation, output and bias.
    /// Independent provider; does not change context precision or legacy dispatch.
    pub fn linear_fp16(&self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<(Tensor, Operation)> {
        self.linear_lowp(x,w,bias,15)
    }
    /// Three FP16-rounded high/low products, FP32 accumulation/output; small-low
    /// product omitted. Finite operands must fit FP16 range. Explicit opt-in.
    pub fn linear_f16x3(&self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<(Tensor, Operation)> {
        self.linear_lowp(x,w,bias,19)
    }
    /// Exact-erf GELU followed by three-product compensated Linear. GELU and
    /// hi/lo splitting share one kernel, retaining the standalone FP32 rounding
    /// order without an intermediate F32 activation tensor. The compensated
    /// products have the same approximation/range contract as linear_f16x3.
    pub fn linear_gelu_f16x3(&self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<(Tensor, Operation)> {
        self.linear_lowp(x,w,bias,27)
    }
    /// LayerNorm then compensated Linear; params=[linear bias, norm weight, norm bias].
    /// Immutable F32 params and matrix, row width <=1024; no intermediate F32 norm tensor.
    pub fn linear_norm_f16x3(&self,x:&Tensor,w:&Tensor,params:&Tensor,eps:f32)->Result<(Tensor,Operation)>{
        if !w.immutable.get() || !params.immutable.get() || w.shape.len()!=2 || x.shape.last()!=Some(&w.shape[1]) || !eps.is_finite() || eps<=0. {return Err("norm linear contract mismatch".into());}
        let(n,k)=(w.shape[0],w.shape[1]);if k>1024 || params.shape!=[n+2*k]{return Err("norm linear parameter shape mismatch".into());}
        let m=x.len()/k;let mut shape=x.shape.clone();*shape.last_mut().unwrap()=n;let y=self.zeros(&shape)?;
        let op=self.prepare(26,&[m,n,k],&[eps],&[x,w,params],&y)?;Ok((y,op))
    }
    /// Experimental E4M3 per-tensor scaled Linear, FP32 accumulation/output.
    pub fn linear_fp8(&self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<(Tensor, Operation)> {
        self.linear_lowp(x,w,bias,18)
    }
    fn linear_lowp(&self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>, kind: i32) -> Result<(Tensor, Operation)> {
        if !w.immutable.get() || w.shape.len()!=2 || x.shape.last()!=Some(&w.shape[1]) {
            return Err("FP16 linear requires immutable compatible weights".into());
        }
        let (n,k)=(w.shape[0],w.shape[1]);let m=x.len()/k;
        if kind==18 {let (mp,np,kp)=((m+15)/16*16,(n+15)/16*16,(k+15)/16*16);count(&[mp,kp])?;count(&[np,kp])?;count(&[mp,np])?;}
        if bias.is_some_and(|b|b.shape!=[n]) { return Err("FP16 linear bias mismatch".into()); }
        let mut shape=x.shape.clone();*shape.last_mut().unwrap()=n;
        let y=self.zeros(&shape)?;let mut args=vec![x,w];if let Some(b)=bias{args.push(b);}
        let op=self.prepare(kind,&[m,n,k],&[],&args,&y)?;Ok((y,op))
    }
    /// Opt-in NCL forward/transposed Conv1d through prepared FP16 GEMM lowering.
    pub fn conv1d_fp16(&self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>, stride: usize, pad: usize, transpose: bool) -> Result<(Tensor, Operation)> {
        self.conv1d_prepared(x,w,bias,stride,pad,transpose,if transpose{17}else{16})
    }
    /// Independent high/low three-product Conv1d; operands must fit FP16 range.
    pub fn conv1d_f16x3(&self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>, stride: usize, pad: usize, transpose: bool) -> Result<(Tensor, Operation)> {
        self.conv1d_prepared(x,w,bias,stride,pad,transpose,if transpose{21}else{20})
    }
    /// F32 transposed Conv1d with prepared weight packing and GEMM/col2im.
    pub fn conv_transpose1d_gemm(&self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>, stride: usize, pad: usize) -> Result<(Tensor, Operation)> {
        self.conv1d_prepared(x,w,bias,stride,pad,true,22)
    }
    fn conv1d_prepared(&self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>, stride: usize, pad: usize, transpose: bool, kind: i32) -> Result<(Tensor, Operation)> {
        if x.shape.len()!=3 || w.shape.len()!=3 || !w.immutable.get() || stride==0 {
            return Err("FP16 Conv1d shape/stride/immutable weights required".into());
        }
        let (batch,ci,length)=(x.shape[0],x.shape[1],x.shape[2]);let kernel=w.shape[2];
        let co=if transpose {w.shape[1]}else{w.shape[0]};
        if (if transpose {w.shape[0]}else{w.shape[1]})!=ci || bias.is_some_and(|b|b.shape!=[co]) {
            return Err("FP16 Conv1d channel/bias mismatch".into());
        }
        let padding=pad.checked_mul(2).ok_or("padding overflow")?;
        let out=if transpose {
            (length-1).checked_mul(stride).and_then(|v|v.checked_add(kernel)).and_then(|v|v.checked_sub(padding)).ok_or("transpose output overflow")?
        }else{
            length.checked_add(padding).and_then(|v|v.checked_sub(kernel)).ok_or("convolution kernel exceeds input")?/stride+1
        };
        count(&[ci,co,kernel])?;count(&[ci,kernel,out])?;count(&[co,kernel,length])?;
        let y=self.zeros(&[batch,co,out])?;let mut args=vec![x,w];if let Some(b)=bias{args.push(b);}
        let op=self.prepare(kind,&[batch,ci,length,co,kernel,out,stride,pad],&[],&args,&y)?;Ok((y,op))
    }
    pub fn conv2d(
        &self,
        x: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
        stride: [usize; 2],
        pad: [usize; 2],
        transpose: bool,
    ) -> Result<(Tensor, Operation)> {
        self.conv2d_with_precision(x, w, bias, stride, pad, transpose, ConvPrecision::ContextDefault)
    }
    /// Keep a sensitive convolution in BF16 within a mixed FP8 context.
    /// This never changes context precision or dispatch during graph replay.
    pub fn conv2d_with_precision(
        &self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>,
        stride: [usize; 2], pad: [usize; 2], transpose: bool, precision: ConvPrecision,
    ) -> Result<(Tensor, Operation)> {
        self.conv2d_impl(x, w, bias, stride, pad, transpose, precision, false)
    }
    /// Explicit prepared FP32 im2col + GEMM for [B,C,1,L] Conv1d.
    /// Reuses scratch, rejects TF32/BF16 and never changes another operation's provider.
    pub fn conv1d_im2col(
        &self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>, stride: usize, pad: usize,
    ) -> Result<(Tensor, Operation)> {
        if x.shape.len()!=4 || w.shape.len()!=4 || x.shape[2]!=1 || w.shape[2]!=1 {
            return Err("im2col Conv1d requires [B,C,1,L] and [Cout,Cin,1,K]".into());
        }
        self.conv2d_impl(x,w,bias,[1,stride],[0,pad],false,ConvPrecision::ContextDefault,true)
    }
    fn conv2d_impl(
        &self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>,
        stride: [usize; 2], pad: [usize; 2], transpose: bool, precision: ConvPrecision, im2col: bool,
    ) -> Result<(Tensor, Operation)> {
        if precision == ConvPrecision::Bf16 && !self.is_bf16() {
            return Err("BF16 convolution override requires a BF16 or mixed FP8 context".into());
        }
        if !w.immutable.get() {
            return Err("convolution requires immutable weights".into());
        }
        if x.shape.len() != 4 || w.shape.len() != 4 || stride.contains(&0) {
            return Err("convolution requires NCHW and OIHW/IOHW weights".into());
        }
        let (n, ci, h, ww) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        let co = if transpose { w.shape[1] } else { w.shape[0] };
        if ci != w.shape[if transpose { 0 } else { 1 }] {
            return Err("convolution channels mismatch".into());
        }
        if bias.is_some_and(|b| b.shape != [co]) {
            return Err("convolution bias mismatch".into());
        }
        let (kh, kw) = (w.shape[2], w.shape[3]);
        let size = |v: usize, k: usize, s: usize, p: usize| -> Result<usize> {
            let z = if transpose {
                (v as i64 - 1) * s as i64 - 2 * p as i64 + k as i64
            } else {
                let q = v as i64 + 2 * p as i64 - k as i64;
                if q < 0 {
                    return Err("invalid convolution extent".into());
                }
                q / s as i64 + 1
            };
            if z <= 0 || z > i32::MAX as i64 {
                Err("invalid convolution extent".into())
            } else {
                Ok(z as usize)
            }
        };
        let oh = size(h, kh, stride[0], pad[0])?;
        let ow = size(ww, kw, stride[1], pad[1])?;
        let y = self.zeros(&[n, co, oh, ow])?;
        let mut a = vec![x, w];
        if let Some(b) = bias {
            a.push(b)
        }
        let op = self.prepare(
            if transpose { 2 } else { 1 },
            &[
                n, ci, h, ww, co, kh, kw, 0, oh, ow, pad[0], pad[1], stride[0], stride[1], usize::from(precision == ConvPrecision::Bf16), usize::from(im2col),
            ],
            &[],
            &a,
            &y,
        )?;
        Ok((y, op))
    }
    /// Eval BatchNorm parameters are [gamma, beta, running mean, running variance].
    pub fn batch_norm(&self,x:&Tensor,parameters:&Tensor,eps:f32)->Result<(Tensor,Operation)>{
        if x.shape.len()!=4 || parameters.shape!=[4,x.shape[1]] || !eps.is_finite() || eps<=0. {return Err("batch norm shape/epsilon mismatch".into())}
        let y=self.zeros(&x.shape)?;
        let op=self.prepare(12,&[x.len(),x.shape[1],x.shape[2]*x.shape[3],usize::from(self.is_bf16())],&[eps],&[x,parameters],&y)?;
        Ok((y,op))
    }
    /// Frozen affine normalization with FP32 parameters and FP32 output.
    /// Preserve separate reciprocal-sqrt, scale, bias and elementwise rounding.
    pub fn frozen_batch_norm(&self,x:&Tensor,parameters:&Tensor,eps:f32)->Result<(Tensor,Operation)>{
        if x.shape.len()!=4 || parameters.shape!=[4,x.shape[1]] || !eps.is_finite() || eps<=0. {return Err("frozen batch norm shape/epsilon mismatch".into())}
        let y=self.zeros(&x.shape)?;
        let op=self.prepare(12,&[x.len(),x.shape[1],x.shape[2]*x.shape[3],0,1],&[eps],&[x,parameters],&y)?;
        Ok((y,op))
    }
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Result<(Tensor, Operation)> {
        if a.shape != b.shape {
            return Err("add requires equal shapes".into());
        }
        self.element(a, Some(b), None, 0, 1, 1, &[])
    }
    pub fn activation(&self, x: &Tensor, activation: Activation) -> Result<(Tensor, Operation)> {
        self.element(x, None, None, activation as usize, 1, 1, &[])
    }
    pub fn affine(
        &self,
        x: &Tensor,
        scale: &Tensor,
        bias: &Tensor,
        inner: usize,
    ) -> Result<(Tensor, Operation)> {
        if inner == 0 || scale.shape != bias.shape || x.len() % (scale.len() * inner) != 0 {
            return Err("affine broadcasting mismatch".into());
        }
        self.element(x, Some(scale), Some(bias), 4, inner, scale.len(), &[])
    }
    pub fn scale(&self, x: &Tensor, scale: f32, bias: f32) -> Result<(Tensor, Operation)> {
        self.element(x, None, None, 5, 1, 1, &[scale, bias])
    }
    fn element(
        &self,
        x: &Tensor,
        b: Option<&Tensor>,
        c: Option<&Tensor>,
        mode: usize,
        inner: usize,
        channels: usize,
        f: &[f32],
    ) -> Result<(Tensor, Operation)> {
        let y = self.zeros(&x.shape)?;
        let mut args = vec![x];
        if let Some(b) = b {
            args.push(b)
        }
        if let Some(c) = c {
            args.push(c)
        }
        let op = self.prepare(3, &[x.len(), mode, inner, channels], f, &args, &y)?;
        Ok((y, op))
    }
    pub fn copy_into(&self, x: &Tensor, y: &Tensor) -> Result<Operation> {
        if x.len() != y.len() {
            return Err("copy length mismatch".into());
        }
        self.prepare(3, &[x.len(), 7, 1, 1], &[], &[x], y)
    }
    /// Normalizes contiguous groups; affine channels repeat every `spatial` elements.
    pub fn norm(
        &self,
        x: &Tensor,
        w: &Tensor,
        b: &Tensor,
        group_width: usize,
        spatial: usize,
        eps: f32,
    ) -> Result<(Tensor, Operation)> {
        if group_width == 0
            || spatial == 0
            || x.len() % group_width != 0
            || w.shape != b.shape
            || x.len() % (w.len() * spatial) != 0
            || !eps.is_finite()
            || eps <= 0.
        {
            return Err("normalization shape/epsilon mismatch".into());
        }
        let y = self.zeros(&x.shape)?;
        let op = self.prepare(
            4,
            &[x.len() / group_width, group_width, w.len(), spatial],
            &[eps],
            &[x, w, b],
            &y,
        )?;
        Ok((y, op))
    }
    pub fn softmax(&self, x: &Tensor, scale: f32) -> Result<(Tensor, Operation)> {
        let width = *x.shape.last().unwrap();
        let y = self.zeros(&x.shape)?;
        let op = self.prepare(5, &[x.len() / width, width], &[scale], &[x], &y)?;
        Ok((y, op))
    }
    /// Independent block-reduction provider for last-axis LayerNorm.
    /// Legacy `norm` dispatch and kernels are not changed by this opt-in API.
    pub fn layer_norm_block(
        &self, x: &Tensor, w: &Tensor, b: &Tensor, eps: f32,
    ) -> Result<(Tensor, Operation)> {
        let width = *x.shape.last().unwrap();
        if w.shape != [width] || b.shape != [width] || !eps.is_finite() || eps <= 0. {
            return Err("block layer normalization shape/epsilon mismatch".into());
        }
        let y = self.zeros(&x.shape)?;
        let op = self.prepare(13, &[x.len() / width, width], &[eps], &[x, w, b], &y)?;
        Ok((y, op))
    }
    /// Independent, capture-safe last-axis Softmax with synchronized broadcasts.
    pub fn softmax_block(&self, x: &Tensor, scale: f32) -> Result<(Tensor, Operation)> {
        if !scale.is_finite() { return Err("softmax scale must be finite".into()); }
        let width = *x.shape.last().unwrap();
        let y = self.zeros(&x.shape)?;
        let op = self.prepare(14, &[x.len() / width, width], &[scale], &[x], &y)?;
        Ok((y, op))
    }
    pub fn permute(&self, x: &Tensor, axes: &[usize]) -> Result<(Tensor, Operation)> {
        let rank = x.shape.len();
        if rank > 4 || rank != axes.len() {
            return Err("permute rank 1..4 required".into());
        }
        let mut sorted = axes.to_vec();
        sorted.sort_unstable();
        if sorted != (0..rank).collect::<Vec<_>>() {
            return Err("invalid permutation".into());
        }
        let shape = axes.iter().map(|&i| x.shape[i]).collect::<Vec<_>>();
        let mut strides = vec![1; rank];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * x.shape[i + 1];
        }
        let mut dims = [1; 4];
        let mut step = [0; 4];
        for j in 0..rank {
            dims[4 - rank + j] = shape[j];
            step[4 - rank + j] = strides[axes[j]];
        }
        let mut p = vec![x.len()];
        p.extend(dims);
        p.extend(step);
        let y = self.zeros(&shape)?;
        let op = self.prepare(6, &p, &[], &[x], &y)?;
        Ok((y, op))
    }
    pub fn concat(&self, a: &Tensor, b: &Tensor, axis: usize) -> Result<(Tensor, Operation)> {
        if a.shape.len() != b.shape.len()
            || axis >= a.shape.len()
            || a.shape
                .iter()
                .zip(&b.shape)
                .enumerate()
                .any(|(i, (a, b))| i != axis && a != b)
        {
            return Err("concat shape mismatch".into());
        }
        let mut shape = a.shape.clone();
        shape[axis] = shape[axis]
            .checked_add(b.shape[axis])
            .ok_or("concat overflow")?;
        let inner = a.shape[axis + 1..].iter().product();
        let y = self.zeros(&shape)?;
        let op = self.prepare(
            7,
            &[y.len(), a.shape[axis], b.shape[axis], inner],
            &[],
            &[a, b],
            &y,
        )?;
        Ok((y, op))
    }
    pub fn slice(
        &self,
        x: &Tensor,
        axis: usize,
        start: usize,
        length: usize,
    ) -> Result<(Tensor, Operation)> {
        if axis >= x.shape.len()
            || length == 0
            || start.checked_add(length).is_none_or(|e| e > x.shape[axis])
        {
            return Err("slice bounds".into());
        }
        let mut shape = x.shape.clone();
        shape[axis] = length;
        let y = self.zeros(&shape)?;
        let inner = x.shape[axis + 1..].iter().product();
        let op = self.prepare(
            8,
            &[y.len(), x.shape[axis], length, start, inner],
            &[],
            &[x],
            &y,
        )?;
        Ok((y, op))
    }
    pub fn maxpool2d(
        &self,
        x: &Tensor,
        k: usize,
        stride: usize,
        pad: usize,
    ) -> Result<(Tensor, Operation)> {
        if x.shape.len() != 4
            || stride == 0
            || k == 0
            || x.shape[2] + 2 * pad < k
            || x.shape[3] + 2 * pad < k
        {
            return Err("pool shape mismatch".into());
        }
        let oh = (x.shape[2] + 2 * pad - k) / stride + 1;
        let ow = (x.shape[3] + 2 * pad - k) / stride + 1;
        let y = self.zeros(&[x.shape[0], x.shape[1], oh, ow])?;
        let op = self.prepare(
            9,
            &[
                y.len(),
                x.shape[1],
                x.shape[2],
                x.shape[3],
                oh,
                ow,
                k,
                stride,
                pad,
            ],
            &[],
            &[x],
            &y,
        )?;
        Ok((y, op))
    }
    /// Explicit F32 row LayerNorm, widths 1..=1024, one warp per row.
    pub fn layer_norm_warp(&self,x:&Tensor,w:&Tensor,b:&Tensor,eps:f32)->Result<(Tensor,Operation)> {
        let width=*x.shape.last().ok_or("empty norm shape")?;
        if width>1024 || w.shape!=[width] || b.shape!=[width] || !eps.is_finite() || eps<=0. {return Err("warp norm shape/epsilon mismatch".into());}
        let y=self.zeros(&x.shape)?;let op=self.prepare(24,&[x.len()/width,width],&[eps],&[x,w,b],&y)?;Ok((y,op))
    }
    /// x + branch * per-last-axis gamma with separate FP32 multiply/add rounding.
    pub fn scaled_residual(&self,x:&Tensor,branch:&Tensor,gamma:&Tensor)->Result<(Tensor,Operation)> {
        let width=*x.shape.last().ok_or("empty residual shape")?;
        if x.shape!=branch.shape || gamma.shape!=[width] {return Err("scaled residual shape mismatch".into());}
        let y=self.zeros(&x.shape)?;let op=self.prepare(25,&[x.len(),width],&[],&[x,branch,gamma],&y)?;Ok((y,op))
    }
    /// F32 packed QKV [B,S,3*H*64] -> [B,S,H*64], scale 1/sqrt(64).
    /// Unmasked self-attention, no dropout; high-accuracy three-TF32 products.
    pub fn attention_qkv_f32(&self,qkv:&Tensor,heads:usize)->Result<(Tensor,Operation)> {
        if heads==0 || qkv.shape.len()!=3 || heads.checked_mul(192)!=qkv.shape.last().copied() {
            return Err("F32 fused attention requires packed QKV with head_dim=64".into());
        }
        let (batch,seq)=(qkv.shape[0],qkv.shape[1]);
        if batch>65535 || heads>65535 {return Err("F32 attention CUDA grid exceeds limit".into());}
        let y=self.zeros(&[batch,seq,heads*64])?;
        let op=self.prepare(23,&[batch,seq,heads],&[],&[qkv],&y)?;Ok((y,op))
    }
    pub fn bmm(
        &self,
        a: &Tensor,
        b: &Tensor,
        transpose_b: bool,
        scale: f32,
    ) -> Result<(Tensor, Operation)> {
        if a.shape.len() != 3
            || b.shape.len() != 3
            || a.shape[0] != b.shape[0]
            || a.shape[2] != b.shape[if transpose_b { 2 } else { 1 }]
        {
            return Err("batched matmul shape mismatch".into());
        }
        let (batch, m, k) = (a.shape[0], a.shape[1], a.shape[2]);
        let n = b.shape[if transpose_b { 1 } else { 2 }];
        let y = self.zeros(&[batch, m, n])?;
        let op = self.prepare(
            10,
            &[batch, m, n, k, usize::from(transpose_b)],
            &[scale],
            &[a, b],
            &y,
        )?;
        Ok((y, op))
    }
    pub fn axpby_into(
        &self,
        a: &Tensor,
        b: &Tensor,
        c: Option<&Tensor>,
        y: &Tensor,
        coeff: [f32; 3],
        clip: f32,
    ) -> Result<Operation> {
        if a.shape != b.shape || a.shape != y.shape || c.is_some_and(|c| c.shape != a.shape) {
            return Err("linear combination shape mismatch".into());
        }
        let mut args = vec![a, b];
        if let Some(c) = c {
            args.push(c)
        }
        self.prepare(
            11,
            &[y.len()],
            &[coeff[0], coeff[1], coeff[2], clip],
            &args,
            y,
        )
    }
    pub fn capture(&self, operations: &[Operation]) -> Result<Graph> {
        if operations.is_empty()
            || operations
                .iter()
                .any(|op| !Rc::ptr_eq(&op.0._context.0, &self.0))
        {
            return Err("capture requires nonempty operations from this context".into());
        }
        for op in operations{op.0.captured.set(true);}
        self.synchronize()?;
        check(unsafe { apx_tensor_capture_begin(self.0.raw) })?;
        let run = operations.iter().try_for_each(Operation::run);
        let mut raw = std::ptr::null_mut();
        let end = check(unsafe { apx_tensor_capture_end(self.0.raw, &mut raw) });
        if let Err(e) = run.and(end) {
            if !raw.is_null() {
                unsafe { apx_tensor_graph_destroy(raw) }
            }
            return Err(e);
        }
        Ok(Graph {
            raw,
            context: self.clone(),
            _operations: operations.to_vec(),
        })
    }
}
#[derive(Clone, Copy)]
pub enum Activation {
    Relu = 1,
    Mish = 2,
    Silu = 3,
    /// Exact erf GELU (PyTorch approximate="none"). FP32 input/output.
    Gelu = 9,
}
#[derive(Clone)]
pub struct Tensor {
    buffer: CudaBuffer,
    shape: Vec<usize>,
    context: Context,
    immutable: Rc<Cell<bool>>,
}
impl Tensor {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn len(&self) -> usize {
        self.buffer.len() / 4
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        if count(shape)? != self.len() {
            return Err("reshape count mismatch".into());
        }
        Ok(Self {
            shape: shape.to_vec(),
            ..self.clone()
        })
    }
    pub fn view(&self, offset: usize, shape: &[usize]) -> Result<Self> {
        let n = count(shape)?;
        Ok(Self {
            buffer: self
                .buffer
                .view(offset.checked_mul(4).ok_or("offset overflow")?, n * 4)?,
            shape: shape.to_vec(),
            context: self.context.clone(),
            immutable: self.immutable.clone(),
        })
    }
    pub fn write(&self, values: &[f32]) -> Result<()> {
        if self.immutable.get() {
            return Err("cannot overwrite immutable tensor".into());
        }
        if values.len() != self.len() {
            return Err("upload count mismatch".into());
        }
        self.context.synchronize()?;
        let bytes =
            unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) };
        self.buffer.copy_from_host(bytes)
    }
    pub fn read(&self) -> Result<Vec<f32>> {
        self.context.synchronize()?;
        let mut out = vec![0f32; self.len()];
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out.len() * 4) };
        self.buffer.copy_to_host(bytes)?;
        Ok(out)
    }

}
struct OperationInner {
    raw: *mut c_void,
    tuned:Cell<bool>,captured:Cell<bool>,
    _context: Context,
    _tensors: Vec<Tensor>,
}
impl Drop for OperationInner {
    fn drop(&mut self) {
        let _ = self._context.synchronize();
        unsafe { apx_tensor_destroy(self.raw) }
    }
}
#[derive(Clone)]
pub struct Operation(Rc<OperationInner>);
impl Operation {
    pub fn tune(&self)->Result<()>{
        if self.0.tuned.get(){return Ok(())}
        if self.0.captured.get(){return Err("cannot retune an operation referenced by a graph".into())}
        check(unsafe{apx_tensor_tune(self.0.raw)})?;self.0.tuned.set(true);Ok(())
    }
    pub fn run(&self) -> Result<()> {
        check(unsafe { apx_tensor_enqueue(self.0.raw) })
    }
}
pub struct Graph {
    raw: *mut c_void,
    context: Context,
    _operations: Vec<Operation>,
}
impl Graph {
    pub fn replay(&self) -> Result<()> {
        check(unsafe { apx_tensor_graph_replay(self.context.0.raw, self.raw) })
    }
}
impl Drop for Graph {
    fn drop(&mut self) {
        let _ = self.context.synchronize();
        unsafe { apx_tensor_graph_destroy(self.raw) }
    }
}

/// Prepared RGB upload storage; source bytes and normalized output have explicit
/// distinct types. Output is written on the same stream used by the model graph.
pub struct RgbProcessor { input:CudaBuffer, output:Tensor, context:Context }
impl Context {
    pub fn rgb_processor(&self,output:&Tensor)->Result<RgbProcessor>{
        if output.shape.len()!=4 || output.shape[0]!=1 || output.shape[1]!=3 || output.immutable.get() || !Rc::ptr_eq(&self.0,&output.context.0){return Err("RGB processor requires mutable N1C3HW output in this context".into())}
        Ok(RgbProcessor{input:CudaBuffer::alloc_zeros(output.len(),self.0.device)?,output:output.clone(),context:self.clone()})
    }
}
impl RgbProcessor {
    pub fn write(&self,bytes:&[u8],mean:[f32;3],std:[f32;3])->Result<()>{
        if bytes.len()!=self.input.len() || mean.iter().any(|v|!v.is_finite()) || std.iter().any(|v|!v.is_finite()||*v<=0.) {return Err("invalid RGB input/normalization".into())}
        self.context.synchronize()?;self.input.copy_from_host(bytes)?;
        check(unsafe{apx_tensor_rgb_normalize(self.context.0.raw,(bytes.len()/3)as i32,self.input.ptr().cast(),self.output.buffer.ptr().cast(),mean.as_ptr(),std.as_ptr())})
    }
}

/// Batched HWC RGB upload with symmetric horizontal edge padding and FP32 normalization.
/// Separate prepared provider; existing single-image RGB behavior is unchanged.
pub struct RgbBatchProcessor {input:CudaBuffer,output:Tensor,context:Context,width:usize}

/// Borrowed, producer-complete RGB allocations. Ownership stays with the caller.
pub struct ReadyDeviceRgb<'a> { frames: &'a [usize], frame_bytes: usize }
impl<'a> ReadyDeviceRgb<'a> {
    /// # Safety
    /// Every address must remain allocated and immutable through the synchronous
    /// request; the producer must have completed its writes before construction.
    /// The backend additionally checks CUDA device, allocation bounds and type.
    pub unsafe fn from_raw_frames(frames: &'a [usize], frame_bytes: usize) -> Self {
        Self { frames, frame_bytes }
    }
}
impl Context {
    pub fn rgb_batch_processor(&self,output:&Tensor,width:usize)->Result<RgbBatchProcessor>{
        if output.shape.len()!=4 || output.shape[1]!=3 || width==0 || width>output.shape[3]
            || (output.shape[3]-width)%2!=0 || output.immutable.get() || !Rc::ptr_eq(&self.0,&output.context.0){
            return Err("RGB batch requires mutable NCHW with symmetric horizontal padding".into());
        }
        let size=count(&[output.shape[0],output.shape[2],width,3])?;
        Ok(RgbBatchProcessor{input:CudaBuffer::alloc_zeros(size,self.0.device)?,output:output.clone(),context:self.clone(),width})
    }
}
impl RgbBatchProcessor {
    pub fn write_device(&self, source:&ReadyDeviceRgb<'_>, mean:[f32;3], std:[f32;3])->Result<()> {
        let s=&self.output.shape;
        if source.frames.len()!=s[0] || source.frame_bytes!=s[2]*self.width*3
            || mean.iter().any(|v|!v.is_finite()) || std.iter().any(|v|!v.is_finite()||*v<=0.) {
            return Err("device RGB batch input/normalization mismatch".into());
        }
        check(unsafe {apx_tensor_rgb_batch_device(self.context.0.raw,s[0]as i32,s[2]as i32,
            self.width as i32,s[3]as i32,source.frames.as_ptr(),self.input.ptr().cast(),
            self.output.buffer.ptr().cast(),mean.as_ptr(),std.as_ptr())})
    }
    pub fn write(&self,bytes:&[u8],mean:[f32;3],std:[f32;3])->Result<()>{
        if bytes.len()!=self.input.len() || mean.iter().any(|v|!v.is_finite()) || std.iter().any(|v|!v.is_finite()||*v<=0.){return Err("RGB batch input/normalization mismatch".into());}
        self.context.synchronize()?;self.input.copy_from_host(bytes)?;let s=&self.output.shape;
        check(unsafe{apx_tensor_rgb_batch(self.context.0.raw,s[0]as i32,s[2]as i32,self.width as i32,s[3]as i32,self.input.ptr().cast(),self.output.buffer.ptr().cast(),mean.as_ptr(),std.as_ptr())})
    }
}
