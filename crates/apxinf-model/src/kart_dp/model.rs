//! Native fixed-profile math. Static assets are derived at export with provenance.
use super::weights::{Result, Weights};
use apxinf_cuda_next::tensor_ops::{Activation, Context, Operation, Tensor};
pub(super) struct Model {
    pub ctx: Context,
    pub image: Tensor,
    pub state: Tensor,
    pub noise: Tensor,
    pub output: Tensor,
    pub operations: Vec<Operation>,
    pub diagnostics: Vec<(String, Tensor)>,
}
struct Builder<'a> {
    ctx: Context,
    w: &'a mut Weights,
    ops: Vec<Operation>,
    diagnostics: Vec<(String, Tensor)>,
}
impl Builder<'_> {
    fn take(&mut self, r: Result<(Tensor, Operation)>) -> Result<Tensor> {
        let (t, o) = r?;
        self.ops.push(o);
        Ok(t)
    }
    fn record(&mut self, n: &str, t: &Tensor) {
        self.diagnostics.push((n.into(), t.clone()));
    }
    fn linear(&mut self, x: &Tensor, n: &str) -> Result<Tensor> {
        let w = self.w.get(&format!("{n}.weight"))?;
        let b = self.w.get(&format!("{n}.bias"))?;
        self.take(self.ctx.linear(x, &w, Some(&b)))
    }
    fn act(&mut self, x: &Tensor, a: Activation) -> Result<Tensor> {
        self.take(self.ctx.activation(x, a))
    }
    fn add(&mut self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        self.take(self.ctx.add(x, y))
    }
    fn norm(&mut self, x: &Tensor, n: &str, groups: Option<usize>) -> Result<Tensor> {
        let w = self.w.get(&format!("{n}.weight"))?;
        let b = self.w.get(&format!("{n}.bias"))?;
        let (width, spatial, eps) = if let Some(g) = groups {
            (x.shape()[1] * x.shape()[2] / g, x.shape()[2], 1e-5)
        } else {
            (384, 1, 1e-6)
        };
        self.take(self.ctx.norm(x, &w, &b, width, spatial, eps))
    }
    fn conv(
        &mut self,
        x: &Tensor,
        n: &str,
        stride: [usize; 2],
        pad: [usize; 2],
        transpose: bool,
    ) -> Result<Tensor> {
        let w = self.w.get(&format!("{n}.weight"))?;
        let b = self.w.get(&format!("{n}.bias"))?;
        self.take(self.ctx.conv2d(x, &w, Some(&b), stride, pad, transpose))
    }
    fn conv1(
        &mut self,
        x: &Tensor,
        n: &str,
        stride: usize,
        pad: usize,
        transpose: bool,
    ) -> Result<Tensor> {
        let w = self.w.get(&format!("{n}.weight"))?;
        let s = w.shape();
        let w = w.reshape(&[s[0], s[1], 1, s[2]])?;
        let bias = self.w.get(&format!("{n}.bias"))?;
        let s = x.shape();
        let x = x.reshape(&[s[0], s[1], 1, s[2]])?;
        // Thor profiling: forward temporal convolutions dominate the fixed profile.
        // Select prepared FP32 GEMM locally; transposed/vision convolutions retain cuDNN.
        let operation = if transpose {
            self.ctx.conv2d(&x,&w,Some(&bias),[1,stride],[0,pad],true)
        } else {
            self.ctx.conv1d_im2col(&x,&w,Some(&bias),stride,pad)
        };
        let y = self.take(operation)?;
        y.reshape(&[y.shape()[0], y.shape()[1], y.shape()[3]])
    }
    fn vision(&mut self, image: &Tensor) -> Result<Tensor> {
        let ctx = self.ctx.clone();
        let p = "vision.backbone";
        let x = self.conv(
            image,
            &format!("{p}.patch_embed.proj"),
            [14, 14],
            [0, 0],
            false,
        )?;
        let x = x.reshape(&[4, 384, 336])?;
        let x = self.take(ctx.permute(&x, &[0, 2, 1]))?;
        let cls = self.w.get("native.cls_tokens")?;
        let mut x = self.take(ctx.concat(&cls, &x, 1))?;
        let pos = self.w.get("native.position")?;
        x = self.add(&x, &pos)?;
        self.record("vision.tokens", &x);
        let zeros = ctx.tensor(&[384], &vec![0.; 384])?;
        for i in 0..12 {
            let n = format!("{p}.blocks.{i}");
            let y = self.norm(&x, &format!("{n}.norm1"), None)?;
            let qkv = self.linear(&y, &format!("{n}.attn.qkv"))?;
            let mut pieces = Vec::new();
            for j in 0..3 {
                let q = self
                    .take(ctx.slice(&qkv, 2, j * 384, 384))?
                    .reshape(&[4, 337, 6, 64])?;
                pieces.push(
                    self.take(ctx.permute(&q, &[0, 2, 1, 3]))?
                        .reshape(&[24, 337, 64])?,
                );
            }
            let scores = self.take(ctx.bmm(&pieces[0], &pieces[1], true, 0.125))?;
            let probs = self.take(ctx.softmax(&scores, 1.))?;
            let y = self
                .take(ctx.bmm(&probs, &pieces[2], false, 1.))?
                .reshape(&[4, 6, 337, 64])?;
            let y = self
                .take(ctx.permute(&y, &[0, 2, 1, 3]))?
                .reshape(&[4, 337, 384])?;
            let y = self.linear(&y, &format!("{n}.attn.proj"))?;
            let gamma = self.w.get(&format!("{n}.ls1.gamma"))?;
            let y = self.take(ctx.affine(&y, &gamma, &zeros, 1))?;
            x = self.add(&x, &y)?;
            let y = self.norm(&x, &format!("{n}.norm2"), None)?;
            let y = self.linear(&y, &format!("{n}.mlp.fc1"))?;
            let y = self.act(&y, Activation::Gelu)?;
            let y = self.linear(&y, &format!("{n}.mlp.fc2"))?;
            let gamma = self.w.get(&format!("{n}.ls2.gamma"))?;
            let y = self.take(ctx.affine(&y, &gamma, &zeros, 1))?;
            x = self.add(&x, &y)?;
            self.record(&format!("vision.block.{i}"), &x);
        }
        x = self.norm(&x, &format!("{p}.norm"), None)?;
        x = self.take(ctx.slice(&x, 1, 1, 336))?;
        x = self
            .take(ctx.permute(&x, &[0, 2, 1]))?
            .reshape(&[4, 384, 16, 21])?;
        self.record("vision.patch_features", &x);
        x = self
            .conv(&x, "vision.proj", [1, 1], [0, 0], false)?
            .reshape(&[4, 384, 336])?;
        // Adaptive 4x4 spatial means represented by a fixed sparse linear map.
        let mut pool = vec![0f32; 16 * 336];
        for iy in 0..4 {
            for ix in 0..4 {
                let (y0, y1, x0, x1) = (
                    iy * 16 / 4,
                    ((iy + 1) * 16 + 3) / 4,
                    ix * 21 / 4,
                    ((ix + 1) * 21 + 3) / 4,
                );
                for y in y0..y1 {
                    for x in x0..x1 {
                        pool[(iy * 4 + ix) * 336 + y * 21 + x] =
                            1. / ((y1 - y0) * (x1 - x0)) as f32;
                    }
                }
            }
        }
        let pw = ctx.tensor(&[16, 336], &pool)?;
        x = self.take(ctx.linear(&x, &pw, None))?;
        x = self
            .take(ctx.permute(&x, &[0, 2, 1]))?
            .reshape(&[1, 4 * 16 * 384])?;
        self.record("vision.output", &x);
        Ok(x)
    }
    fn residual(&mut self, x: &Tensor, condition_silu: &Tensor, n: &str) -> Result<Tensor> {
        let ctx = self.ctx.clone();
        let film = self.linear(condition_silu, &format!("{n}.film.1"))?;
        let channels = film.len() / 2;
        let scale = film.view(0, &[channels])?;
        let scale = self.take(ctx.scale(&scale, 1., 1.))?;
        let bias = film.view(channels, &[channels])?;
        let y = self.conv1(x, &format!("{n}.conv1"), 1, 1, false)?;
        let y = self.norm(&y, &format!("{n}.norm1"), Some(8))?;
        let y = self.take(ctx.affine(&y, &scale, &bias, y.shape()[2]))?;
        let y = self.act(&y, Activation::Silu)?;
        let y = self.conv1(&y, &format!("{n}.conv2"), 1, 1, false)?;
        let y = self.norm(&y, &format!("{n}.norm2"), Some(8))?;
        let y = self.act(&y, Activation::Silu)?;
        let skip = if self.w.has(&format!("{n}.skip.weight")) {
            self.conv1(x, &format!("{n}.skip"), 1, 0, false)?
        } else {
            x.clone()
        };
        self.add(&skip, &y)
    }
    fn denoise(&mut self, sample: &Tensor, obs: &Tensor, time: &Tensor) -> Result<Tensor> {
        let ctx = self.ctx.clone();
        let cond = self.take(ctx.concat(obs, time, 1))?;
        let cond = self.act(&cond, Activation::Silu)?;
        let x = self.take(ctx.permute(sample, &[0, 2, 1]))?;
        let d1 = self.residual(&x, &cond, "down1")?;
        let x = self.conv1(&d1, "downsample1", 2, 1, false)?;
        let d2 = self.residual(&x, &cond, "down2")?;
        let x = self.conv1(&d2, "downsample2", 2, 1, false)?;
        let x = self.residual(&x, &cond, "mid")?;
        let x = self.conv1(&x, "upsample2", 2, 1, true)?;
        let x = self.take(ctx.concat(&x, &d2, 1))?;
        let x = self.residual(&x, &cond, "up2")?;
        let x = self.conv1(&x, "upsample1", 2, 1, true)?;
        let x = self.take(ctx.concat(&x, &d1, 1))?;
        let x = self.residual(&x, &cond, "up1")?;
        let x = self.conv1(&x, "head", 1, 0, false)?;
        self.take(ctx.permute(&x, &[0, 2, 1]))
    }
}
impl Model {
    pub fn build(ctx: Context, w: &mut Weights) -> Result<Self> {
        let image = ctx.zeros(&[4, 3, 224, 294])?;
        let state = ctx.zeros(&[1, 248])?;
        let noise = ctx.zeros(&[1, 24, 3])?;
        let mut b = Builder {
            ctx: ctx.clone(),
            w,
            ops: vec![],
            diagnostics: vec![],
        };
        let vision = b.vision(&image)?;
        let obs = b.take(ctx.concat(&state, &vision, 1))?;
        let obs = b.linear(&obs, "obs_proj.0")?;
        let obs = b.act(&obs, Activation::Silu)?;
        b.record("condition", &obs);
        let time_inputs = b.w.get("native.time_inputs")?;
        let coeff = b.w.values("native.ddim_coefficients")?;
        if coeff.len() != 40 || time_inputs.shape() != [10, 384] {
            return Err("invalid DDIM assets".into());
        }
        let mut sample = noise.clone();
        for i in 0..10 {
            let ti = time_inputs.view(i * 384, &[1, 384])?;
            let time = b.linear(&ti, "time.1")?;
            let time = b.act(&time, Activation::Silu)?;
            let time = b.linear(&time, "time.3")?;
            let eps = b.denoise(&sample, &obs, &time)?;
            b.record(&format!("epsilon.{i}"), &eps);
            // Preserve the DDIM reference operation boundaries before clipping x0.
            let eps_scaled = b.take(ctx.scale(&eps, coeff[i * 4], 0.))?;
            let delta = ctx.zeros(&[1, 24, 3])?;
            b.ops
                .push(ctx.axpby_into(&sample, &eps_scaled, None, &delta, [1., -1., 0.], 0.)?);
            let x0 = b.take(ctx.scale(&delta, coeff[i * 4 + 1], 0.))?;
            let clipped = ctx.zeros(&[1, 24, 3])?;
            b.ops
                .push(ctx.axpby_into(&x0, &x0, None, &clipped, [1., 0., 0.], 1.)?);
            let a = b.take(ctx.scale(&clipped, coeff[i * 4 + 2], 0.))?;
            let d = b.take(ctx.scale(&eps, coeff[i * 4 + 3], 0.))?;
            sample = b.add(&a, &d)?;
            b.record(&format!("sample.{i}"), &sample);
        }
        b.w.finish()?;
        Ok(Self {
            ctx,
            image,
            state,
            noise,
            output: sample,
            operations: b.ops,
            diagnostics: b.diagnostics,
        })
    }
    pub fn forward(&self) -> Result<()> {
        self.operations.iter().try_for_each(Operation::run)
    }
}
