//! Diffusion Policy model math: ResNet condition, conditional U-Net and DDPM100.
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
pub(super) struct Builder<'a> {
    pub ctx: Context,
    pub w: &'a mut Weights,
    pub ops: Vec<Operation>,
    pub diagnostics: Vec<(String, Tensor)>,
}
impl<'a> Builder<'a> {
    pub fn take(&mut self, r: Result<(Tensor, Operation)>) -> Result<Tensor> {
        let (t, op) = r?;
        self.ops.push(op);
        Ok(t)
    }
    pub fn cast_activation(&mut self, x: &Tensor) -> Result<Tensor> {
        if self.ctx.is_bf16() {
            self.take(self.ctx.round_bf16(x))
        } else {
            Ok(x.clone())
        }
    }
    pub fn record(&mut self, name: &str, t: &Tensor) {
        self.diagnostics.push((name.into(), t.clone()));
    }
    pub fn linear(&mut self, x: &Tensor, name: &str) -> Result<Tensor> {
        let w = self.w.get(&format!("{name}.weight"))?;
        let b = self.w.get(&format!("{name}.bias"))?;
        self.take(self.ctx.linear(x, &w, Some(&b)))
    }
    pub fn conv(&mut self, x: &Tensor, name: &str, stride: usize, pad: usize) -> Result<Tensor> {
        let w = self.w.get(&format!("{name}.weight"))?;
        let b = if self.w.has(&format!("{name}.bias")) {
            Some(self.w.get(&format!("{name}.bias"))?)
        } else {
            None
        };
        self.take(
            self.ctx
                .conv2d(x, &w, b.as_ref(), [stride, stride], [pad, pad], false),
        )
    }
    pub fn bn(&mut self, x: &Tensor, name: &str) -> Result<Tensor> {
        let w = self.w.values(&format!("{name}.weight"))?;
        let b = self.w.values(&format!("{name}.bias"))?;
        let mean = self.w.values(&format!("{name}.running_mean"))?;
        let var = self.w.values(&format!("{name}.running_var"))?;
        if [b.len(), mean.len(), var.len()]
            .iter()
            .any(|&n| n != w.len())
        {
            return Err("batch norm parameter lengths differ".into());
        }
        if self.ctx.is_bf16() {
            let parameters=self.ctx.tensor(&[4,w.len()], &[w.as_slice(),b.as_slice(),mean.as_slice(),var.as_slice()].concat())?;
            return self.take(self.ctx.batch_norm(x,&parameters,1e-5));
        }
        let scale = w
            .iter()
            .zip(&var)
            .map(|(w, v)| w / (v + 1e-5).sqrt())
            .collect::<Vec<_>>();
        let bias = b
            .iter()
            .zip(&mean)
            .zip(&scale)
            .map(|((b, m), s)| b - m * s)
            .collect::<Vec<_>>();
        let s = self.ctx.tensor(&[w.len()], &scale)?;
        let b = self.ctx.tensor(&[w.len()], &bias)?;
        let y = self.take(self.ctx.affine(x, &s, &b, x.shape()[2] * x.shape()[3]))?;
        self.cast_activation(&y)
    }
    pub fn relu(&mut self, x: &Tensor) -> Result<Tensor> {
        self.take(self.ctx.activation(x, Activation::Relu))
    }
    pub fn add(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.take(self.ctx.add(a, b))
    }
    pub fn norm(&mut self, x: &Tensor, name: &str) -> Result<Tensor> {
        let w = self.w.get(&format!("{name}.weight"))?;
        let b = self.w.get(&format!("{name}.bias"))?;
        self.take(self.ctx.norm(x, &w, &b, 512, 1, 1e-5))
    }
}
impl<'a> Builder<'a> {
    fn conv1d(
        &mut self,
        x: &Tensor,
        name: &str,
        stride: usize,
        pad: usize,
        transpose: bool,
    ) -> Result<Tensor> {
        let w = self.w.get(&format!("{name}.weight"))?;
        let s = w.shape();
        if s.len() != 3 {
            return Err(format!("Conv1d weight rank: {name}"));
        }
        let w = w.reshape(&[s[0], s[1], 1, s[2]])?;
        let bias = self.w.get(&format!("{name}.bias"))?;
        let xs = x.shape();
        let x4 = x.reshape(&[xs[0], xs[1], 1, xs[2]])?;
        let y =
            self.take(
                self.ctx
                    .conv2d(&x4, &w, Some(&bias), [1, stride], [0, pad], transpose),
            )?;
        y.reshape(&[y.shape()[0], y.shape()[1], y.shape()[3]])
    }
    fn conv_block(&mut self, x: &Tensor, name: &str) -> Result<Tensor> {
        let x = self.conv1d(x, &format!("{name}.block.0"), 1, 2, false)?;
        let weight = self.w.get(&format!("{name}.block.1.weight"))?;
        let bias = self.w.get(&format!("{name}.block.1.bias"))?;
        let spatial = x.shape()[2];
        let x = self.take(
            self.ctx
                .norm(&x, &weight, &bias, x.len() / 8, spatial, 1e-5),
        )?;
        self.take(self.ctx.activation(&x, Activation::Mish))
    }
    fn residual(&mut self, x: &Tensor, condition: &Tensor, name: &str) -> Result<Tensor> {
        let mut y = self.conv_block(x, &format!("{name}.conv1"))?;
        let cond = self.take(self.ctx.activation(condition, Activation::Mish))?;
        let cond = self.linear(&cond, &format!("{name}.cond_encoder.1"))?;
        let channels = y.shape()[1];
        let scale = cond.view(0, &[channels])?;
        let bias = cond.view(channels, &[channels])?;
        y = self.take(self.ctx.affine(&y, &scale, &bias, y.shape()[2]))?;
        y = self.conv_block(&y, &format!("{name}.conv2"))?;
        let skip = if self.w.has(&format!("{name}.residual_conv.weight")) {
            self.conv1d(x, &format!("{name}.residual_conv"), 1, 0, false)?
        } else {
            x.clone()
        };
        self.add(&y, &skip)
    }
}
impl Model {
    pub fn build(ctx: Context, w: &mut Weights) -> Result<Self> {
        let image = ctx.zeros(&[1, 3, 360, 640])?;
        let state = ctx.zeros(&[1, 33])?;
        // Index zero is x_T, indices 1..100 are step noises; the final draw is unused.
        let noise = ctx.zeros(&[101, 56, 2])?;
        let sample = ctx.zeros(&[1, 56, 2])?;
        let mut b = Builder {
            ctx: ctx.clone(),
            w,
            ops: Vec::new(),
            diagnostics: Vec::new(),
        };
        let p = "diffusion.rgb_encoder.0.backbone";
        let mut x = b.conv(&image, &format!("{p}.0"), 2, 3)?;
        x = b.bn(&x, &format!("{p}.1"))?;
        x = b.relu(&x)?;
        x = b.take(ctx.maxpool2d(&x, 3, 2, 1))?;
        for layer in 4..=7 {
            for block in 0..2 {
                let p = format!("{p}.{layer}.{block}");
                let stride = if layer > 4 && block == 0 { 2 } else { 1 };
                let skip = x.clone();
                x = b.conv(&x, &format!("{p}.conv1"), stride, 1)?;
                x = b.bn(&x, &format!("{p}.bn1"))?;
                x = b.relu(&x)?;
                x = b.conv(&x, &format!("{p}.conv2"), 1, 1)?;
                x = b.bn(&x, &format!("{p}.bn2"))?;
                let skip = if stride == 2 {
                    let s = b.conv(&skip, &format!("{p}.downsample.0"), stride, 0)?;
                    b.bn(&s, &format!("{p}.downsample.1"))?
                } else {
                    skip
                };
                x = b.add(&x, &skip)?;
                x = b.cast_activation(&x)?;
                x = b.relu(&x)?;
            }
            b.record(&format!("backbone.layer{}", layer - 3), &x);
        }
        x = b
            .conv(&x, "diffusion.rgb_encoder.0.pool.nets", 1, 0)?
            .reshape(&[32, 240])?;
        x = b.take(ctx.softmax(&x, 1.))?;
        x = b.cast_activation(&x)?;
        let grid =
            b.w.get("diffusion.rgb_encoder.0.pool.pos_grid")?
                .reshape(&[1, 240, 2])?;
        x = b
            .take(ctx.bmm(&x.reshape(&[1, 32, 240])?, &grid, false, 1.))?
            .reshape(&[1, 64])?;
        x = b.linear(&x, "diffusion.rgb_encoder.0.out")?;
        x = b.relu(&x)?;
        let condition = b.take(ctx.concat(&state, &x, 1))?;
        b.record("condition", &condition);
        b.ops
            .push(ctx.copy_into(&noise.view(0, &[1, 56, 2])?, &sample)?);
        let prefix_ops = std::mem::take(&mut b.ops);
        let time_input = ctx.zeros(&[1, 128])?;
        let mut time = b.linear(&time_input, "diffusion.unet.diffusion_step_encoder.1")?;
        time = b.take(ctx.activation(&time, Activation::Mish))?;
        time = b.cast_activation(&time)?;
        time = b.linear(&time, "diffusion.unet.diffusion_step_encoder.3")?;
        let global = b.take(ctx.concat(&time, &condition, 1))?;
        let mut x = b.take(ctx.permute(&sample, &[0, 2, 1]))?;
        let mut skips = Vec::new();
        for i in 0..3 {
            let p = format!("diffusion.unet.down_modules.{i}");
            x = b.residual(&x, &global, &format!("{p}.0"))?;
            x = b.residual(&x, &global, &format!("{p}.1"))?;
            skips.push(x.clone());
            if i < 2 {
                x = b.conv1d(&x, &format!("{p}.2"), 2, 1, false)?;
            }
        }
        for i in 0..2 {
            x = b.residual(&x, &global, &format!("diffusion.unet.mid_modules.{i}"))?;
        }
        for i in 0..2 {
            let p = format!("diffusion.unet.up_modules.{i}");
            x = b.take(ctx.concat(&x, &skips.pop().unwrap(), 1))?;
            x = b.residual(&x, &global, &format!("{p}.0"))?;
            x = b.residual(&x, &global, &format!("{p}.1"))?;
            x = b.conv1d(&x, &format!("{p}.2"), 2, 1, true)?;
        }
        x = b.conv_block(&x, "diffusion.unet.final_conv.0")?;
        x = b.conv1d(&x, "diffusion.unet.final_conv.1", 1, 0, false)?;
        let epsilon = b.take(ctx.permute(&x, &[0, 2, 1]))?;
        b.record("unet.last_epsilon", &epsilon);
        let unet_ops = std::mem::take(&mut b.ops);
        let mut operations = prefix_ops;
        // Diffusers squaredcos_cap_v2 creates float32 betas from this f64 formula.
        let alpha_bar = |t: f64| {
            ((t + 0.008) / 1.008 * std::f64::consts::PI / 2.)
                .cos()
                .powi(2)
        };
        let betas = (0..100)
            .map(|i| {
                (1. - alpha_bar((i + 1) as f64 / 100.) / alpha_bar(i as f64 / 100.)).min(0.999)
                    as f32
            })
            .collect::<Vec<_>>();
        let mut cumulative = Vec::new();
        let mut product = 1f64;
        for beta in betas {
            product *= (1f32 - beta) as f64;
            cumulative.push(product as f32);
        }
        let original = ctx.zeros(&[1, 56, 2])?;
        for (step, t) in (0..100).rev().enumerate() {
            let mut values = Vec::with_capacity(128);
            for trig in 0..2 {
                for d in 0..64 {
                    let angle = t as f32 * ((d as f32) * (-10000f32.ln() / 63.)).exp();
                    values.push(if trig == 0 { angle.sin() } else { angle.cos() });
                }
            }
            let embedding = ctx.tensor(&[1, 128], &values)?;
            operations.push(ctx.copy_into(&embedding, &time_input)?);
            operations.extend(unet_ops.iter().cloned());
            let snapshot = ctx.zeros(&[1, 56, 2])?;
            operations.push(ctx.copy_into(&epsilon, &snapshot)?);
            b.diagnostics.push((format!("ddpm.{t}.epsilon"), snapshot));
            let at = cumulative[t];
            let prev = if t == 0 { 1. } else { cumulative[t - 1] };
            let bt = 1. - at;
            let bp = 1. - prev;
            let current_alpha = at / prev;
            let current_beta = 1. - current_alpha;
            // Torch autocast preserves the epsilon tensor's BF16 dtype when multiplied
            // by a scalar, before subtracting it from the FP32 sample.
            if ctx.is_bf16() {
                let (scaled, op) = ctx.scale(&epsilon, bt.sqrt(), 0.)?;
                operations.push(op);
                let (rounded, op) = ctx.round_bf16(&scaled)?;
                operations.push(op);
                // Diffusers subtracts in FP32 before dividing; distributing the
                // reciprocal changes cancellation at early noisy timesteps.
                let difference = ctx.zeros(&[1, 56, 2])?;
                operations.push(ctx.axpby_into(
                    &sample,
                    &rounded,
                    None,
                    &difference,
                    [1., -1., 0.],
                    0.,
                )?);
                operations.push(ctx.axpby_into(
                    &difference,
                    &difference,
                    None,
                    &original,
                    [1. / at.sqrt(), 0., 0.],
                    1.,
                )?);
            } else {
                operations.push(ctx.axpby_into(
                    &sample,
                    &epsilon,
                    None,
                    &original,
                    [1. / at.sqrt(), -bt.sqrt() / at.sqrt(), 0.],
                    1.,
                )?);
            }
            let coeff_original = prev.sqrt() * current_beta / bt;
            let coeff_sample = current_alpha.sqrt() * bp / bt;
            let variance = (bp / bt * current_beta).max(1e-20).sqrt();
            let step_noise = noise.view((step + 1) * 112, &[1, 56, 2])?;
            if ctx.is_bf16() && t > 0 {
                let (scaled, op) = ctx.scale(&step_noise, variance, 0.)?;
                operations.push(op);
                let (rounded, op) = ctx.round_bf16(&scaled)?;
                operations.push(op);
                operations.push(ctx.axpby_into(
                    &original,
                    &sample,
                    Some(&rounded),
                    &sample,
                    [coeff_original, coeff_sample, 1.],
                    0.,
                )?);
            } else {
                operations.push(ctx.axpby_into(
                    &original,
                    &sample,
                    Some(&step_noise),
                    &sample,
                    [
                        coeff_original,
                        coeff_sample,
                        if t > 0 { variance } else { 0. },
                    ],
                    0.,
                )?);
            }
            let snapshot = ctx.zeros(&[1, 56, 2])?;
            operations.push(ctx.copy_into(&sample, &snapshot)?);
            b.diagnostics.push((format!("ddpm.{t}.sample"), snapshot));
        }
        b.w.finish()?;
        Ok(Self {
            ctx,
            image,
            state,
            noise,
            output: sample,
            operations,
            diagnostics: b.diagnostics,
        })
    }
    pub fn forward(&self) -> Result<()> {
        self.operations.iter().try_for_each(Operation::run)
    }
}
