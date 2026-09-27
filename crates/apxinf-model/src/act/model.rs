//! ACT evaluation math. Preparation builds stable device operations; no runner dependency.
use super::weights::{Result, Weights};
use apxinf_cuda_next::tensor_ops::{Activation, Context, Operation, Tensor};
pub(super) struct Model {
    pub ctx: Context,
    pub image: Tensor,
    pub state: Tensor,
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
        let parameters = self.ctx.tensor(
            &[4, w.len()],
            &[w.as_slice(), b.as_slice(), mean.as_slice(), var.as_slice()].concat(),
        )?;
        self.take(self.ctx.frozen_batch_norm(x, &parameters, 1e-5))
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
    fn attention(&mut self, q: &Tensor, k: &Tensor, v: &Tensor, name: &str) -> Result<Tensor> {
        let w = self.w.get(&format!("{name}.in_proj_weight"))?;
        let bias = self.w.get(&format!("{name}.in_proj_bias"))?;
        if w.shape() != [1536, 512] || bias.shape() != [1536] {
            return Err("ACT attention projection shape".into());
        }
        let mut projected = Vec::new();
        for (i, x) in [q, k, v].iter().enumerate() {
            let ww = w.view(i * 512 * 512, &[512, 512])?;
            let bb = bias.view(i * 512, &[512])?;
            let y = self.take(self.ctx.linear(x, &ww, Some(&bb)))?;
            let y = y.reshape(&[x.shape()[0], 8, 64])?;
            let y = self.take(self.ctx.permute(&y, &[1, 0, 2]))?;
            projected.push(y);
        }
        let scores = self.take(self.ctx.bmm(&projected[0], &projected[1], true, 1. / 8.))?;
        let probs = self.take(self.ctx.softmax(&scores, 1.))?;
        let output = self.take(self.ctx.bmm(&probs, &projected[2], false, 1.))?;
        let output = self
            .take(self.ctx.permute(&output, &[1, 0, 2]))?
            .reshape(&[q.shape()[0], 512])?;
        self.linear(&output, &format!("{name}.out_proj"))
    }
}
impl Model {
    pub fn build(ctx: Context, w: &mut Weights) -> Result<Self> {
        let image = ctx.zeros(&[1, 3, 360, 640])?;
        let state = ctx.zeros(&[1, 33])?;
        let mut b = Builder {
            ctx: ctx.clone(),
            w,
            ops: Vec::new(),
            diagnostics: Vec::new(),
        };
        let mut x = b.conv(&image, "model.backbone.conv1", 2, 3)?;
        x = b.bn(&x, "model.backbone.bn1")?;
        x = b.relu(&x)?;
        x = b.take(ctx.maxpool2d(&x, 3, 2, 1))?;
        for layer in 1..=4 {
            for block in 0..2 {
                let p = format!("model.backbone.layer{layer}.{block}");
                let stride = if layer > 1 && block == 0 { 2 } else { 1 };
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
                x = b.relu(&x)?;
            }
            b.record(&format!("backbone.layer{layer}"), &x);
        }
        let (h, ww) = (x.shape()[2], x.shape()[3]);
        x = b.conv(&x, "model.encoder_img_feat_input_proj", 1, 0)?;
        x = b
            .take(ctx.permute(&x, &[0, 2, 3, 1]))?
            .reshape(&[h * ww, 512])?;
        let latent = ctx.zeros(&[1, 32])?;
        let latent = b.linear(&latent, "model.encoder_latent_input_proj")?;
        let st = b.linear(&state, "model.encoder_robot_state_input_proj")?;
        let prefix = b.take(ctx.concat(&latent, &st, 0))?;
        x = b.take(ctx.concat(&prefix, &x, 0))?;
        let learned = b.w.values("model.encoder_1d_feature_pos_embed.weight")?;
        if learned.len() != 1024 {
            return Err("ACT position prefix shape".into());
        }
        let mut pos = learned;
        for iy in 0..h {
            for ix in 0..ww {
                for axis in 0..2 {
                    for d in 0..256 {
                        let coordinate = if axis == 0 {
                            (iy + 1) as f32 / (h as f32 + 1e-6)
                        } else {
                            (ix + 1) as f32 / (ww as f32 + 1e-6)
                        };
                        let angle = coordinate * (2. * std::f32::consts::PI)
                            / 10000f32.powf((2 * (d / 2)) as f32 / 256.);
                        pos.push(if d % 2 == 0 { angle.sin() } else { angle.cos() });
                    }
                }
            }
        }
        let pos = ctx.tensor(&[h * ww + 2, 512], &pos)?;
        b.record("encoder.input", &x);
        for layer in 0..4 {
            let p = format!("model.encoder.layers.{layer}");
            let qp = b.add(&x, &pos)?;
            let a = b.attention(&qp, &qp, &x, &format!("{p}.self_attn"))?;
            x = b.add(&x, &a)?;
            x = b.norm(&x, &format!("{p}.norm1"))?;
            let f = b.linear(&x, &format!("{p}.linear1"))?;
            let f = b.relu(&f)?;
            let f = b.linear(&f, &format!("{p}.linear2"))?;
            x = b.add(&x, &f)?;
            x = b.norm(&x, &format!("{p}.norm2"))?;
            b.record(&format!("encoder.{layer}"), &x);
        }
        let memory = x;
        let query_pos = b.w.get("model.decoder_pos_embed.weight")?;
        let mut x = ctx.zeros(&[50, 512])?;
        let p = "model.decoder.layers.0";
        let q = b.add(&x, &query_pos)?;
        let a = b.attention(&q, &q, &x, &format!("{p}.self_attn"))?;
        x = b.add(&x, &a)?;
        x = b.norm(&x, &format!("{p}.norm1"))?;
        let q = b.add(&x, &query_pos)?;
        let k = b.add(&memory, &pos)?;
        let a = b.attention(&q, &k, &memory, &format!("{p}.multihead_attn"))?;
        x = b.add(&x, &a)?;
        x = b.norm(&x, &format!("{p}.norm2"))?;
        let f = b.linear(&x, &format!("{p}.linear1"))?;
        let f = b.relu(&f)?;
        let f = b.linear(&f, &format!("{p}.linear2"))?;
        x = b.add(&x, &f)?;
        x = b.norm(&x, &format!("{p}.norm3"))?;
        x = b.norm(&x, "model.decoder.norm")?;
        b.record("decoder", &x);
        let output = b.linear(&x, "model.action_head")?;
        b.w.finish()?;
        Ok(Self {
            ctx,
            image,
            state,
            output,
            operations: b.ops,
            diagnostics: b.diagnostics,
        })
    }
    pub fn forward(&self) -> Result<()> {
        self.operations.iter().try_for_each(Operation::run)
    }
}
