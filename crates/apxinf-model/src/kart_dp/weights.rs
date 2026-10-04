use apxinf_cuda_next::tensor_ops::{Context, Tensor};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};
pub(super) type Result<T> = std::result::Result<T, String>;
pub(super) struct Weights {
    host: HashMap<String, apxinf_core::Tensor>,
    device: HashMap<String, Tensor>,
    used: HashSet<String>,
    ctx: Context,
}
impl Weights {
    pub fn load(path: &Path, ctx: &Context) -> Result<Self> {
        // Original policy/static constants are retained in the export for strict
        // independent reference reload. Their native replacements are required below.
        let (host, _) = apxinf_loader::safetensors::load_native_selected(
            &path.join("model.safetensors"),
            &|k| {
                !matches!(
                    k,
                    "initial_noise"
                        | "vision.mean"
                        | "vision.std"
                        | "vision.backbone.cls_token"
                        | "vision.backbone.pos_embed"
                        | "vision.backbone.mask_token"
                        | "time.0.freq"
                )
            },
        )?;
        Ok(Self {
            host,
            device: HashMap::new(),
            used: HashSet::new(),
            ctx: ctx.clone(),
        })
    }
    pub fn get(&mut self, name: &str) -> Result<Tensor> {
        self.used.insert(name.into());
        if let Some(t) = self.device.get(name) {
            return Ok(t.clone());
        }
        let h = self
            .host
            .get(name)
            .ok_or_else(|| format!("missing weight {name}"))?;
        let values = h.to_f32_vec().map_err(|e| e.to_string())?;
        let t = self.ctx.tensor(h.shape().dims(), &values)?;
        self.device.insert(name.into(), t.clone());
        Ok(t)
    }
    pub fn values(&mut self, name: &str) -> Result<Vec<f32>> {
        self.used.insert(name.into());
        self.host
            .get(name)
            .ok_or_else(|| format!("missing weight {name}"))?
            .to_f32_vec()
            .map_err(|e| e.to_string())
    }
    pub fn has(&self, name: &str) -> bool {
        self.host.contains_key(name)
    }
    /// External vision owns these tensors; mark the exact namespace as mapped.
    pub fn external_vision(&mut self) {
        for name in self.host.keys() {
            if name.starts_with("vision.") || matches!(name.as_str(),"native.position"|"native.cls_tokens") {
                self.used.insert(name.clone());
            }
        }
    }
    pub fn finish(&self) -> Result<()> {
        let mut unused = self
            .host
            .keys()
            .filter(|k| !self.used.contains(*k))
            .cloned()
            .collect::<Vec<_>>();
        unused.sort();
        if unused.is_empty() {
            Ok(())
        } else {
            Err(format!("unmapped inference weights: {unused:?}"))
        }
    }
}
