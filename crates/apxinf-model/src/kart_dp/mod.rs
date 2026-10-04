//! DINOv2 small / compact temporal diffusion policy, fixed Kart profile.
mod config;
mod model;
mod model_runner;
mod weights;
use crate::{LoadOptions, LoadedModel};
use apxinf_core::{Backend, Device, Error, Result};
use std::{path::Path, sync::Arc};
pub(crate) fn register_builtin() {
    crate::registry::register("kart_dp-cuda", load);
}
fn load(
    path: &Path,
    device: Device,
    _backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if options.precision != crate::ModelPrecision::Auto
        || options.calibration_path.is_some()
        || options.uniform_fp8_scale.is_some()
        || options.assets.keys().any(|k| k != "vision_features")
        || options.config.is_some()
        || options.synthetic.is_some()
        || !matches!(options.model_variant.as_deref(), None | Some("f32" | "fp16" | "fp8_policy" | "f32_gemm" | "f32_compensated" | "f32_fast"))
    {
        return Err(Error::Other(
            "kart_dp supports exported checkpoint config and f32/fp16/fp8_policy/f32_gemm/f32_compensated/f32_fast only".into(),
        ));
    }
    let external_vision = if let Some(contract) = options.assets.get("vision_features") {
        let data: serde_json::Value = serde_json::from_slice(&std::fs::read(contract)?).map_err(|e|Error::Other(e.to_string()))?;
        if data["input_contract"] != "kart_dp_vision_features_v1" {
            return Err(Error::Other("invalid external vision feature contract".into()));
        }
        true
    } else { false };
    config::validate(path).map_err(Error::Other)?;
    let Device::Cuda(index) = device else {
        return Err(Error::Other("kart_dp requires CUDA".into()));
    };
    let ctx = apxinf_cuda_next::tensor_ops::Context::new(index).map_err(Error::Other)?;
    let mut weights = weights::Weights::load(path, &ctx).map_err(Error::Other)?;
    let variant = match options.model_variant.as_deref() {Some("f32_fast")=>"f32_fast",Some("f32_gemm")=>"f32_gemm",Some("f32_compensated")=>"f32_compensated",Some("fp16")=>"fp16",Some("fp8_policy")=>"fp8_policy",_=>"f32"};
    let model = model::Model::build(ctx, &mut weights, variant, external_vision).map_err(Error::Other)?;
    Ok(LoadedModel::Vla(Box::new(model_runner::ModelRunner::new(
        model,
        options.autotune,
    ))))
}
