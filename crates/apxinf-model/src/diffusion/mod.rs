//! LeRobot Diffusion, fixed single-camera ResNet18 inference profile.
mod config;
mod model;
mod model_runner;
mod weights;
use crate::{LoadOptions, LoadedModel};
use apxinf_core::{Backend, Device, Error, Result};
use std::{path::Path, sync::Arc};
pub(crate) fn register_builtin() {
    crate::registry::register("diffusion-cuda", load);
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
        || !options.assets.is_empty()
        || options.config.is_some()
        || options.synthetic.is_some()
        || !matches!(
            options.model_variant.as_deref(),
            None | Some("f32") | Some("tf32") | Some("bf16")
        )
    {
        return Err(Error::Other(
            "Diffusion supports checkpoint-native config and f32/tf32/bf16 only".into(),
        ));
    }
    let steps = config::validate(path).map_err(Error::Other)?;
    let Device::Cuda(index) = device else {
        return Err(Error::Other("Diffusion requires CUDA".into()));
    };
    let variant = match options.model_variant.as_deref() {
        Some("tf32") => "tf32",
        Some("bf16") => "bf16",
        _ => "f32",
    };
    let ctx = if variant == "bf16" {
        apxinf_cuda_next::tensor_ops::Context::with_bf16(index)
    } else if variant == "tf32" {
        apxinf_cuda_next::tensor_ops::Context::with_tf32(index)
    } else {
        apxinf_cuda_next::tensor_ops::Context::new(index)
    }
    .map_err(Error::Other)?;
    let mut w = weights::Weights::load(path, &ctx).map_err(Error::Other)?;
    let model = model::Model::build(ctx, &mut w, steps).map_err(Error::Other)?;
    Ok(LoadedModel::Vla(Box::new(model_runner::ModelRunner::new(
        model, variant, options.autotune,
    )?)))
}
