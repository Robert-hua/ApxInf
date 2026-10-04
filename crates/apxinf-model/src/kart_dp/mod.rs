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
        || !options.assets.is_empty()
        || options.config.is_some()
        || options.synthetic.is_some()
        || !matches!(options.model_variant.as_deref(), None | Some("f32"))
    {
        return Err(Error::Other(
            "kart_dp supports exported checkpoint config and f32 only".into(),
        ));
    }
    config::validate(path).map_err(Error::Other)?;
    let Device::Cuda(index) = device else {
        return Err(Error::Other("kart_dp requires CUDA".into()));
    };
    let ctx = apxinf_cuda_next::tensor_ops::Context::new(index).map_err(Error::Other)?;
    let mut weights = weights::Weights::load(path, &ctx).map_err(Error::Other)?;
    let model = model::Model::build(ctx, &mut weights).map_err(Error::Other)?;
    Ok(LoadedModel::Vla(Box::new(model_runner::ModelRunner::new(
        model,
        options.autotune,
    ))))
}
