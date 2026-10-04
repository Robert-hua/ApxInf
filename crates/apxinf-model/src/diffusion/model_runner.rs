use super::model::Model;
use crate::vla::*;
use apxinf_core::{Error, Result};
use apxinf_cuda_next::tensor_ops::{Graph,RgbProcessor};
use std::{cell::RefCell, collections::BTreeMap};
pub(super) struct ModelRunner {
    pub model: Model,
    rgb:RgbProcessor,
    variant: &'static str,
    autotune:bool,
    graph: RefCell<Option<Graph>>,
}
impl ModelRunner {
    pub fn new(model: Model, variant: &'static str, autotune:bool) -> Result<Self> {
        let rgb=model.ctx.rgb_processor(&model.image).map_err(Error::Other)?;
        Ok(Self {
            rgb,
            model,
            variant,
            autotune,
            graph: RefCell::new(None),
        })
    }
}
impl VlaRuntime for ModelRunner {
    fn model_variant(&self) -> Option<&'static str> {
        Some(self.variant)
    }
    fn contract(&self) -> VlaContract {
        VlaContract {
            action_shape: [56, 2],
            patch_shape: [0, 0],
            max_token_len: 0,
            num_views: 1,
            image_size: 0,
            patch_size: 0,
            accepts_rgb_u8: false,
        }
    }
    fn tensor_profile(&self) -> Option<TensorProfile> {
        Some(TensorProfile {
            image_shape: vec![1, 3, 360, 640],
            state_shape: vec![1, 33],
            action_shape: [56, 2],
            noise_shape: Some(self.model.noise.shape().to_vec()),
        })
    }
    fn infer_tensors_host_f32(&self, r: &TensorRequest<'_>) -> Result<Vec<f32>> {
        let profile = self.tensor_profile().unwrap();
        if r.state.shape != profile.state_shape
            || r.noise.is_none_or(|n| n.shape != self.model.noise.shape())
        {
            return Err(Error::Other("Diffusion tensor profile mismatch".into()));
        }
        let state = r.state.values;
        let noise = r.noise.unwrap().values;
        if state.iter()
            .chain(noise)
            .any(|v| !v.is_finite())
        {
            return Err(Error::Other("nonfinite Diffusion input".into()));
        }
        self.model.noise.write(&noise).map_err(Error::Other)?;
        self.model.state.write(&state).map_err(Error::Other)?;
        match r.image {
            ImageTensor::DeviceRgb{..}=>return Err(Error::Other("device RGB input is not supported by Diffusion".into())),
            ImageTensor::Normalized(image)=>{
                if image.shape!=profile.image_shape || image.values.iter().any(|v|!v.is_finite()){return Err(Error::Other("invalid normalized image".into()))}
                self.model.image.write(image.values).map_err(Error::Other)?;
            }
            ImageTensor::RgbU8{shape,values,mean,std}=>{
                if shape!=[360,640,3]{return Err(Error::Other("invalid RGB shape".into()))}
                self.rgb.write(values,mean,std).map_err(Error::Other)?;
            }
        }
        let graph = self
            .graph
            .try_borrow()
            .map_err(|_| Error::Other("Diffusion execution busy".into()))?;
        if let Some(graph) = graph.as_ref() {
            graph.replay()
        } else {
            self.model.forward()
        }
        .map_err(Error::Other)?;
        self.model.output.read().map_err(Error::Other)
    }
    fn prepare_tensors(&self, policy: ExecutionPolicy) -> Result<PreparationStatus> {
        let mut graph = self
            .graph
            .try_borrow_mut()
            .map_err(|_| Error::Other("Diffusion preparation busy".into()))?;
        *graph = None;
        if policy != ExecutionPolicy::Eager {
            self.model.forward().map_err(Error::Other)?;
            self.model.ctx.synchronize().map_err(Error::Other)?;
            if self.autotune {
                for op in &self.model.operations{op.tune().map_err(Error::Other)?;}
                self.model.forward().map_err(Error::Other)?;
                self.model.ctx.synchronize().map_err(Error::Other)?;
            }
            *graph = Some(
                self.model
                    .ctx
                    .capture(&self.model.operations)
                    .map_err(Error::Other)?,
            );
        }
        Ok(PreparationStatus::Ready {
            mode: if graph.is_some() {
                ExecutionMode::Graph
            } else {
                ExecutionMode::Eager
            },
            fallback_reason: None,
        })
    }
    fn tensor_diagnostics(&self) -> Result<BTreeMap<String, (Vec<usize>, Vec<f32>)>> {
        self.model
            .diagnostics
            .iter()
            .map(|(name, t)| {
                Ok((
                    name.clone(),
                    (t.shape().to_vec(), t.read().map_err(Error::Other)?),
                ))
            })
            .collect()
    }
    fn clear_prepared(&self) -> Result<()> {
        self.graph
            .try_borrow_mut()
            .map_err(|_| Error::Other("Diffusion busy".into()))?
            .take();
        Ok(())
    }
    fn execution_mode(&self) -> &'static str {
        if self.graph.borrow().is_some() {
            "graph"
        } else {
            "eager"
        }
    }
    fn infer(&self, _: &VlaRequest<'_>) -> Result<Action> {
        Err(Error::Other(
            "Diffusion requires the vision/state tensor seam".into(),
        ))
    }
    fn infer_host_f32(&self, _: &VlaRequest<'_>) -> Result<Vec<f32>> {
        Err(Error::Other(
            "Diffusion requires the vision/state tensor seam".into(),
        ))
    }
    fn prepare(&self, _: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        Err(Error::Other(
            "Diffusion uses prepare_tensors and its loaded fixed profile".into(),
        ))
    }
}
