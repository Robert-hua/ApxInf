use super::model::Model;
use crate::vla::*;
use apxinf_core::{Error, Result};
use apxinf_cuda_next::tensor_ops::Graph;
use std::{cell::RefCell, collections::BTreeMap};
pub(super) struct ModelRunner {
    model: Model,
    autotune: bool,
    graph: RefCell<Option<Graph>>,
}
impl ModelRunner {
    pub fn new(model: Model, autotune: bool) -> Self {
        Self {
            model,
            autotune,
            graph: RefCell::new(None),
        }
    }
}
impl VlaRuntime for ModelRunner {
    fn model_variant(&self) -> Option<&'static str> {
        Some(self.model.variant)
    }
    fn contract(&self) -> VlaContract {
        VlaContract {
            action_shape: [24, self.model.action_dim],
            patch_shape: [0, 0],
            max_token_len: 0,
            num_views: 1,
            image_size: 0,
            patch_size: 0,
            accepts_rgb_u8: self.model.rgb.is_some(),
        }
    }
    fn tensor_profile(&self) -> Option<TensorProfile> {
        Some(TensorProfile {
            image_shape: self.model.image.shape().to_vec(),
            state_shape: vec![1, 248],
            action_shape: [24, self.model.action_dim],
            noise_shape: Some(vec![1, 24, self.model.action_dim]),
        })
    }
    fn infer_tensors_host_f32(&self, r: &TensorRequest<'_>) -> Result<Vec<f32>> {
        // Borrow before writing any shared buffers, including input upload.
        let graph = self
            .graph
            .try_borrow_mut()
            .map_err(|_| Error::Other("kart_dp busy".into()))?;
        let Some(noise) = r.noise else {
            return Err(Error::Other(
                "kart_dp requires explicit initial noise".into(),
            ));
        };
        if r.state.shape != [1,248] || noise.shape != [1,24,self.model.action_dim]
            || r.state.values.iter().chain(noise.values).any(|v|!v.is_finite()) {
            return Err(Error::Other("kart_dp state/noise profile mismatch".into()));
        }
        match r.image {
            ImageTensor::DeviceRgb{source,mean,std}=>{
                self.model.rgb.as_ref().ok_or_else(||Error::Other("device RGB requires native f32_fast".into()))?
                    .write_device(source,mean,std).map_err(Error::Other)?;
            }
            ImageTensor::Normalized(image)=>{
                if image.shape!=self.model.image.shape() || image.values.iter().any(|v|!v.is_finite()) {
                    return Err(Error::Other("kart_dp float image profile mismatch".into()));
                }
                self.model.image.write(image.values).map_err(Error::Other)?;
            }
            ImageTensor::RgbU8{shape,values,mean,std}=>{
                if self.model.rgb.is_none() {return Err(Error::Other("RGB upload requires native f32_fast on the original image profile".into()));}
                let expected=[4,self.model.image.shape()[2],self.model.image.shape()[3].saturating_sub(2),3];
                if shape!=expected {return Err(Error::Other("kart_dp RGB batch shape mismatch".into()));}
                self.model.rgb.as_ref().ok_or_else(||Error::Other("RGB upload requires native f32_fast".into()))?
                    .write(values,mean,std).map_err(Error::Other)?;
            }
        }
        self.model
            .state
            .write(r.state.values)
            .map_err(Error::Other)?;
        self.model.noise.write(noise.values).map_err(Error::Other)?;
        if let Some(g) = graph.as_ref() {
            g.replay()
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
            .map_err(|_| Error::Other("kart_dp busy".into()))?;
        *graph = None;
        // Fix the same prepared tactics before either execution mode. Tuning
        // only during Graph preparation would compare untuned eager arithmetic
        // with tuned Graph arithmetic when the user requests autotune.
        if self.autotune || policy != ExecutionPolicy::Eager {
            self.model.forward().map_err(Error::Other)?;
            self.model.ctx.synchronize().map_err(Error::Other)?;
            if self.autotune {
                for o in &self.model.operations {
                    o.tune().map_err(Error::Other)?;
                }
            }
            self.model.forward().map_err(Error::Other)?;
            self.model.ctx.synchronize().map_err(Error::Other)?;
        }
        if policy != ExecutionPolicy::Eager {
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
    fn clear_prepared(&self) -> Result<()> {
        self.graph
            .try_borrow_mut()
            .map_err(|_| Error::Other("kart_dp busy".into()))?
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
    fn tensor_diagnostics(&self) -> Result<BTreeMap<String, (Vec<usize>, Vec<f32>)>> {
        let _guard = self
            .graph
            .try_borrow_mut()
            .map_err(|_| Error::Other("kart_dp busy".into()))?;
        self.model
            .diagnostics
            .iter()
            .map(|(n, t)| {
                Ok((
                    n.clone(),
                    (t.shape().to_vec(), t.read().map_err(Error::Other)?),
                ))
            })
            .collect()
    }
    fn infer(&self, _: &VlaRequest<'_>) -> Result<Action> {
        Err(Error::Other("kart_dp uses tensor input seam".into()))
    }
    fn infer_host_f32(&self, _: &VlaRequest<'_>) -> Result<Vec<f32>> {
        Err(Error::Other("kart_dp uses tensor input seam".into()))
    }
    fn prepare(&self, _: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        Err(Error::Other(
            "kart_dp uses prepare_tensors for its fixed profile".into(),
        ))
    }
}
