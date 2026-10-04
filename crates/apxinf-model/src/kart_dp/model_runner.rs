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
            action_shape: [24, 3],
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
            image_shape: self.model.image.shape().to_vec(),
            state_shape: vec![1, 248],
            action_shape: [24, 3],
            noise_shape: Some(vec![1, 24, 3]),
        })
    }
    fn infer_tensors_host_f32(&self, r: &TensorRequest<'_>) -> Result<Vec<f32>> {
        // Borrow before writing any shared buffers, including input upload.
        let graph = self
            .graph
            .try_borrow_mut()
            .map_err(|_| Error::Other("kart_dp busy".into()))?;
        let ImageTensor::Normalized(image) = r.image else {
            return Err(Error::Other(
                "kart_dp requires canonical float tensor image".into(),
            ));
        };
        let Some(noise) = r.noise else {
            return Err(Error::Other(
                "kart_dp requires explicit initial noise".into(),
            ));
        };
        if image.shape != self.model.image.shape()
            || r.state.shape != [1, 248]
            || noise.shape != [1, 24, 3]
            || image
                .values
                .iter()
                .chain(r.state.values)
                .chain(noise.values)
                .any(|v| !v.is_finite())
        {
            return Err(Error::Other(
                "kart_dp tensor profile mismatch or nonfinite input".into(),
            ));
        }
        self.model.image.write(image.values).map_err(Error::Other)?;
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
        if policy != ExecutionPolicy::Eager {
            self.model.forward().map_err(Error::Other)?;
            self.model.ctx.synchronize().map_err(Error::Other)?;
            if self.autotune {
                for o in &self.model.operations {
                    o.tune().map_err(Error::Other)?;
                }
            }
            self.model.forward().map_err(Error::Other)?;
            self.model.ctx.synchronize().map_err(Error::Other)?;
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
