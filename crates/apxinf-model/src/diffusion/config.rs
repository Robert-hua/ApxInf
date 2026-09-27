use super::weights::Result;
use serde_json::{json, Value};
use std::path::Path;
pub(super) fn validate(path: &Path) -> Result<usize> {
    let v: Value = serde_json::from_str(
        &std::fs::read_to_string(path.join("config.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    for (k, expected) in [
        ("type", json!("diffusion")),
        ("vision_backbone", json!("resnet18")),
        ("n_obs_steps", json!(1)),
        ("horizon", json!(56)),
        ("n_action_steps", json!(8)),
        ("use_group_norm", json!(false)),
        ("spatial_softmax_num_keypoints", json!(32)),
        ("use_separate_rgb_encoder_per_camera", json!(true)),
        ("down_dims", json!([512, 1024, 2048])),
        ("kernel_size", json!(5)),
        ("n_groups", json!(8)),
        ("diffusion_step_embed_dim", json!(128)),
        ("use_film_scale_modulation", json!(true)),
        ("noise_scheduler_type", json!("DDPM")),
        ("num_train_timesteps", json!(100)),
        ("beta_schedule", json!("squaredcos_cap_v2")),
        ("prediction_type", json!("epsilon")),
        ("clip_sample", json!(true)),
        ("clip_sample_range", json!(1.0)),
        ("resize_shape", Value::Null),
        ("crop_shape", Value::Null),
    ] {
        if v.get(k) != Some(&expected) {
            return Err(format!(
                "Diffusion unsupported {k}: expected {expected}, got {:?}",
                v.get(k)
            ));
        }
    }
    let inputs = v["input_features"].as_object().ok_or("missing inputs")?;
    if inputs.len() != 2
        || v["input_features"]["observation.state"]["shape"] != json!([33])
        || inputs
            .values()
            .filter(|x| x["type"] == "VISUAL" && x["shape"] == json!([3, 360, 640]))
            .count()
            != 1
        || v["output_features"]["action"]["shape"] != json!([2])
    {
        return Err("Diffusion supports one RGB 360x640, state33, action2".into());
    }
    let steps = v["num_inference_steps"].as_u64().ok_or("missing inference steps")? as usize;
    timesteps(steps)?;
    Ok(steps)
}

// Diffusers DDPM leading spacing with num_train_timesteps=100, steps_offset=0.
pub(super) fn timesteps(steps: usize) -> Result<Vec<usize>> {
    if !matches!(steps,10|100) {return Err("Diffusion supports DDPM10 or DDPM100".into())}
    Ok((0..steps).rev().map(|i|i*(100/steps)).collect())
}

#[cfg(test)]
mod tests {
    use super::timesteps;
    #[test]
    fn ddpm10_uses_training_schedule_and_skips_previous_steps() {
        assert_eq!(timesteps(10).unwrap(),vec![90,80,70,60,50,40,30,20,10,0]);
        assert_eq!(timesteps(100).unwrap(),(0..100).rev().collect::<Vec<_>>());
        for steps in [0,1,9,20,101] {assert!(timesteps(steps).is_err());}
    }
}
