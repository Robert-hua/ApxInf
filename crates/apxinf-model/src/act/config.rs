use super::weights::Result;
use serde_json::{json, Value};
use std::path::Path;
pub(super) fn validate(path: &Path) -> Result<()> {
    let v: Value = serde_json::from_str(
        &std::fs::read_to_string(path.join("config.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    for (k, expected) in [
        ("type", json!("act")),
        ("vision_backbone", json!("resnet18")),
        ("n_obs_steps", json!(1)),
        ("chunk_size", json!(50)),
        ("n_action_steps", json!(8)),
        ("dim_model", json!(512)),
        ("n_heads", json!(8)),
        ("dim_feedforward", json!(3200)),
        ("n_encoder_layers", json!(4)),
        ("n_decoder_layers", json!(1)),
        ("latent_dim", json!(32)),
        ("pre_norm", json!(false)),
        ("replace_final_stride_with_dilation", json!(false)),
        ("feedforward_activation", json!("relu")),
        ("temporal_ensemble_coeff", Value::Null),
    ] {
        if v.get(k) != Some(&expected) {
            return Err(format!(
                "ACT unsupported {k}: expected {expected}, got {:?}",
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
        return Err("ACT supports one RGB 360x640, state33, action2".into());
    }
    Ok(())
}
