use super::weights::Result;
use serde_json::{json, Value};
use std::path::Path;

pub(super) fn validate(path: &Path) -> Result<()> {
    let v: Value = serde_json::from_str(
        &std::fs::read_to_string(path.join("config.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    for (k, expected) in [
        ("model_type", json!("kart_dp")),
        ("format_version", json!(1)),
        ("vision_backbone", json!("dinov2_vits14")),
        ("dim", json!(384)),
        ("obs_steps", json!(4)),
        ("state_dim", json!(62)),
        ("horizon", json!(24)),
        ("image_shape", json!([224, 288, 3])),
        ("inference_steps", json!(10)),
        ("scheduler", json!("DDIM")),
        ("eta", json!(0)),
        ("action_axes", json!(["steering", "throttle", "brake"])),
    ] {
        if v.get(k) != Some(&expected) {
            return Err(format!("unsupported kart_dp {k}: expected {expected}"));
        }
    }
    Ok(())
}
