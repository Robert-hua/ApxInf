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
        ("inference_steps", json!(10)),
        ("scheduler", json!("DDIM")),
        ("eta", json!(0)),
    ] {
        if v.get(k) != Some(&expected) {
            return Err(format!("unsupported kart_dp {k}: expected {expected}"));
        }
    }
    let shape=v.get("image_shape").ok_or("missing kart_dp image_shape")?;
    if shape!=&json!([224,288,3]) && shape!=&json!([240,320,3]) {
        return Err(format!("unsupported kart_dp image_shape: {shape}"));
    }
    let axes=v.get("action_axes").ok_or("missing kart_dp action_axes")?;
    if axes!=&json!(["steering","throttle","brake"]) && axes!=&json!(["steering","ry"]) {
        return Err(format!("unsupported kart_dp action_axes: {axes}"));
    }
    if (shape==&json!([240,320,3])) != (axes==&json!(["steering","ry"])) {
        return Err("kart_dp image_shape and action_axes identify different model profiles".into());
    }
    Ok(())
}
