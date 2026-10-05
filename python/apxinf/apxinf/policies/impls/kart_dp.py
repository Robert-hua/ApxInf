"""Native Kart DINOv2/compact DP. Explicit oldest-to-newest observation window."""
from __future__ import annotations
import json
from pathlib import Path
from threading import Lock
import numpy as np
from ..registry import register_policy

@register_policy("kart_dp")
class KartDpPolicy:
    @classmethod
    def from_pretrained(cls, model_dir, *, device="cuda:0", model_runner=None,
                        model_variant="f32", autotune=False, vision_engine=None):
        root = Path(model_dir)
        config = json.loads((root / "config.json").read_text())
        if config.get("model_type") != "kart_dp" or config.get("format_version") != 1:
            raise ValueError("Expected exported kart_dp format_version 1")
        if model_variant not in ("f32", "fp16", "fp8_policy", "f32_gemm", "f32_compensated", "f32_fast"):
            raise ValueError("kart_dp supports f32 or experimental fp16/fp8_policy/f32_gemm/f32_compensated")
        adapter = json.loads((root / "adapter.json").read_text())
        if vision_engine is not None and config.get("image_shape") != [224,288,3]:
            raise ValueError("TensorRT vision currently supports only the original 3-axis Kart profile")
        vision=None
        try:
            if vision_engine is not None:
                from .kart_dp_trt import KartTrtVision
                if model_runner is not None:raise ValueError("vision_engine owns its native runner contract")
                vision=KartTrtVision(vision_engine,root,int(str(device).split(":")[-1]))
            if model_runner is None:
                from apxinf import ModelRunner
                assets={"vision_features":vision.manifest_path} if vision else {}
                model_runner = ModelRunner.load("kart_dp", root, device=device,
                                               model_variant=model_variant, autotune=autotune,assets=assets)
            return cls(config, adapter, model_runner, model_variant=model_variant,vision=vision)
        except BaseException:
            if vision is not None:vision.close()
            raise

    def __init__(self, config, adapter, model_runner, *, model_variant="f32", vision=None):
        self.model_runner = model_runner
        self.vision = vision
        self._lock = Lock()
        self._closed = False
        self.state_mean = np.asarray(adapter["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(adapter["state_std"], dtype=np.float32)
        self.noise = np.asarray(adapter["initial_noise"], dtype=np.float32)
        if (self.state_mean.shape != (62,) or self.state_std.shape != (62,)
                or self.noise.shape != (1, 24, len(config.get("action_axes", [])))
                or not all(np.isfinite(x).all() for x in (self.state_mean,self.state_std,self.noise))
                or (self.state_std <= 0).any()):
            raise ValueError("Invalid kart_dp normalization/noise assets")
        if config.get("action_axes") not in (["steering", "throttle", "brake"], ["steering", "ry"]):
            raise ValueError("Unsupported kart_dp action order")
        self.action_axes = list(config["action_axes"])
        self.action_dim = len(self.action_axes)
        self.image_shape = list(config.get("image_shape", [224,288,3]))
        self.profile_1004 = self.image_shape == [240,320,3]
        self.mean = np.asarray(adapter["image_mean"],np.float32).reshape(1,3,1,1)
        self.std = np.asarray(adapter["image_std"],np.float32).reshape(1,3,1,1)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std<=0).any():
            raise ValueError("Invalid image normalization")
        self._gpu_rgb = model_variant in ("f32_fast",) and vision is None
        self.metadata = {"model_type":"kart_dp", "model_variant":model_variant, "action_horizon":24,
                         "prediction_horizon":24, "action_dim":self.action_dim, "state_dim":62,
                         "n_obs_steps":4, "image_shape":self.image_shape,
                         "obs_stride":config.get("obs_stride"), "action_stride":config.get("action_stride"),
                         "action_axes":config["action_axes"], "action_units":"recorded_normalized_controls",
                         "vision_backend":"tensorrt" if vision else "native",
                         "vision_handoff":"host D2H/H2D features" if vision else "GPU resident",
                         "scheduler":"DDIM", "denoising_steps":10, "weight_branch":config["weight_branch"]}

    def _infer(self, observation, noise):
        if self._closed:
            raise RuntimeError("kart_dp policy is closed")
        source=observation["observation.images.front"]
        device_rgb=isinstance(source,(list,tuple)) and len(source)==4 and all(hasattr(x,"__cuda_array_interface__") for x in source)
        if device_rgb and not self._gpu_rgb:
            raise ValueError("CUDA RGB input requires native f32_fast")
        image=source if device_rgb else np.asarray(source)
        state=np.asarray(observation["observation.state"],dtype=np.float32)
        valid_shapes=((4,240,320,3),) if self.profile_1004 else ((4,240,320,3),(4,224,288,3))
        if not device_rgb and (image.dtype!=np.uint8 or image.shape not in valid_shapes):
            raise ValueError("kart_dp requires four RGB uint8 frames matching the selected fixed profile")
        if state.shape!=(4,62) or not np.isfinite(state).all():
            raise ValueError("kart_dp requires finite oldest-to-newest state [4,62]")
        if self.profile_1004 and not device_rgb:
            image=image.transpose(0,3,1,2).astype(np.float32)/np.float32(255)
            image=np.pad(image,((0,0),(0,0),(6,6),(1,1)),mode="edge")
            image=np.ascontiguousarray((image-self.mean)/self.std)
        elif not device_rgb and image.shape[1:3]==(240,320):
            image=image[:,8:232,16:304]
        if not self._gpu_rgb and not self.profile_1004:
            image=image.transpose(0,3,1,2).astype(np.float32)/np.float32(255)
            if self.vision is not None:
                image=self.vision.infer(image[None])
            else:
                image=np.pad(image,((0,0),(0,0),(0,0),(3,3)),mode="edge")
                image=np.ascontiguousarray((image-self.mean)/self.std)
        state=np.ascontiguousarray(((state-self.state_mean)/self.state_std).reshape(1,248))
        noise=self.noise if noise is None else np.asarray(noise,dtype=np.float32)
        if noise.shape!=(1,24,self.action_dim) or not np.isfinite(noise).all():
            raise ValueError(f"kart_dp noise must be finite [1,24,{self.action_dim}]")
        if device_rgb:
            prediction=np.asarray(self.model_runner.infer_device_pixels(image,state,self.mean.reshape(3).tolist(),self.std.reshape(3).tolist(),np.ascontiguousarray(noise)))
        elif self._gpu_rgb and not self.profile_1004:
            prediction=np.asarray(self.model_runner.infer_pixels(np.ascontiguousarray(image),state,self.mean.reshape(3).tolist(),self.std.reshape(3).tolist(),np.ascontiguousarray(noise)))
        else:
            prediction=np.asarray(self.model_runner.infer_tensors(image,state,np.ascontiguousarray(noise)))
        if prediction.shape!=(24,self.action_dim) or not np.isfinite(prediction).all():
            raise ValueError("Invalid kart_dp native output")
        prediction=prediction.copy()
        low=np.array([-1.0 if a in ("steering","ry") else 0.0 for a in self.action_axes],np.float32)
        actions=np.clip(prediction,low,np.ones(self.action_dim,np.float32))
        return {"actions":actions,"prediction":prediction}

    def infer(self, observation, *, noise=None, **kwargs):
        if kwargs: raise TypeError(f"Unsupported kart_dp arguments: {sorted(kwargs)}")
        if not self._lock.acquire(blocking=False): raise RuntimeError("kart_dp policy is busy")
        try: return self._infer(observation,noise)
        finally: self._lock.release()

    def prepare(self, observation, *, mode="graph", noise=None):
        if not self._lock.acquire(blocking=False): raise RuntimeError("kart_dp policy is busy")
        try:
            self._infer(observation,noise)
            if self.vision is not None:self.vision.prepare(mode)
            return self.model_runner.prepare_tensors(mode)
        finally: self._lock.release()

    def reset(self):
        """Stateless: the caller supplies all four historical observations."""

    def close(self):
        if not self._lock.acquire(blocking=False): raise RuntimeError("kart_dp policy is busy")
        try:
            self._closed=True
            self.model_runner=None
            if self.vision is not None:self.vision.close();self.vision=None
        finally: self._lock.release()
