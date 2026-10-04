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
                        model_variant="f32", autotune=False):
        root = Path(model_dir)
        config = json.loads((root / "config.json").read_text())
        if config.get("model_type") != "kart_dp" or config.get("format_version") != 1:
            raise ValueError("Expected exported kart_dp format_version 1")
        if model_variant != "f32":
            raise ValueError("kart_dp currently supports f32 only")
        adapter = json.loads((root / "adapter.json").read_text())
        if model_runner is None:
            from apxinf import ModelRunner
            model_runner = ModelRunner.load("kart_dp", root, device=device,
                                           model_variant=model_variant, autotune=autotune)
        return cls(config, adapter, model_runner)

    def __init__(self, config, adapter, model_runner):
        self.model_runner = model_runner
        self._lock = Lock()
        self._closed = False
        self.state_mean = np.asarray(adapter["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(adapter["state_std"], dtype=np.float32)
        self.noise = np.asarray(adapter["initial_noise"], dtype=np.float32)
        if (self.state_mean.shape != (62,) or self.state_std.shape != (62,)
                or self.noise.shape != (1, 24, 3)
                or not all(np.isfinite(x).all() for x in (self.state_mean,self.state_std,self.noise))
                or (self.state_std <= 0).any()):
            raise ValueError("Invalid kart_dp normalization/noise assets")
        if config.get("action_axes") != ["steering", "throttle", "brake"]:
            raise ValueError("Unsupported kart_dp action order")
        self.mean = np.asarray(adapter["image_mean"],np.float32).reshape(1,3,1,1)
        self.std = np.asarray(adapter["image_std"],np.float32).reshape(1,3,1,1)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std<=0).any():
            raise ValueError("Invalid image normalization")
        self.metadata = {"model_type":"kart_dp", "model_variant":"f32", "action_horizon":24,
                         "prediction_horizon":24, "action_dim":3, "state_dim":62,
                         "n_obs_steps":4, "image_shape":[224,288,3],
                         "obs_stride":config.get("obs_stride"), "action_stride":config.get("action_stride"),
                         "action_axes":config["action_axes"], "action_units":"recorded_normalized_controls",
                         "scheduler":"DDIM", "denoising_steps":10, "weight_branch":config["weight_branch"]}

    def _infer(self, observation, noise):
        if self._closed:
            raise RuntimeError("kart_dp policy is closed")
        image=np.asarray(observation["observation.images.front"])
        state=np.asarray(observation["observation.state"],dtype=np.float32)
        if image.dtype!=np.uint8 or image.shape not in ((4,240,320,3),(4,224,288,3)):
            raise ValueError("kart_dp requires four RGB uint8 frames [4,240,320,3] or [4,224,288,3]")
        if state.shape!=(4,62) or not np.isfinite(state).all():
            raise ValueError("kart_dp requires finite oldest-to-newest state [4,62]")
        if image.shape[1:3]==(240,320):
            image=image[:,8:232,16:304]
        image=image.transpose(0,3,1,2).astype(np.float32)/np.float32(255)
        image=np.pad(image,((0,0),(0,0),(0,0),(3,3)),mode="edge")
        image=np.ascontiguousarray((image-self.mean)/self.std)
        state=np.ascontiguousarray(((state-self.state_mean)/self.state_std).reshape(1,248))
        noise=self.noise if noise is None else np.asarray(noise,dtype=np.float32)
        if noise.shape!=(1,24,3) or not np.isfinite(noise).all():
            raise ValueError("kart_dp noise must be finite [1,24,3]")
        prediction=np.asarray(self.model_runner.infer_tensors(image,state,np.ascontiguousarray(noise)))
        if prediction.shape!=(24,3) or not np.isfinite(prediction).all():
            raise ValueError("Invalid kart_dp native output")
        prediction=prediction.copy()
        actions=np.clip(prediction,np.array([-1,0,0],np.float32),np.ones(3,np.float32))
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
            return self.model_runner.prepare_tensors(mode)
        finally: self._lock.release()

    def reset(self):
        """Stateless: the caller supplies all four historical observations."""

    def close(self):
        if not self._lock.acquire(blocking=False): raise RuntimeError("kart_dp policy is busy")
        try:
            self._closed=True
            self.model_runner=None
        finally: self._lock.release()
