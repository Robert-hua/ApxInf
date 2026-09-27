"""LeRobot ACT ResNet18 policy, with a native ApxInf model runner."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from ..registry import register_policy

@register_policy("act")
class ACTPolicy:
    @classmethod
    def from_pretrained(cls, model_dir, *, device="cuda:0", model_runner=None, model_variant="f32", autotune=False):
        root = Path(model_dir)
        config = json.loads((root / "config.json").read_text())
        adapter = json.loads((root / "adapter.json").read_text())
        if config["type"] != "act":
            raise ValueError("ACT policy requires ACT checkpoint")
        if model_runner is None:
            from apxinf import ModelRunner
            model_runner = ModelRunner.load("act", root, device=device, model_variant=model_variant, autotune=autotune)
        return cls(config, adapter, model_runner)

    def __init__(self, config, adapter, model_runner):
        self.model_runner = model_runner
        self.image_key = next(k for k, v in config["input_features"].items() if v["type"] == "VISUAL")
        stats = adapter["stats"]["norm_stats"]
        self.q01 = np.asarray(stats["state"]["q01"], dtype=np.float32)
        self.state_range = np.maximum(np.asarray(stats["state"]["q99"], dtype=np.float32) - self.q01, np.float32(1e-6))
        if self.q01.shape != (33,) or self.state_range.shape != (33,) or not np.isfinite(self.q01).all() or not np.isfinite(self.state_range).all():
            raise ValueError("Invalid state normalization statistics")
        self.mean = np.array([.485, .456, .406], dtype=np.float32)[:, None, None]
        self.std = np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
        if adapter["config"]["policy"]["action_space"] != "lxry":
            raise ValueError("Native ACT profile currently supports lxry only")
        self.action_mean = np.zeros(2, dtype=np.float32)
        self.action_std = np.ones(2, dtype=np.float32)
        self.action_horizon, self.action_dim = 8, 2
        self.metadata = {"model_type": "act", "model_variant": getattr(model_runner, "model_variant", None),
                         "action_horizon": 8, "prediction_horizon": 50, "action_dim": 2,
                         "image_shape": [360, 640, 3], "state_dim": 33, "image_keys": [self.image_key],
                         "state_key": "observation.state", "action_units": "lxry_control_units"}
        self._closed = False

    def infer(self, observation, *, noise=None, **kwargs):
        if self._closed:
            raise RuntimeError("ACT policy is closed")
        if noise is not None:
            raise ValueError("ACT uses zero latent; explicit stochastic noise is unsupported")
        if kwargs:
            raise TypeError(f"Unsupported ACT inference arguments: {sorted(kwargs)}")
        start = time.perf_counter()
        image = np.asarray(observation[self.image_key])
        if image.dtype != np.uint8 or image.shape != (360, 640, 3):
            raise ValueError("ACT image must be RGB uint8 HWC [360,640,3]")
        state = np.asarray(observation["observation.state"], dtype=np.float32)
        if state.shape != (33,) or not np.isfinite(state).all():
            raise ValueError("ACT state must be finite [33]")
        image = np.ascontiguousarray(image)
        state = np.ascontiguousarray((2 * (state - self.q01) / self.state_range - 1)[None])
        begin_model = time.perf_counter()
        normalized = self.model_runner.infer_pixels(image, state, self.mean[:, 0, 0].tolist(), self.std[:, 0, 0].tolist())
        model_ms = (time.perf_counter() - begin_model) * 1000
        prediction = normalized * self.action_std + self.action_mean
        if prediction.shape != (50, 2) or not np.isfinite(prediction).all():
            raise ValueError("invalid native ACT prediction")
        return {"actions": prediction[:8].copy(), "prediction": prediction,
                "normalized_actions": normalized,
                "timing": {"model_ms": model_ms, "total_ms": (time.perf_counter() - start) * 1000}}

    def prepare(self, observation, *, mode="graph"):
        self.infer(observation)
        return self.model_runner.prepare_tensors(mode)

    def reset(self):
        """No queue or temporal ensemble in this supported fixed profile."""

    def close(self):
        self._closed = True
        self.model_runner = None
