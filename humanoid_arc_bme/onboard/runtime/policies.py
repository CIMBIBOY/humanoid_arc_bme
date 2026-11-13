"""Model policies behind the runtime's `Policy` protocol.

`OnnxLegPolicy` runs an exported lower-body actor (tools/training/export_lowerbody_onnx.py: policy.onnx + layout.json) at the
control rate: RobotState -> obs_builders.LowerBodyObsBuilder -> ONNX (onnxruntime, CPU, lazy import; the `onnx` extra)
-> clamp + decode -> JointTargets for the 12 leg joints. The runtime then passes them through S1 (`filter_joints`).
The command (vx, vy, wz, height) comes from a callable, so a planner, a joystick or a test can drive it.
`session` can be injected (anything with `run(None, {name: x})`), which is how the tests run without onnxruntime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from humanoid_arc_bme.onboard.api.contract import JointTargets

from .obs_builders import LowerBodyLayout, LowerBodyObsBuilder


def ort_session(onnx_path, threads: int = 1):
    import onnxruntime as ort  # lazy: only where the model actually runs

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])


class OnnxLegPolicy:
    joint_policy = True  # the runtime routes this policy's output through S1.filter_joints

    def __init__(self, model_dir, command: Callable[[], tuple], session=None, base_lin_vel: Callable | None = None):
        d = Path(model_dir)
        self.layout = LowerBodyLayout.from_json(d / "layout.json")
        self.builder = LowerBodyObsBuilder(self.layout)
        self.session = session if session is not None else ort_session(d / "policy.onnx")
        self.input_name = self.session.get_inputs()[0].name if hasattr(self.session, "get_inputs") else "obs"
        self.command = command
        self.base_lin_vel = base_lin_vel  # only for the rough4 layout (needs a velocity the G1 does not measure)
        self.joint_names = self.layout.action_joint_names
        self.last_action = np.zeros(len(self.joint_names))
        self.last_obs: np.ndarray | None = None

    def reset(self) -> None:
        self.builder.reset()
        self.last_action = np.zeros(len(self.joint_names))

    def act(self, state, frames=None) -> JointTargets:
        v = None if not self.layout.lin_vel else self.base_lin_vel()
        obs = self.builder.build(state, self.last_action, self.command(), base_lin_vel=v)
        raw = self.session.run(None, {self.input_name: obs[None]})[0][0]
        self.last_action, targets = self.builder.decode(raw)
        self.last_obs = obs
        return JointTargets(self.joint_names, tuple(float(x) for x in targets))
