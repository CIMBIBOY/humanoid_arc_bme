"""Terms for the lower-body training env: a height command and scripted arm motion."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class VelocityHeightCommand(UniformVelocityCommand):
    """[vx, vy, wz, pelvis height]: the Agile policy's command layout."""

    cfg: VelocityHeightCommandCfg

    def __init__(self, cfg: VelocityHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.height_command = torch.full((self.num_envs,), sum(cfg.height_range) / 2, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.vel_command_b, self.height_command.unsqueeze(-1)], dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        self.height_command[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.height_range)


@configclass
class VelocityHeightCommandCfg(UniformVelocityCommandCfg):
    class_type: type = VelocityHeightCommand
    height_range: tuple[float, float] = (0.6, 0.78)


def track_pelvis_height_exp(env: ManagerBasedEnv, command_name: str, std: float) -> torch.Tensor:
    """exp(-(pelvis height - commanded height)^2 / std^2); the ground is at z = 0."""
    z = env.scene["robot"].data.root_pos_w[:, 2]
    return torch.exp(-torch.square(z - env.command_manager.get_term(command_name).height_command) / std**2)


class ArmMotionAction(ActionTerm):
    """Drives every non-leg joint (waist, arms, hands) along a trajectory; takes no policy action.

    Each episode plays either a recorded demo (the joint targets the scripted pick-place produced through
    Pink IK) at a random speed and start, or a random sum of sinusoids around the default pose.
    """

    cfg: ArmMotionActionCfg

    def __init__(self, cfg: ArmMotionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._joint_ids, names = self._asset.find_joints(cfg.joint_names)
        u = len(self._joint_ids)
        self._default = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._lo, self._hi = limits[..., 0], limits[..., 1]
        self._range = self._hi - self._lo

        self._demos, self._demo_len = self._load_demos(cfg.demo_files)
        n = self.num_envs
        self._t = torch.zeros(n, device=self.device)  # seconds along the trajectory
        self._speed = torch.ones(n, device=self.device)
        self._use_demo = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._demo_id = torch.zeros(n, dtype=torch.long, device=self.device)
        self._amp = torch.zeros(n, cfg.num_sines, u, device=self.device)
        self._freq = torch.zeros(n, cfg.num_sines, u, device=self.device)
        self._phase = torch.zeros(n, cfg.num_sines, u, device=self.device)
        self._walk = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._swing_amp = torch.zeros(n, device=self.device)
        self._swing_freq = torch.ones(n, device=self.device)
        self._swing_phase = torch.zeros(n, device=self.device)
        pitch_l, pitch_r = names.index("left_shoulder_pitch_joint"), names.index("right_shoulder_pitch_joint")
        self._swing_sign = torch.zeros(u, device=self.device)
        self._swing_sign[pitch_l], self._swing_sign[pitch_r] = 1.0, -1.0
        self._target = self._default.clone()
        self._raw = torch.zeros(n, 0, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._target

    def _load_demos(self, files: Sequence[str]):
        if not files or self.cfg.demo_prob <= 0:
            return None, None
        import h5py

        missing = [path for path in files if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                f"arm-motion demos not found: {missing}. Record them with humanoid_arc_core's tools/record_demos.py"
                " (--speed 1.5) or point $HUMANOID_ARC_ARM_DEMOS at existing files."
            )
        demos = []
        for path in files:
            with h5py.File(path, "r") as f:
                for name in f["data"]:
                    demos.append(torch.as_tensor(f["data"][name]["processed_actions"][:], dtype=torch.float32))
        lens = torch.tensor([len(d) for d in demos], device=self.device)
        lib = torch.zeros(len(demos), int(lens.max()), len(self._joint_ids), device=self.device)
        for i, d in enumerate(demos):
            lib[i, : len(d)] = d.to(self.device)[:, self._joint_ids]
            lib[i, len(d) :] = lib[i, len(d) - 1]
        return lib, lens

    def _trajectory(self, t: torch.Tensor, ids: torch.Tensor | slice = slice(None)) -> torch.Tensor:
        """Joint targets at time `t` (seconds) for the envs `ids`."""
        q = self._default[ids] + (
            self._amp[ids] * torch.sin(2 * math.pi * self._freq[ids] * t[:, None, None] + self._phase[ids])
        ).sum(1)
        if self._demos is not None:
            demo_id = self._demo_id[ids]
            length = self._demo_len[demo_id]
            frame = (t / self._env.step_dt) % length  # loop: a demo ends where it started (rest pose)
            i0 = frame.long()
            i1 = (i0 + 1) % length
            w = (frame - i0).unsqueeze(-1)
            q_demo = torch.lerp(self._demos[demo_id, i0], self._demos[demo_id, i1], w)
            q = torch.where(self._use_demo[ids].unsqueeze(-1), q_demo, q)
        swing = self._swing_amp[ids] * torch.sin(2 * math.pi * self._swing_freq[ids] * t + self._swing_phase[ids])
        q = torch.where(self._walk[ids].unsqueeze(-1), self._default[ids] + swing.unsqueeze(-1) * self._swing_sign, q)
        return torch.clamp(q, self._lo[ids], self._hi[ids])

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        k = len(env_ids)
        cfg = self.cfg
        u = len(self._joint_ids)

        def uni(lo, hi, *shape):
            return torch.empty(*shape, device=self.device).uniform_(lo, hi)

        mode = uni(0, 1, k)
        self._walk[env_ids] = mode < cfg.walk_prob
        held = (mode >= cfg.walk_prob) & (mode < cfg.walk_prob + cfg.held_prob)
        self._use_demo[env_ids] = (
            (held | ((mode >= cfg.walk_prob + cfg.held_prob) & (uni(0, 1, k) < cfg.demo_prob)))
            if self._demos is not None
            else False
        )
        self._swing_amp[env_ids] = uni(*cfg.swing_amplitude, k)
        self._swing_freq[env_ids] = uni(*cfg.swing_frequency, k)
        self._swing_phase[env_ids] = uni(0, 2 * math.pi, k)
        if self._demos is not None:
            self._demo_id[env_ids] = torch.randint(0, len(self._demos), (k,), device=self.device)
            length_s = self._demo_len[self._demo_id[env_ids]] * self._env.step_dt
            demo_start = uni(0, 1, k) * length_s * 0.7
        else:
            demo_start = torch.zeros(k, device=self.device)
        self._speed[env_ids] = torch.where(held, torch.zeros(k, device=self.device), uni(*cfg.speed_range, k))
        amp_on = ~self._walk[env_ids] & ~held
        amp_frac = uni(*cfg.amplitude_range, k, cfg.num_sines, 1) * (uni(0, 1, k, cfg.num_sines, u) < 0.6) * amp_on[:, None, None]
        self._amp[env_ids] = amp_frac * self._range[env_ids].unsqueeze(1) / 2 / cfg.num_sines
        self._freq[env_ids] = uni(*cfg.frequency_range, k, cfg.num_sines, u)
        self._phase[env_ids] = uni(0, 2 * math.pi, k, cfg.num_sines, u)
        self._t[env_ids] = torch.where(self._use_demo[env_ids], demo_start, torch.zeros_like(demo_start))

        q0 = self._trajectory(self._t[env_ids], env_ids)
        self._target[env_ids] = q0
        self._asset.write_joint_state_to_sim(q0, torch.zeros_like(q0), joint_ids=self._joint_ids, env_ids=env_ids)
        return {}

    def process_actions(self, actions: torch.Tensor):
        self._t += self._env.step_dt * self._speed
        self._target = self._trajectory(self._t)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._target, joint_ids=self._joint_ids)


@configclass
class ArmMotionActionCfg(ActionTermCfg):
    class_type: type = ArmMotionAction
    joint_names: list[str] = [".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*", "waist_.*", ".*_hand_.*"]
    demo_files: list[str] = []
    demo_prob: float = 0.5  # among envs that are neither walking-arms nor held
    walk_prob: float = 0.3  # arms at the sides with a counter-swing of the shoulder pitch
    held_prob: float = 0.2  # a frozen pose from a demo (manipulating while standing)
    swing_amplitude: tuple[float, float] = (0.1, 0.4)  # rad
    swing_frequency: tuple[float, float] = (0.8, 1.4)  # Hz
    speed_range: tuple[float, float] = (0.6, 1.3)
    num_sines: int = 2
    amplitude_range: tuple[float, float] = (0.2, 1.0)  # fraction of half the joint range, split across the sines
    frequency_range: tuple[float, float] = (0.2, 1.0)  # Hz
