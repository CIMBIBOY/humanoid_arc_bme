"""humanoid_arc_bme/onboard/runtime/obs_builders.py: synthetic checks (always) and sim parity on recorded steps (when the export dump exists).

The parity data is written by tools/training/export_lowerbody_onnx.py (closedloop.npz + layout.json, 16 envs x 300 steps of the
Play env, torch-driven). Set HUMANOID_ARC_OBS_DUMPS to a comma-separated list of export dirs to check others.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest

from humanoid_arc_bme.onboard.runtime.obs_builders import LowerBodyLayout, LowerBodyObsBuilder, quat_rotate_inverse
from humanoid_arc_bme.onboard.api.contract import Imu, RobotState

ROOT = Path(__file__).resolve().parents[2]
DUMPS = [Path(p) for p in os.environ.get("HUMANOID_ARC_OBS_DUMPS", "").split(",") if p] or [
    ROOT / "outputs" / "W4" / "onnx" / name for name in ("deployH", "deployE", "rough4")]


def _layout(hist=3, lin_vel=False, clamp=True):
    names = ("j0", "j1", "j2")
    return LowerBodyLayout(obs_joint_names=names, obs_default_pos=np.array([0.1, 0.0, -0.1]), action_joint_names=("j0", "j2"),
                           action_offset=np.array([0.1, -0.1]), action_scale=0.25,
                           raw_lo=np.array([-1.0, -2.0]) if clamp else None, raw_hi=np.array([1.0, 2.0]) if clamp else None,
                           hist=hist, lin_vel=lin_vel)


def _state(t, names=("j2", "j0", "j1", "extra"), quat=(1.0, 0.0, 0.0, 0.0)):
    q = tuple(0.1 * t + i for i in range(len(names)))
    return RobotState(stamp=t, joint_names=names, joint_pos=q, joint_vel=tuple(-x for x in q), base_vel=(0.0, 0.0, 0.0),
                      imu=Imu(quat, (0.01 * t, 0.0, 0.0), (0.0, 0.0, 9.81)), wrist_poses={})


def test_gravity_matches_rotation():
    # pitched 90 deg about y: gravity (0,0,-1) seen in the body frame is (+1, 0, 0)
    q = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)
    np.testing.assert_allclose(quat_rotate_inverse(q, (0, 0, -1)), (1.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(quat_rotate_inverse((1, 0, 0, 0), (0, 0, -1)), (0, 0, -1))


def test_history_fill_order_and_joint_lookup_by_name():
    b = LowerBodyObsBuilder(_layout(hist=3))
    assert b.obs_dim == 4 + 3 * (3 + 3 + 3 + 3 + 2)
    o0 = b.build(_state(0), (0.0, 0.0), (0.5, 0, 0, 0.7))
    # first build fills every slot with frame 0: gyro x history = (0, 0, 0)
    np.testing.assert_allclose(o0[4:13], [0, 0, 0] * 3)
    b.build(_state(1), (0.1, 0.2), (0.5, 0, 0, 0.7))
    o2 = b.build(_state(2), (0.3, 0.4), (0.5, 0, 0, 0.7))
    np.testing.assert_allclose(o2[4:13:3], [0.0, 0.01, 0.02], atol=1e-7)  # gyro x: oldest first
    jp = o2[4 + 9 + 9:4 + 9 + 9 + 9]  # joint_pos history: 3 frames x 3 joints (j0, j1, j2 order, not the state's order)
    st = _state(2)
    want = np.array([st.joint_pos[1], st.joint_pos[2], st.joint_pos[0]]) - np.array([0.1, 0.0, -0.1])
    np.testing.assert_allclose(jp[6:9], want, atol=1e-6)
    np.testing.assert_allclose(o2[-2:], [0.3, 0.4], atol=1e-7)  # newest last action
    b.reset()
    o3 = b.build(_state(3), (0.0, 0.0), (0, 0, 0, 0.7))
    np.testing.assert_allclose(o3[4:13:3], [0.03] * 3, atol=1e-7)


def test_decode_clamps_and_maps_to_targets():
    b = LowerBodyObsBuilder(_layout())
    a, tgt = b.decode([5.0, -0.5])
    np.testing.assert_allclose(a, [1.0, -0.5])
    np.testing.assert_allclose(tgt, [0.1 + 0.25, -0.1 - 0.125])


def test_lin_vel_layout_needs_a_velocity():
    b = LowerBodyObsBuilder(_layout(hist=1, lin_vel=True))
    with pytest.raises(ValueError):
        b.build(_state(0), (0.0, 0.0), (0, 0, 0, 0.7))
    assert b.build(_state(0), (0.0, 0.0), (0, 0, 0, 0.7), base_lin_vel=(0.1, 0, 0)).shape == (b.obs_dim,)


def test_missing_joint_is_an_error():
    b = LowerBodyObsBuilder(_layout())
    with pytest.raises(ValueError):
        b.build(_state(0, names=("j0", "j1")), (0.0, 0.0), (0, 0, 0, 0.7))


def _parity(d: Path):
    lay = LowerBodyLayout.from_json(d / "layout.json")
    z = np.load(d / "closedloop.npz")
    names = tuple(__import__("json").loads((d / "layout.json").read_text())["all_joint_names"])
    t_len, n_env = z["obs"].shape[:2]
    worst, n = 0.0, 0
    for e in range(n_env):
        b = LowerBodyObsBuilder(lay)
        for t in range(t_len):
            if z["dones"][t, e]:
                b.reset()
            q = z["root_quat_w"][t, e].astype(np.float64)
            st = RobotState(stamp=0.02 * t, joint_names=names, joint_pos=tuple(z["joint_pos"][t, e]),
                            joint_vel=tuple(z["joint_vel"][t, e]), base_vel=(0.0, 0.0, 0.0),
                            imu=Imu(tuple(q / np.linalg.norm(q)), tuple(z["root_ang_vel_b"][t, e]), (0.0, 0.0, 9.81)),
                            wrist_poses={})
            o = b.build(st, z["last_action"][t, e], z["command"][t, e],
                        base_lin_vel=z["root_lin_vel_b"][t, e] if lay.lin_vel else None)
            worst = max(worst, float(np.abs(o - z["obs"][t, e]).max()))
            n += 1
    return worst, n


@pytest.mark.parametrize("d", DUMPS, ids=lambda p: p.name)
def test_sim_parity(d):
    if not (d / "closedloop.npz").exists() or "joint_pos" not in np.load(d / "closedloop.npz").files:
        pytest.skip(f"no raw-state dump in {d} (run tools/training/export_lowerbody_onnx.py)")
    worst, n = _parity(d)
    print(f"{d.name}: {n} observations, max |builder - sim| {worst:.2e}")
    assert worst < 1e-5
