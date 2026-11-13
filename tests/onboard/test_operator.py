"""Operator safety core: envelope, ramp, arming rules, deadman, stale state, link loss, STOP latch."""

import math
from pathlib import Path

import pytest

from humanoid_arc_bme.onboard.runtime.operator import ARMED, DISARMED, OperatorLimits, RateLimiter, Supervisor, load_operator_limits, tilt_deg_from_quat
from humanoid_arc_bme.onboard.api.safety import load_limits

ROOT = Path(__file__).resolve().parents[2]
CFG = {"vx_min": -0.3, "vx_max": 0.5, "vy_max": 0.2, "wz_max": 0.5, "accel": {"vx": 0.5, "vy": 0.5, "wz": 1.0},
       "deadman_timeout_s": 0.3, "heartbeat_min_hz": 5.0, "state_stale_s": 0.1, "link_loss_s": 0.5, "arm_challenge_ttl_s": 30,
       "max_tilt_deg": 20.0}


class FixedRng:
    @staticmethod
    def choice(seq):
        return seq[0]


def sup(**over):
    log = []
    s = Supervisor(OperatorLimits.from_dict({**CFG, **over}), on_transition=log.append, rng=FixedRng)
    return s, log


def feed(s, t0, t1, hz=10.0, target=(0.0, 0.0, 0.0)):
    t = t0
    while t <= t1 + 1e-9:
        s.heartbeat(t, target)
        t += 1.0 / hz
    return t1


def armed_sup(t=0.0, **over):
    s, log = sup(**over)
    feed(s, t - 1.0, t)
    code = s.request_arm(t, "fake")
    ok, why = s.confirm_arm(t, f"ARM {code}", "fake", True)
    assert ok, why
    return s, log


# ---------------------------------------------------------------- limits

def test_config_file_has_the_first_test_defaults_and_s1_narrows_them():
    raw = load_operator_limits(ROOT / "config" / "g1_limits.yaml")
    assert (raw.vx_min, raw.vx_max, raw.vy_max, raw.wz_max) == (-0.1, 0.2, 0.1, 0.3)  # = first-walk protocol caps (UI-05 list)
    s1 = load_limits(ROOT / "config" / "g1_limits.yaml").base_vel_max
    eff = load_operator_limits(ROOT / "config" / "g1_limits.yaml", s1)
    assert (eff.vx_min, eff.vx_max, eff.vy_max, eff.wz_max) == (-0.1, 0.2, 0.1, 0.3)
    assert eff.vx_max <= s1[0] and eff.vy_max <= s1[1] and eff.wz_max <= s1[2]


def test_clip_and_bad_values():
    lim = OperatorLimits.from_dict(CFG)
    assert lim.clip(2.0, -1.0, 9.0) == (0.5, -0.2, 0.5)
    assert lim.clip(-2.0, 0.1, -9.0) == (-0.3, 0.1, -0.5)
    with pytest.raises(ValueError):
        lim.clip(math.nan, 0, 0)
    for bad in ({"vx_max": math.inf}, {"vx_min": 0.1}, {"deadman_timeout_s": 5.0}, {"heartbeat_min_hz": 0}, {"vy_max": "x"}):
        with pytest.raises(ValueError):
            OperatorLimits.from_dict({**CFG, **bad})
    with pytest.raises(ValueError):
        OperatorLimits.from_dict({k: v for k, v in CFG.items() if k != "accel"})


def test_curvature_keeps_the_path_and_lowers_speed():
    lim = OperatorLimits.from_dict(CFG)
    vx, vy, wz = lim.clip_curvature(0.4, 1.0)  # 0.4 rad/s < 0.5: unchanged
    assert (vx, vy, wz) == pytest.approx((0.4, 0.0, 0.4))
    vx, vy, wz = lim.clip_curvature(0.5, 2.0)  # 1.0 rad/s > 0.5: speed 0.25, same curvature
    assert vx == pytest.approx(0.25) and wz == pytest.approx(0.5) and wz / vx == pytest.approx(2.0)
    assert lim.clip_curvature(5.0, 0.0) == (0.5, 0.0, 0.0)


def test_rate_limiter_ramps_per_axis():
    r = RateLimiter((0.5, 0.5, 1.0))
    out = [r.step((0.5, -0.2, 0.5), 0.02) for _ in range(5)]
    assert out[0] == pytest.approx((0.01, -0.01, 0.02))
    assert out[4] == pytest.approx((0.05, -0.05, 0.1))
    for _ in range(100):
        r.step((0.5, -0.2, 0.5), 0.02)
    assert r.value == (0.5, -0.2, 0.5)  # exact once in reach
    with pytest.raises(ValueError):
        r.step((0, 0, 0), 0.0)


# ---------------------------------------------------------------- arming rules

def test_starts_disarmed_and_commands_zero():
    s, _ = sup()
    assert s.state == DISARMED and not s.armed
    s.heartbeat(0.0, (0.5, 0.0, 0.0))
    assert s.command(0.02) == (0.0, 0.0, 0.0)


def test_arming_needs_request_code_preflight_and_heartbeat():
    s, log = sup()
    feed(s, 0.0, 1.0)
    ok, why = s.confirm_arm(1.0, "ARM AAAA", "fake", True)
    assert not ok and "no arming request" in why  # confirmation without a request
    code = s.request_arm(1.0, "fake")
    assert code == "AAAA"
    ok, why = s.confirm_arm(1.0, "arm aaaa", "fake", True)
    assert not ok and "does not match" in why
    ok, why = s.confirm_arm(1.0, f"ARM {code}", "fake", True)  # the wrong try consumed the code
    assert not ok and "no arming request" in why
    s.request_arm(1.0, "fake")
    ok, why = s.confirm_arm(1.0, "ARM AAAA", "fake", False, ["state_fresh"])
    assert not ok and "preflight" in why and "state_fresh" in why
    s.request_arm(1.0, "fake")
    ok, why = s.confirm_arm(1.0, "ARM AAAA", "vendor", True)
    assert not ok and "controller changed" in why
    s.request_arm(1.0, "fake")
    ok, why = s.confirm_arm(40.0, "ARM AAAA", "fake", True)
    assert not ok and ("expired" in why)
    assert s.state == DISARMED
    assert sum(r["reason"] == "arm_refused" for r in log) == 6


def test_arming_refused_with_slow_heartbeat():
    s, _ = sup()
    feed(s, 0.0, 1.0, hz=3.0)
    s.request_arm(1.0, "fake")
    ok, why = s.confirm_arm(1.0, "ARM AAAA", "fake", True)
    assert not ok and "heartbeat" in why


def test_arm_then_command_ramps_from_zero():
    s, log = armed_sup(1.0)
    assert s.armed and log[-1]["to"] == ARMED and log[-1]["reason"] == "operator_confirmed"
    assert s.command(0.02) == (0.0, 0.0, 0.0)  # arming resets the target
    s.heartbeat(1.02, (0.5, 0.0, 0.0), held=True)  # UI-03: input held across arming: ignored until released once
    assert s.command(0.02) == (0.0, 0.0, 0.0)
    s.heartbeat(1.04, (0.0, 0.0, 0.0), held=False)  # neutral (released) first
    s.heartbeat(1.06, (0.5, 0.0, 0.0), held=True)  # a new press
    assert s.command(0.02) == pytest.approx((0.01, 0.0, 0.0))


# ---------------------------------------------------------------- UI-03 hold-to-run, UI-04 validation

def test_heartbeat_alone_never_authorises_motion():
    s, _ = armed_sup(1.0)
    s.heartbeat(1.02, (0.0, 0.0, 0.0))
    for k in range(50):  # 10 Hz tab-alive heartbeats with a stale (slider) value but no held input
        s.heartbeat(1.1 + 0.1 * k, (0.4, 0.1, 0.3))
        assert s.target == (0.0, 0.0, 0.0) and s.command(0.1) == (0.0, 0.0, 0.0)
    assert s.armed  # the link is alive: armed and standing


def test_release_returns_to_zero_and_re_arming_needs_neutral_again():
    s, _ = armed_sup(1.0)
    s.heartbeat(1.02, (0.0, 0.0, 0.0), held=False)
    assert s.heartbeat(1.04, (0.3, 0.0, 0.0), held=True)[0] == 0.3
    assert s.heartbeat(1.06, (0.3, 0.0, 0.0), held=False) == (0.0, 0.0, 0.0)  # released: zero at once
    s.disarm(1.1)
    feed(s, 1.1, 2.0)
    code = s.request_arm(2.0, "fake")
    assert s.confirm_arm(2.0, f"ARM {code}", "fake", True)[0]
    assert s.heartbeat(2.02, (0.3, 0.0, 0.0), held=True) == (0.0, 0.0, 0.0)  # held across the new arming: ignored


@pytest.mark.parametrize("bad", [(math.nan, 0, 0), (0, math.inf, 0), (0, 0, -math.inf), ("0.1", 0, 0), (True, 0, 0),
                                 (None, 0, 0), (0.6, 0, 0), (-0.4, 0, 0), (0, 0.3, 0), (0, 0, 0.9)])
def test_malformed_or_out_of_range_heartbeat_is_rejected_and_does_not_feed_the_deadman(bad):
    s, _ = armed_sup(1.0)
    s.heartbeat(1.0, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        s.heartbeat(1.25, bad, held=True)
    assert s.heartbeat_age(1.25) == pytest.approx(0.25)
    assert s.check(1.31, state_age=0.01) == "deadman_timeout"


def test_held_flag_must_be_a_bool_and_set_target_needs_the_held_input():
    s, _ = armed_sup(1.0)
    with pytest.raises(ValueError):
        s.heartbeat(1.01, (0.0, 0.0, 0.0), held="yes")
    s.heartbeat(1.02, (0.0, 0.0, 0.0), held=False)
    assert s.set_target((0.2, 0.0, 0.0), "waypoints", now=1.03) == (0.0, 0.0, 0.0)  # not held: zero
    s.heartbeat(1.04, (0.0, 0.0, 0.0), held=True)
    assert s.set_target((0.2, 0.0, 0.0), "waypoints", now=1.05)[0] == pytest.approx(0.2)
    assert s.set_target((0.2, 0.0, 0.0), "waypoints", now=1.40) == (0.0, 0.0, 0.0)  # held heartbeat too old
    with pytest.raises(ValueError):
        s.set_target((math.nan, 0.0, 0.0), "waypoints", now=1.05)


# ---------------------------------------------------------------- deadman, stale, link, tilt, fault

def test_deadman_timeout_disarms_after_300ms():
    s, log = armed_sup(1.0)
    s.heartbeat(1.0, (0.3, 0.0, 0.0))
    assert s.check(1.29, state_age=0.01) is None and s.armed
    assert s.check(1.31, state_age=0.01) == "deadman_timeout"
    assert not s.armed and log[-1]["reason"] == "deadman_timeout"
    assert s.command(0.02) == (0.0, 0.0, 0.0) and s.target == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("age,reason", [(0.2, "state_stale"), (0.6, "link_lost"), (None, "link_lost")])
def test_stale_state_and_link_loss_disarm(age, reason):
    s, _ = armed_sup(1.0)
    assert s.check(1.05, state_age=age) == reason and not s.armed


def test_tilt_and_controller_fault_disarm():
    s, _ = armed_sup(1.0)
    assert s.check(1.01, state_age=0.0, tilt_deg=25.0) == "tilt_limit"
    s, _ = armed_sup(1.0)
    assert s.check(1.01, state_age=0.0, controller_fault="rpc 3104") == "controller_fault:rpc 3104"


# ---------------------------------------------------------------- stop

def test_stop_latches_and_blocks_arming_until_reset():
    s, log = armed_sup(1.0)
    s.stop(1.1, "ui")
    assert not s.armed and s.stop_latched and log[-1]["reason"] == "stop:ui"
    with pytest.raises(PermissionError):
        s.request_arm(1.2, "fake")
    s.stop(1.3, "ui")  # a second STOP while disarmed is still logged
    assert log[-1]["reason"] == "stop:ui"
    s.reset_stop(1.4)
    assert not s.stop_latched and log[-1]["reason"] == "stop_reset"
    feed(s, 1.0, 2.0)
    code = s.request_arm(2.0, "fake")
    assert s.confirm_arm(2.0, f"ARM {code}", "fake", True)[0]


def test_every_transition_is_recorded_with_time_and_reason():
    s, log = armed_sup(1.0)
    s.disarm(1.5, "operator")
    assert [r["reason"] for r in log] == ["arm_requested", "operator_confirmed", "operator"]
    assert all({"t", "from", "to", "reason", "controller"} <= r.keys() for r in log)


def test_tilt_from_quat():
    assert tilt_deg_from_quat((1, 0, 0, 0)) == pytest.approx(0.0)
    a = math.radians(30)
    assert tilt_deg_from_quat((math.cos(a / 2), math.sin(a / 2), 0, 0)) == pytest.approx(30.0)
