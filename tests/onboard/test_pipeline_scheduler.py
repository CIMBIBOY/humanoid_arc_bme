"""W2 gate: scheduler orders unload-before-load, enforces budget, rolls back, tracks peak, stops cleanly."""

import json

import pytest

from humanoid_arc_bme.onboard.pipeline.scheduler import BudgetError, PhaseScheduler


class Fake:
    def __init__(self, name, est, calls, fail_load=False, fail_unload=False):
        self.name, self.est_bytes, self.calls = name, est, calls
        self.fail_load, self.fail_unload = fail_load, fail_unload
        self.loaded = False

    def load(self):
        self.calls.append(f"load:{self.name}")
        if self.fail_load:
            raise OSError("boom")
        self.loaded = True

    def unload(self):
        self.calls.append(f"unload:{self.name}")
        if self.fail_unload:
            raise OSError("stuck")
        self.loaded = False


def make(budget=100, probe=None, max_log=1000, **kw):
    calls = []
    wbc, costmap, depth, grasp, perc = (Fake(n, s, calls, **kw.get(n, {})) for n, s in
                                        [("wbc", 30), ("costmap", 20), ("depth", 10), ("grasp", 40), ("perc", 15)])
    s = PhaseScheduler([wbc], {"walk": [costmap, depth], "grasp": [grasp, perc, depth]}, budget, probe, max_log)
    return s, calls, dict(wbc=wbc, costmap=costmap, depth=depth, grasp=grasp, perc=perc)


def test_switch_before_start_raises():
    s, calls, _ = make()
    with pytest.raises(RuntimeError):
        s.switch("walk")
    assert calls == []


def test_unload_before_load_order_and_shared_not_reloaded():
    s, calls, m = make()
    s.start()
    s.switch("walk")
    calls.clear()
    s.switch("grasp")
    assert calls == ["unload:costmap", "load:grasp", "load:perc"]  # depth shared: untouched
    assert m["depth"].loaded and not m["costmap"].loaded and m["wbc"].loaded
    assert s.phase == "grasp"


def test_budget_rejection_leaves_state_untouched():
    s, calls, m = make(budget=90)  # grasp needs 30+40+15+10=95
    s.start()
    s.switch("walk")
    calls.clear()
    with pytest.raises(BudgetError):
        s.switch("grasp")
    assert calls == [] and s.phase == "walk" and m["costmap"].loaded
    with pytest.raises(ValueError, match="unknown phase"):
        s.switch("nope")
    assert calls == [] and s.phase == "walk"


def test_load_failure_rolls_back_and_keeps_resident():
    s, calls, m = make(perc={"fail_load": True})
    s.start()
    s.switch("walk")
    with pytest.raises(OSError):
        s.switch("grasp")
    assert s.phase is None
    assert m["wbc"].loaded
    assert not m["grasp"].loaded and not m["perc"].loaded and not m["costmap"].loaded
    assert not m["depth"].loaded  # the shared model is also unloaded: safe state is resident-only
    assert [k for k, v in m.items() if v.loaded] == ["wbc"]


def test_stop_idempotent_and_unloads_all_despite_failure():
    s, calls, m = make(costmap={"fail_unload": True})
    s.start()
    s.switch("walk")
    with pytest.raises(RuntimeError, match="costmap"):
        s.stop()
    assert not m["wbc"].loaded and not m["depth"].loaded and m["costmap"].loaded
    assert s.phase is None
    m["costmap"].fail_unload = False
    s.stop()
    assert not m["costmap"].loaded
    n = len(calls)
    s.stop()
    assert len(calls) == n
    with pytest.raises(RuntimeError):
        s.switch("walk")


def test_peak_tracking_with_probe_and_report_json():
    state = {"used": 0}

    def probe():
        return state["used"]

    s, calls, m = make(probe=probe)
    orig = {k: (h.load, h.unload) for k, h in m.items()}
    for k, h in m.items():
        def ld(h=h, f=orig[k][0]):
            f()
            state["used"] += h.est_bytes

        def un(h=h, f=orig[k][1]):
            f()
            state["used"] -= h.est_bytes
        h.load, h.unload = ld, un
    s.start()
    s.switch("walk")
    s.switch("grasp")
    assert s.peak_bytes == 95
    r = s.report()
    json.dumps(r)
    assert r["peak_bytes"] == 95 and r["phase"] == "grasp" and "wbc" in r["loaded"]
    assert r["log"][0] == {"phase": None, "event": "load:wbc", "bytes": 30}


def test_log_is_bounded_dropping_oldest():
    s, _, _ = make(max_log=3)
    s.start()
    s.switch("walk")
    s.switch("grasp")
    log = s.report()["log"]
    assert len(log) == 3
    assert log[-1]["event"] == "load:perc"


def test_construction_errors():
    c = []
    a, b = Fake("a", 1, c), Fake("b", 1, c)
    with pytest.raises(ValueError, match="resident"):
        PhaseScheduler([a], {"p": [a]}, 10)
    with pytest.raises(ValueError, match="duplicate"):
        PhaseScheduler([a, a], {}, 10)
    with pytest.raises(ValueError, match="duplicate"):
        PhaseScheduler([a], {"p": [b, b]}, 10)
    with pytest.raises(BudgetError):
        PhaseScheduler([Fake("big", 20, c)], {}, 10)


def test_failed_switch_leaves_only_resident_loaded_and_next_switch_works():
    s, calls, m = make()
    s.start()
    s.switch("walk")
    m["grasp"].fail_load = True
    with pytest.raises(OSError):
        s.switch("grasp")
    assert s.phase is None
    assert [k for k, v in m.items() if v.loaded] == ["wbc"]  # old-phase models are not left behind
    m["grasp"].fail_load = False
    s.switch("walk")
    assert s.phase == "walk" and m["costmap"].loaded and m["depth"].loaded and not m["grasp"].loaded


def test_stale_model_from_failed_rollback_is_cleaned_by_next_switch():
    s, calls, m = make()
    s.start()
    s.switch("walk")
    m["grasp"].fail_load = True
    m["costmap"].fail_unload = True  # rollback cannot unload it either
    with pytest.raises(OSError):
        s.switch("grasp")
    assert m["costmap"].loaded and s.phase is None
    m["grasp"].fail_load = False
    m["costmap"].fail_unload = False
    s.switch("grasp")
    assert not m["costmap"].loaded and m["grasp"].loaded


def test_start_probe_failure_after_load_unloads_that_model():
    calls = []
    a, b = Fake("a", 1, calls), Fake("b", 1, calls)
    n = {"k": 0}

    def probe():
        n["k"] += 1
        if n["k"] == 2:  # b.load() succeeded, then the probe raises
            raise RuntimeError("probe")
        return 0

    s = PhaseScheduler([a, b], {}, 10, probe)
    with pytest.raises(RuntimeError, match="probe"):
        s.start()
    assert not a.loaded and not b.loaded
    assert s.report()["started"] is False


def test_switch_probe_failure_after_load_returns_to_resident_only():
    calls = []
    r, p = Fake("r", 1, calls), Fake("p", 1, calls)
    state = {"boom": False}

    def probe():
        if state["boom"] and p.loaded:
            raise RuntimeError("probe")
        return 0

    s = PhaseScheduler([r], {"ph": [p]}, 10, probe)
    s.start()
    state["boom"] = True
    with pytest.raises(RuntimeError, match="probe"):
        s.switch("ph")
    assert not p.loaded and r.loaded and s.phase is None


def test_same_phase_switch_is_noop_and_unknown_phase_is_valueerror():
    s, calls, m = make()
    s.start()
    s.switch("walk")
    calls.clear()
    s.switch("walk")
    assert calls == [] and s.phase == "walk"
    with pytest.raises(ValueError):
        s.switch("nope")


def test_constructor_rejects_bad_scalars():
    for kw in (dict(budget_bytes=0), dict(budget_bytes=10, max_log=0)):
        with pytest.raises(ValueError):
            PhaseScheduler([], {}, **kw)
