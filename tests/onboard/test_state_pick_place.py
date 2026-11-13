"""Pick-place state machine (skills/state_pick_place.py). No Isaac Sim."""

import pytest

from humanoid_arc_bme.onboard.skills.state_pick_place import (
    COMMANDS,
    DEFAULT_TIMEOUTS,
    TRANSITIONS,
    Observation,
    PickPlaceMachine,
    State,
)

DT = 0.1


def run(machine, script, t0=0.0):
    """script: list of (duration_s, Observation). Returns the visited states."""
    t, visited = t0, [machine.state]
    for dur, obs in script:
        end = t + dur
        while t < end and not machine.terminal:
            machine.step(t, obs)
            if machine.state != visited[-1]:
                visited.append(machine.state)
            t = round(t + DT, 10)
    return visited


# One observation per phase, each making exactly the leaving predicate true.
O = Observation
SEEN = O(object_seen=True)
PRE = O(pregrasp_reached=True)
GRASP = O(grasp_pose_reached=True)
CLOSED = O(closure_confirmed=True, object_held=True)
LIFTED = O(lifted=True, object_held=True)
AT_PLACE = O(place_pose_reached=True, object_held=True)
PLACED = O(placed=True, object_held=True)
OPEN = O(hand_open=True)
CLEAR = O(hand_clear=True)
NOTHING = O()
HELD = O(object_held=True)

NOMINAL = [(0.2, SEEN), (0.2, PRE), (0.2, GRASP), (0.2, CLOSED), (0.2, LIFTED),
           (0.2, AT_PLACE), (0.2, PLACED), (0.2, OPEN), (0.2, CLEAR)]


def test_nominal_reaches_done_through_every_state_in_order():
    m = PickPlaceMachine()
    visited = run(m, NOMINAL)
    assert visited == [State.LOCATE, State.PREGRASP, State.APPROACH, State.CLOSE, State.LIFT,
                       State.CARRY, State.PLACE, State.RELEASE, State.WITHDRAW, State.DONE]
    assert m.reason == ""


def test_object_not_seen_fails_after_retries_with_reason():
    m = PickPlaceMachine()
    run(m, [(30.0, NOTHING)])
    assert m.state is State.FAILED and m.reason == "timeout:LOCATE"
    assert m.retries[State.LOCATE] == 2


def test_closure_never_confirmed_fails_in_close():
    m = PickPlaceMachine()
    run(m, [(0.2, SEEN), (0.2, PRE), (0.2, GRASP), (30.0, NOTHING)])
    assert m.state is State.FAILED and m.reason == "timeout:CLOSE"


@pytest.mark.parametrize("state,upto", [
    (State.LIFT, [SEEN, PRE, GRASP, CLOSED]),
    (State.CARRY, [SEEN, PRE, GRASP, CLOSED, LIFTED]),
    (State.PLACE, [SEEN, PRE, GRASP, CLOSED, LIFTED, AT_PLACE]),
])
def test_object_lost_while_holding_fails_after_retries(state, upto):
    m = PickPlaceMachine(max_retries=0)
    run(m, [(0.2, o) for o in upto] + [(1.0, NOTHING)])
    assert m.state is State.FAILED and m.reason == f"object_dropped:{state.value}"


def test_drop_then_regrasp_succeeds_with_one_retry():
    m = PickPlaceMachine()
    script = [(0.2, o) for o in (SEEN, PRE, GRASP, CLOSED, LIFTED)]
    script += [(0.2, NOTHING)]  # dropped during CARRY -> LOCATE
    script += NOMINAL
    visited = run(m, script)
    assert m.state is State.DONE
    assert m.retries[State.CARRY] == 1
    assert visited.count(State.LOCATE) == 2


@pytest.mark.parametrize("state", list(TRANSITIONS))
def test_timeout_in_each_state(state):
    """Stall in `state` by showing it everything except its leaving predicate."""
    path = [State.LOCATE, State.PREGRASP, State.APPROACH, State.CLOSE, State.LIFT,
            State.CARRY, State.PLACE, State.RELEASE, State.WITHDRAW]
    good = [SEEN, PRE, GRASP, CLOSED, LIFTED, AT_PLACE, PLACED, OPEN, CLEAR]
    i = path.index(state)
    stall = O(object_held=state in (State.LIFT, State.CARRY, State.PLACE))
    m = PickPlaceMachine(max_retries=1)
    run(m, [(0.2, o) for o in good[:i]] + [(100.0, stall)])
    assert m.state is State.FAILED and m.reason == f"timeout:{state.value}"
    assert m.retries[state] == 1


def test_timeout_boundary_is_strict():
    m = PickPlaceMachine(max_retries=0)
    m.step(0.0, NOTHING)
    m.step(DEFAULT_TIMEOUTS[State.LOCATE], NOTHING)
    assert m.state is State.LOCATE
    m.step(DEFAULT_TIMEOUTS[State.LOCATE] + 0.01, NOTHING)
    assert m.state is State.FAILED


def test_terminal_states_never_change():
    m = PickPlaceMachine()
    run(m, NOMINAL)
    m.step(99.0, NOTHING)
    assert m.state is State.DONE
    f = PickPlaceMachine(max_retries=0)
    run(f, [(30.0, NOTHING)])
    f.step(99.0, SEEN)
    assert f.state is State.FAILED


def test_commands_follow_state():
    m = PickPlaceMachine()
    assert m.command().hand_closure == 0.0
    run(m, [(0.2, o) for o in (SEEN, PRE, GRASP)])
    assert m.state is State.CLOSE and m.command().hand_closure == 1.0
    assert set(COMMANDS) == set(State)


def test_waypoints_exist_in_timer_script():
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "humanoid_arc_bme/skills/scripted_pick_place.py").read_text()
    wps = None
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "WAYPOINTS":
            wps = ast.literal_eval(node.value)
    assert wps and all(w in wps for w, _ in COMMANDS.values())


def test_rejects_bad_config_and_time_reversal():
    with pytest.raises(ValueError):
        PickPlaceMachine(timeouts={State.LOCATE: 1.0})
    with pytest.raises(ValueError):
        PickPlaceMachine(max_retries=-1)
    m = PickPlaceMachine()
    m.step(5.0, NOTHING)
    with pytest.raises(ValueError):
        m.step(4.0, NOTHING)
