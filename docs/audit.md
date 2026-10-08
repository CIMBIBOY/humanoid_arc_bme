# Audit: building the table-tour demo

A record of what broke and why, from generalizing the single-table demo
(`run_pick_place.py`, already verified working) into the 4-table continuous
tour (`run_table_tour.py`). Kept because the reasoning ("why this looks
right but isn't") is more useful than the diff.

## Bugs found and fixed

**1. Heading convention mismatch (highest severity -- object never moved at
all).** `layout_square()`/`_offset()` computed world direction as
`(cos(heading + pi/2), sin(heading + pi/2))`; `navigator.navigate()` and
`ScriptedPickPlace` both treat `heading` as a plain yaw, `(cos(heading),
sin(heading))`, with no offset. Both used the *same* `stop.heading` value,
inconsistently. Effect: the robot walked to the correct position and
correctly-computed heading, then reached for a grasp point 90 degrees away
from where the object actually was. Looked, from the pelvis-position log
alone, like everything was working (correct waypoints, smooth phase
transitions, stable standing) -- only caught by adding a debug print of
`(object_home, pick, place_target, actual_object_position)` at each
placement, which showed `actual_object_position == object_home` to 3 decimal
places, every stop, every lap: the object had never been touched. Fixed in
`humanoid_arc_bme/envs/g1_table_tour/env_cfg.py::_offset()`.

**2. Object placement drift across laps.** `_place` (the wrist's place
target) was computed from the object's *current* live position each time
`ScriptedPickPlace.reset()` ran, plus a fixed sideways shift. Each lap's
placement became the next lap's pick location, so the object walked an
additional `place_offset` further from its original spot every lap --
unbounded drift, eventually off the table. Fixed by capturing
`self._object_home` once, at construction (before anything moves the
object), and deriving the place target from that fixed point instead of the
live one. `humanoid_arc_core.skills.scripted_pick_place`.

**3. Stale grasp geometry for stops 1..N-1.** `TableTourController.__init__`
constructs one `ScriptedPickPlace` per stop up front; each one's constructor
internally calls `self.reset(all envs)`, using whatever the robot's pose is
*at that moment* -- the spawn pose, for all four, since none of them have
walked anywhere yet. Correct for stop 0 (spawn pose == stop 0's arrival
pose, by construction), silently wrong for stops 1-3, whose pick/place/grasp
targets need the pose the robot actually arrives at, not the spawn pose. The
symptom would have been the same 90-degrees-off grasp as bug #1, just for
stops 1-3 only. Fixed by deferring each stop's real `reset()` to the exact
step the robot arrives there (`TableTourController.compute()`'s
`start_pickplace` handling), not at construction.

**4. Place-offset overshoot off the table edge.** `DEFAULT_PLACE_OFFSET =
(0, 0.25, 0)` shifted the placement sideways -- the table's narrow axis
(0.3m, so ~0.15m clearance from center), from a pick point that's already
~0.08m off-center on that same axis. The 0.25m shift landed the object
beyond the table edge. Physically this reads as a *drop*, not a targeting
error: `TerminationsCfg`'s `object_dropping` (height < 0.5m) fired shortly
after each release, resetting the whole env back to stop 0 before it ever
reached stop 1 -- an apparent infinite loop retrying the first table forever
(9+ attempts observed, same ~0.12m error every time, in a run that predated
fix #5 below). Fixed by shifting along the table's *long* axis (0.6m, ~2x
the clearance) instead: `DEFAULT_PLACE_OFFSET = (0.20, 0, 0)`.

**5. Success-check comparing against the wrong reference point (not a
simulation bug -- a measurement bug that looked like one).** `_pick` and
`_place` are wrist/IK targets, offset from the object by the ~13cm grasp
geometry (`GRASP_OFFSET`) by design -- the same relationship `_pick` has
always had to the object, needed so the wrist reaches to the right spot to
grasp it. The placement *test* code (in both `run_pick_place.py` and
`run_table_tour.py`) compared the object's final position directly against
`_place`, off by exactly that ~13cm offset. This made every single placement
look like a ~0.06-0.25m "failure" -- including on the single-table demo,
previously verified at ~5mm -- even though the underlying grasp/carry/place
behavior was correct the whole time (confirmed: the object tracked the
wrist at a constant ~13cm offset through lift/transport/lower, exactly the
expected grasp geometry). Fixed by exposing `place_target` (the actual
intended object position, not the wrist target) on `ScriptedPickPlace`, and
using that in both scripts' checks instead.

**6. Action-vector width arithmetic error.** `TableTourController.compute()`
allocated `3 + 7 + hand_dim + 7 + 4` (=35) instead of `7 + 7 + hand_dim + 4`
(=32) -- an extra stray `3 +`. Caught immediately: `env.step()` raised a
shape-mismatch error on the first call, before any real testing was possible.

**7. Multi-table scene broke hardcoded single-object references.** The
single-table `ObservationsCfg`/`TerminationsCfg` (inherited by the tour env
unless overridden) reference a scene entity literally named `"object"` --
both a `SceneEntityCfg("object")` param and, for the `object` observation
term, a hardcoded `env.scene["object"]` inside the MDP function itself. The
tour scene has one object per stop (`object_red`, etc.), no bare `"object"`.
Both managers raised `ValueError` at env construction. Fixed with
`TourObservationsCfg`/`TourTerminationsCfg`: the object-position observation
terms are dropped (nothing reads them -- `TableTourController` reads
`env.scene[...]` directly, same as the single-table script), and
`object_dropping` becomes one termination term per stop, generated the same
way the scene's tables/objects are. Verified first, before relying on it,
that IsaacLab's scene/observation/termination managers all skip `None`-valued
config fields (`isinstance(..., None)` guards in each manager's own
source) -- the same mechanism the scene generation already used to remove
the single-table `table`/`object` fields.

## Not a bug, but flagged for later

- `TableTourController.compute()` runs every stop's `ScriptedPickPlace.compute()`
  for all `n` envs every step (even envs elsewhere), discarding the result
  for envs not currently at that stop, and lets that stop's idle phase clock
  free-run in the background. Harmless -- `reset()` always re-zeros phase/`t`
  on the next real arrival -- but wasteful; masking properly (or batching all
  stops' math together) is worth doing before scaling `--num_envs` way up.
- Tried tightening `navigator.py`'s arrival tolerance 4x/3x as a first theory
  for the placement-precision gap (the real cause was the arm's reach, see `../results.md`); verified it changes
  nothing and reverted rather than leave an ineffective change in place with
  a comment implying it helped.

## Process hygiene (not a code bug, an operational one)

During this audit, three of this session's own background Isaac Sim
processes were left running unmonitored for 1-2+ hours after they were no
longer needed: an old demo livestream, a checkpoint-evaluation run whose
outer `timeout` didn't propagate to the inner Python process (only killed
the wrapping shell), and an earlier tour run superseded by a bugfix. Together
they pushed GPU utilization to 93% and made later verification runs
misleadingly slow -- initially read as "the demo might be hanging," when it
was actually just resource-starved. All confirmed killed. Check process and
GPU state (`nvidia-smi`, `ps`) before relying on timing from a run.
