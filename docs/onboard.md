# The onboard stack: natural-language pick and place on a Unitree G1 (Jetson Orin NX)

Standalone: this repository is everything a university needs. Nothing here imports the private `arc` package (a test
enforces it). The advanced repository replaces parts through the interfaces listed under "Overriding a part".

Status: **validated in a CPU simulator, in unit tests and (2026-11-14) in Isaac Sim with a stand-in locomotion policy. Nothing
has run on the robot.** Every number tagged
UNMEASURED is a placeholder until it has been measured (see "Calibration").

## What it does

    "pick up the red cup"  ->  the robot walks to the table, stands with the cup inside the arm's reach, grasps and lifts it
    "put it down on the left"  ->  places it, returns the arm to rest
    "go forward 2 meters" / "turn left 90 degrees"  ->  walks / turns under the safety guard
    "stop"  ->  always wins, from any state, at any sentence length

No LLM. The sentence goes through a deterministic grammar. An *optional* intent classifier (for example the Jev decision
model, a cloud service) may help with sentences the grammar rejects; it only chooses among six intents, and every number,
object and colour still comes from the grammar (`onboard/command/composite.py`). The robot never depends on it.

## Architecture

    laptop --Tailscale/SSH--> FastAPI command gateway (token, confirm step, STOP without token)   onboard/runtime/command_gateway.py
                                  |
    RealSense (down) --+          v
    Livox (optional) --+--> SensorHub --> RobotStack --------------------------------+          onboard/stack.py
    DLIO pose ---------+     sensors/hub     | perception: YOLO-seg -> scene (class, colour, 3-D position)
    joints + Dex3 feed-+                     | TaskExecutor: Move / Turn / Pick / Place / Stop / Status      onboard/task/
                                             | PipelineRunner: down-camera guard, costmap, speed caps         onboard/pipeline/
                                             v
                                    base (vx, vy, wz)  +  arm / hand joint targets
                                             |
    AutonomyLoop (runtime/autonomy.py) --> OperatorRuntime: arming, hold-to-run deadman, S1 clamp, supervisor
                                             |
                                    GrootPlannerController --ZMQ--> GR00T deploy (whole-body controller, owns rt/lowcmd)

* The robot moves only while the operator has armed it AND holds the HOLD input (the runtime's `ros` mode). A disarm, STOP
  or trip cancels the running task; re-arming never resumes it.
* GR00T takes the arms too: the planner message carries `upper_body_position` (17 joints) and `left/right_hand_joints`
  (7 each). The adapter clamps them with S1 and rate-limits them (`GrootPlannerController.set_upper_body`).
* Perception is ground-referenced (body origin on the floor), the arm pelvis-referenced; `pelvis_height_m` converts.

## Try it in simulation (no GPU, no Isaac)

    python3 scripts/demo_cpusim.py --out demo.mp4 --seed 0       # video: map, down camera, live status
    python3 -m pytest tests/cpusim -q                            # 17 end-to-end tests (~20 s)

The CPU simulator (`humanoid_arc_bme/cpusim`) ray-casts a down camera, runs the real stack and the real GR00T controller
(wire format, S1 clamp, rate limit) against a kinematic robot. It checks the logic and the wiring. It does **not** check
grasp physics: use the Isaac Lab pick-place task (`scripts/pick_place/`) for that.

Measured in it (seeded, with 5 mm localizer noise, random table height 0.68-0.76 m, cup positions and colours): 30 of 30
pick-and-place episodes succeeded. A single hand-tuned scene had hidden failures that randomisation found; keep
randomising when you change anything.

## Try it in Isaac Sim (the simulated G1, its sensors, the robot's own interfaces)

`humanoid_arc_bme/isaac/` puts the same stack on Unitree's G1 asset in Isaac Sim 5.0 (CPU PhysX, 200 Hz). Nothing in `onboard/` is
subclassed or changed for it: the simulator supplies the five objects the robot supplies (a `RobotIO`, a camera source, a lidar
source, a localizer, a hand feed) and a transport for the planner messages, and `isaac/session.py` wires `OperatorRuntime`,
`AutonomyLoop`, `SensorHub` and `GrootPlannerController` exactly as `runtime/cli.py` does.

    planner messages (ZMQ wire format) -> deploy STAND-IN (isaac/standin.py) -> rt/lowcmd -> motor model -> PhysX
    rt/lowstate + Dex3 state <- simulated robot;  D435 (down) + head RGB-D + Mid-360-like lidar + IMU <- isaac/sensors.py

| piece | file | state |
|---|---|---|
| motor order, motor model | `onboard/g1_layout.py`, `isaac/g1_model.py`, `isaac/robot.py` | motor order from Unitree's SDK example and simulator (OPENED); every joint name checked against the asset in the simulator |
| rt/lowcmd, rt/lowstate, Dex3 on DDS | `isaac/dds.py`, `isaac/lowlevel.py` | loopback only, own discovery tag, free domain; a controller in another process drives the robot through it (`scripts/isaac/g1_dds_gate.py`) |
| the deploy | `isaac/standin.py`, `isaac/standin_main.py` | **STAND-IN**: the SONIC binary and ONNX files are not on the development machine. Same topics and documented behaviour (message used once, 1000 ms planner timeout, stop-control exits); the policy inside is NVIDIA AGILE |
| sensors | `isaac/sensors.py`, `config/sensor_mounts.yaml`, `isaac/lidar_profiles/` | mounts equal the asset's frames (checked); lidar is a 40-line approximation of the Mid-360 rosette; depth noise is the stack's `DepthNoise` with guessed values |
| scenes | `isaac/scene.py` | any folder with a `scene.json` manifest and its USD file; this package does not generate scenes |
| scenarios | `isaac/scenarios.py`, `scripts/isaac/run_scenarios.py` | S1-S7 with reset by pose restore, the FastAPI gateway through an in-process HTTP client, the Jev fake |

    ISAAC=<isaac-sim 5.0>/python.sh                       # scripts are run by path; Kit must start before numpy is imported
    $ISAAC scripts/isaac/g1_loco_gate.py --out loco.json            # stand, walk, turn, stop, planner timeout, controller exit
    $ISAAC scripts/isaac/g1_sensor_gate.py --out sensors.json       # each sensor against known geometry, mounts against the asset
    python3 scripts/isaac/dds_loopback_check.py                     # DDS stays on loopback (no Kit)
    $ISAAC scripts/isaac/g1_dds_gate.py --out dds.json --python <python with cyclonedds, unitree_sdk2py, torch, zmq>
    $ISAAC scripts/isaac/run_scenarios.py --scene <scene dir> --out report.json

Extra packages (not installed with the repository): `cyclonedds`, `unitree_sdk2py`, `pyzmq`, `httpx`; folders outside
site-packages go in `$HUMANOID_ARC_DDS_PATHS`. Kit logs go to `$HUMANOID_ARC_KIT_LOGS`.

What the simulator found that the CPU simulator could not (2026-11-14):
* **The down camera sees the robot.** Feet, shins and hands fill about 14 % of the D435 image; the guard took them for obstacles
  0.1-0.3 m ahead and the robot never moved. `RunnerConfig.camera_self_radius_m` (default 0 = off, as before) masks depth points
  inside a body cylinder, like the lidar's mask; the simulated robot uses 0.40 m for both. UNMEASURED on the robot, and an arm that
  reaches forward for a pick is outside any such cylinder.
* **The guard cannot pass a door.** Corridor half width 0.35 m plus costmap inflation 0.30 m stop the robot for anything within
  0.65 m of its path: it stops beside walls and in front of every 0.9 m door. With numbers sized to the body (0.27 m + 0.10 m,
  a guess) it walks across rooms but still stops in front of doors. Door passage needs its own behaviour, not a parameter.
* **The head lidar is blind near the floor.** Mounted 2.3 degrees down at 1.2 m, its nearest floor return is about 5 m away;
  `near_observed` stays false, as the coverage study predicted. Objects taller than the head are seen.
* **Objects on furniture are out of the down camera's view beyond about 0.6 m.** "Pick up the cup" from a metre away ends in "I do
  not see any cup". The head camera sees them; the stack does not use it yet.
* **Arm rest pose matters to perception.** At the joints' zero the forearms point forward into the camera's view. How SONIC carries
  the arms decides what the D435 sees on the robot.
* **A planner with a speed floor.** The stand-in policy ignores commands under about 0.15 m/s and 0.9 rad/s (learned). The stand-in
  hides that with closed-loop bursts; SONIC's documented slow-walk range also starts at 0.2 m/s, the top of S1's limit. The adapter's
  `min_walk_speed` would RAISE a capped speed to the floor: do not use it as it is.
* A second stack tick with the same time stamp latches `bad_stamp` (by design). Anything that steps the stack outside the loop
  must not do so at the loop's stamp.

## Running on the robot: in stages, never skipping one

1. `python3 -m pytest tests -q` on the Jetson. `python3 scripts/export/check_onnx.py <pack>` for the detector.
2. Observe only: `python3 -m humanoid_arc_bme.onboard.runtime observe --io unitree --iface <if>`. (RUNSHEET D4-D5.)
3. Record real data with the robot supported (RGB-D, Livox, DLIO): measure the mounts, tune the detector, replay DLIO.
4. `serve --io unitree --iface <if> --groot --autonomy config/autonomy.yaml` with the robot on the gantry, rope taut, the
   vendor remote's stop in hand. The runtime starts DISARMED. First: velocity commands only (no arm tasks).
5. Arm tasks last, after the GR00T upper-body field order has been confirmed (below).

## Calibration: what is a placeholder (`config/autonomy.yaml`, `config/g1_limits.yaml`)

| item | where | why it matters |
|---|---|---|
| camera mount (position, pitch) | sensor_mounts.yaml (asset values) / autonomy.yaml | sets every distance the guard and the grasp use |
| body self-view radius (camera and lidar) | sensor_mounts.yaml, autonomy.yaml `runner` | without it the guard stops on the robot's own feet and hands |
| `pelvis_height_m` | autonomy.yaml | ground-referenced perception vs pelvis-referenced arm |
| Livox mount and tilt | autonomy.yaml | a level lidar sees nothing nearer than ~1 m |
| Dex3 palm-to-pinch offset, open/closed presets, tool offset | kinematics.py, dex3.py | where the fingers actually are |
| Dex3 joint order and the 17-DoF upper-body order | g1_layout.py | **unverified**: confirm in the deploy's MuJoCo sim before any arm motion |
| depth noise, hand dimensions | sensing.py, grasp_analytic.py | grasp precision |
| S1 limits | g1_limits.yaml | `validated_on_robot: false` until signed off |

## Limitations found (so far)

* **No rear sensor:** the robot never reverses. Next to a table it cannot back away, and turning in place beside the table
  is stopped by the guard (correctly: the torso would sweep within ~0.13 m of the edge). Reposition with the vendor remote.
* **Reach:** a standing pick needs the object near pelvis height and ~0.25 m ahead on the arm's side (IK sweep of the URDF).
  A crouch (GR00T height command) would extend it; not implemented.
* **Close range:** at table distance the whole floor strip is inside the camera's blind zone. The guard has a narrow
  close-range mode (raw 0.12 m clearance, 0.06 m/s, remembered floor plane, camera over memory). Its clearance must be
  confirmed against the real body before use.
* **YOLO:** COCO-class detection of your real objects is untested here (no weights, no real images). Fine-tune on your cups.
  Ultralytics code and weights are AGPL-3.0.
* **Grasp physics, contact, friction:** only in Isaac Lab, not in the CPU simulator.

## Overriding a part (the advanced repository)

| replace | by implementing | where |
|---|---|---|
| detector / semantics (e.g. DAVE's model) | `SemanticsModel.detect(rgb, depth) -> [Detection]` | onboard/pipeline/interfaces.py |
| grasp proposer | `GraspProposer.propose(points, obstacles) -> [GraspCandidate]` | same |
| localizer | `Localizer.estimate(stamp)` | same |
| command parsing | `IntentClassifier.classify(...)` plugged into `CompositeParser` | onboard/command/composite.py |
| sensors | any object with `latest()` / `estimate()` given to `SensorHub` | onboard/sensors/hub.py |

## Tests

`python3 -m pytest tests -q` (needs numpy, pyyaml, pytest; onnxruntime, fastapi, httpx, torch, onnx enable the export,
gateway and detector tests; the rest skip). About 1,900 tests, ~2 minutes.
