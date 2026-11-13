# G1 run sheet (one page). Full version, reasons and UNKNOWNs: docs/plan/robot-deploy.md

HOLD = powers or moves the robot: only you, robot supported (gantry / rope taut), vendor remote stop in hand.
The runtime starts DISARMED. Nothing moves until ARM + typed code in the UI. STOP = big button / Space / Esc /
`curl -X POST http://<ip>:8080/api/stop` (no token needed). Effective envelope (= first-walk caps, from config): vx -0.1..0.2, vy +-0.1, wz +-0.3.
HOLD-TO-RUN: the robot moves ONLY while the pad / a drive key / the HOLD button is pressed; after ARM release it once ("ARMED: release input" goes away) before moving. Sliders only scale speed. Access = SSH tunnel; `--lan` is refused without `--lan-plain-http-ack` (do not use it).

| # | do | expect | HOLD |
|---|---|---|---|
| D0 | workstation: `python3 -m pytest -q tests/runtime`; `python3 deploy/scripts/make_policy_pack.py outputs/W4/onnx/deployE packs/deployE-e2`; `python3 -m humanoid_arc_bme.onboard.runtime.bundle /tmp/g1_bundle.tar.gz` (or commit + git clone) | pass; pack; bundle | no |
| D1 | scp bundle + pack to the Jetson; `tar xzf` into ~/humanoid_arc; `sha256sum -c` in the pack | OK | no* |
| D2 | `bash deploy/scripts/install.sh` | preflight table, unitree_sdk2py + cyclonedds 0.10.2 OK | no |
| D3 | `bash deploy/scripts/test.sh` | all pass | no |
| D4 | robot ON (supported); `bash deploy/scripts/preflight.sh --iface <if> --listen 3` | 192.168.123.x, lowstate msg/s | **HOLD** |
| D5 | `python3 -m humanoid_arc_bme.onboard.runtime observe --io unitree --iface <if> --seconds 10` | no_state 0, stale 0 | **HOLD** |
| D6 | `bash deploy/scripts/run.sh --io unitree --iface <if> --shadow ~/g1_models/deployE-e2`; laptop `ssh -L 8080:127.0.0.1:8080 <user>@<jetson>`, open printed URL; DO NOT ARM | preflight GREEN, 50 Hz, state age small | **HOLD** |
| D7 | vendor remote: stand / walk mode. UI: controller vendor, ARM + code, pad untouched 10 s, Disarm | sent ~10/s, last_code 0, robot stands | **HOLD** |
| D8 | while armed at zero: STOP button (must say CONFIRMED); Space; curl stop; pad held then tab kill / lid close; unplug laptop LAN | STOP LATCHED; deadman_timeout <= 0.3 s | **HOLD** |
| D9 | rope slack, 5 m clear: arm, release pad once, hold pad 0.1 m/s 3 s, release. Then 0.1 m/s + `kill -9 <pid>`: time to stop | walks, stops; stops by itself after kill (measure) | **HOLD** |
| D10 | wz 0.2 rad/s 3 s; vy 0.05 3 s; curvature v 0.1 kappa 1.0 | as commanded | **HOLD** |
| D11 | UI Logs: download events/cmd/state/telemetry.mcap or scp ~/g1_runs/run_* | files open | no |

*D1: no, unless switching the dev computer on means powering the robot.
Not day one: policy controller on the robot (needs B3.3 validated limits, gantry, --allow-lowcmd, RELEASE VENDOR MODE).
