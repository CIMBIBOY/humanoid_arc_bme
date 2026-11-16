# Live simulation view

One web page on the stack's own interfaces: the simulated G1 with the full onboard stack in a generated scene, in real time.

    ISAAC=<isaac-sim 5.0>/python.sh
    $ISAAC scripts/isaac/sim_ui.py --scene <scene dir> --host <127.0.0.1 | tailscale address> --yolo <yolo11n-seg.onnx>
    # prints  SIM_UI_READY http://<host>:<port>/#token=<token>   (port: a free one chosen by the kernel; add &view=sem or &view=lang to open a tab)

`--host` must be loopback or a Tailscale address (`command_gateway.check_bind_host`); every call except the page and STOP needs the
token. Without `--yolo` the simulator's segmentation stands in for the detector (the page says so). Run it through
`tools/guard/guarded_run.py` on a shared machine: the guard pauses the simulation when the machine is overloaded, and the page then
freezes and shows "connection lost" until it resumes.

| view | content |
|---|---|
| Map & cameras | lidar map built with the ESTIMATED pose (4 colours by height: floor, low, body, overhead; ceiling returns are not drawn), the current scan on top, the pose trail, estimated and true pose, the goal; click the map to send a goal. Side column: D435 colour and depth, head camera colour and depth (depth colour scale shown) |
| Semantics | the detector's masks and labels on the D435 (the camera the robot acts on) and on the head camera (display only), the detection list, and the stack's scene memory (label, colour, range, body-frame position) |
| Language | the stack's command gateway: sentence -> understood command -> Confirm -> executes. Understanding: grammar only, grammar + Jev (fake), grammar + Jev outage (the fake times out: the robot must keep working) |

Operating: Arm (the two-step arm with the heartbeat check is done for you), then HOLD TO RUN (button, or W/A/S/D/Q/E: velocity teleop). The
runtime's deadman stays in force: no heartbeat or no HOLD means no motion; STOP (button, Esc, Space) needs no token; Disarm and Reset
robot are separate. "Fault tests" switch a sensor or link failure on while the robot walks.

What is simulated and what is real is listed on the page ("What is real here"): the locomotion policy is a stand-in behind the SONIC
interface, the pose is simulator truth plus noise (DLIO is not run), Jev is a fake with a placeholder schema.

Design notes: the simulator's main thread owns Kit; HTTP threads only read the newest snapshot and queue requests, except the
operator heartbeat and the command gateway, which are the robot's own thread-safe entry points. The heartbeat doubles as the state
feed (10 Hz). Images are JPEGs encoded on demand, one request in flight per image; the lidar map is sent once a second, the scan
at 4 Hz.
