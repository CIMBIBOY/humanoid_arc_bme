# G1 EDU (Dex3) inventory: what the vendor sources say, what is still UNKNOWN, and the read-only run sheet

Written 2026-10-30, for the first inspection of the robot (B0.1-B0.7, B3.1-B3.2 of B-robot-g1 (private plan)).
What we already know about the robot: G1 EDU with Dex3 hands, a Jetson Orin NX dev computer, JetPack 6.2 and
ROS 2 Humble installed (from you, not checked). Everything below about the vendor stack was read from Unitree or
third-party pages through `Aether/tools/web_scraper` on 2026-10-30.

Tags: **OPENED** = the page or file text was read (URL given). **ABSTRACT** = only a summary or listing was seen.
**UNKNOWN** = no source read gives it; the scripts or the manual have to give it. The scripts' results are
**MEASURED**. Unitree's own developer site (support.unitree.com) is rendered by JavaScript and returned no text to
the scraper, so nothing here comes from the official G1 SDK Development Guide. Treat every OPENED row as "a
Unitree repo says so", not "this robot does so".

Scripts: `deploy/scripts/inventory/` (`inventory.sh`, `sdk_probe.py`, `ros_topic_probe.py`, `camera_probe.py`,
`report_to_facts.py`, shared `probe_common.py`). Tests: `tests/runtime/test_inventory_scripts.py`.

---

## 1. Hardware (product page)

Source: https://www.unitree.com/g1 (OPENED; the parameter table, G1 vs G1 EDU columns).

| fact (G1 EDU column) | tag |
|---|---|
| 23-43 DoF total; leg 6 per side; waist 1 (+2 optional); arm 5 (+2 optional wrist) | OPENED |
| Hand: 7 DoF, "optional force control of three-fingered hand" Dex3-1: thumb 3 active DoF, index 2, middle 2; Dex3-1 "can optionally be installed with tactile sensor arrays" | OPENED |
| "Sensing sensor: Depth Camera + 3D LiDAR"; 4-microphone array; 5 W speaker; WiFi 6, Bluetooth 5.2 | OPENED |
| Camera and LiDAR **models are not named** on the page (D435i / MID360 are what people report, not confirmed by any source read) | UNKNOWN |
| "Basic computing power: 8-core high-performance CPU"; EDU "High computing power module: multiple brands and models available, such as Orin" | OPENED |
| 13-string lithium battery, 9000 mAh quick-release smart battery, charger 54 V 5 A, about 2 h; about 35 kg+; knee max torque 120 N·m; arm max load about 3 kg | OPENED |
| Joint ranges: waist Z ±155°, X ±30°, Y ±30°; knee 0-165°; hip P ±154°, R -30..+170°, Y ±158°; wrist P ±92.5°, Y ±92.5°; dual encoders; local air cooling | OPENED |
| EDU: secondary development YES; warranty 18 months ("read the product warranty brochure") | OPENED |

## 2. Computers and network

| fact | tag | source |
|---|---|---|
| Two onboard computers on 192.168.123.0/24: motion-control **PC1 = 192.168.123.161**, development **PC2 = 192.168.123.164**; they sit behind an internal switch connected to the robot's RJ45 port | OPENED | xr_teleoperate wiki "CycloneDDS and UnitreeSDK (en)" and "Network": https://github.com/unitreerobotics/xr_teleoperate/wiki |
| Host on the robot network: 192.168.123.99 recommended (xr wiki), 192.168.123.222 per "the G1 SDK Development Guide" (xr wiki and GEAR WBC page) | OPENED | same; [vendor/gr00t-wbc-g1-jetpack6.md](vendor/gr00t-wbc-g1-jetpack6.md) |
| PC2 login `unitree@192.168.123.164` (the factory password is on the LeRobot page; change it); the robot-side interface is named `eth0`; WiFi (`wlan0`) is blocked by default (`rfkill`) | OPENED | LeRobot G1 page: https://huggingface.co/docs/lerobot/unitree_g1 |
| PC2 is a Jetson Orin NX (teleimager's install steps say "for Jetson Orin NX (ARM architecture)", run as `unitree@ubuntu`) | OPENED | https://github.com/silencht/teleimager README |
| PC2 has a wireless card, off by default; bridging it into 192.168.123.x failed for the xr author "due to CycloneDDS" | OPENED | xr wiki "Network" |
| USB-C power on the dev ports No.6/7/8: 5 V 1.5 A each; No.9: PD up to 20 V 2 A (40 W); 6 A total shared; use a PD hub on No.9 for several RealSense cameras (the page does not say which robot model) | OPENED | xr wiki "Camera_and_Image" |
| Which module (NX 8 GB / 16 GB), L4T release, disk, power mode on **this** robot | UNKNOWN | `inventory.sh` (U01, U02, U09, U10) |

## 3. SDK and DDS

| fact | tag | source |
|---|---|---|
| `unitree_sdk2_python`: Python ≥ 3.8, `cyclonedds == 0.10.2`, numpy, opencv-python; install `pip3 install -e .` from the clone; if cyclonedds is not found build it (`releases/0.10.x`) and set `CYCLONEDDS_HOME` | OPENED | https://github.com/unitreerobotics/unitree_sdk2_python README |
| `unitree_sdk2` (C++): Ubuntu 20.04, aarch64 and x86_64, gcc 9.4; bundles CycloneDDS C/C++ 0.10.2 in `thirdparty` | OPENED | unitree_sdk2 README; xr wiki DDS page |
| Default DDS domain 0: `ChannelFactoryInitialize(0, "<nic>")`; without a NIC name the SDK auto-selects one | OPENED | xr wiki DDS page; `unitree_sdk2py/core/channel.py` |
| `ChannelSubscriber` creates only a DataReader; `ChannelPublisher` creates the DataWriter (Init) | OPENED | `unitree_sdk2py/core/channel.py` (read in full) |
| `idl/unitree_hg` is the G1/H1-2 message set, `idl/unitree_go` the Go2/B2/H1 set | OPENED | `example/g1/readme.md` |
| hg IDL types: BmsCmd, BmsState, HandCmd, HandState, IMUState, LowCmd, LowState, MainBoardState, MotorCmd, MotorState, PressSensorState | OPENED (listing) | `unitree_sdk2py/idl/unitree_hg/msg/dds_/` |
| `LowState`: version[2], mode_pr, mode_machine, tick, imu_state, **motor_state[35]**, wireless_remote[40], reserve[4], crc. No battery field (unlike the Go2 LowState) | OPENED | unitree_ros2 `unitree_hg/msg/LowState.msg` |
| `MotorState`: mode, q, dq, ddq, tau_est, temperature int16[2], vol, sensor[2], motorstate, reserve[4]. `IMUState`: quaternion[4], gyroscope[3], accelerometer[3], rpy[3], temperature | OPENED | `MotorState.msg`, `IMUState.msg` |
| `BmsState`: version_high/low, fn, cell_vol[40], bmsvoltage[3], current, soc, soh, temperature[12], cycle, manufacturer_date, bmsstate[5]. `MainBoardState`: fan_state[6], temperature[6], value[6], state[6] | OPENED | `BmsState.msg`, `MainBoardState.msg` |
| Which **topic** carries BmsState / MainBoardState on the G1 | UNKNOWN | `sdk_probe.py` tries `rt/lf/bmsstate`, `rt/bmsstate`, `rt/lf/mainboardstate` (U19) |
| G1 body joint order 0-28: L hip pitch/roll/yaw, knee, ankle pitch/roll (0-5), R the same (6-11), waist yaw/roll/pitch (12-14; roll/pitch invalid on waist-locked 23/29-DoF), L shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw (15-21), R the same (22-28). Ankles and waist have PR / AB (parallel) modes | OPENED | `example/g1/low_level/g1_low_level_example.py` |
| Low-level example: 2 ms control period, CRC on LowCmd, copies `mode_machine` from the first LowState, and **loops `MotionSwitcherClient.CheckMode()` / `ReleaseMode()` until no mode is active before publishing `rt/lowcmd`** | OPENED | same |
| `rt/lowstate` rate on the real robot | UNKNOWN | `sdk_probe.py` (U14) |
| torso IMU on `rt/secondary_imu` (G1 only) | OPENED (simulator README) | https://github.com/unitreerobotics/unitree_mujoco README (U18 checks the robot) |
| "SportModeState is not readable after the built-in motion control service is turned off" (real hardware) | OPENED | unitree_mujoco README |

## 4. Vendor services (RPC): motion switcher, loco client, arm and hand SDK topics

All of these are request/response clients: **calling any of them, including the "get" calls, publishes a request.**
The inventory probes therefore call none of them; the current mode and FSM id stay UNKNOWN until a Hold step.

| fact | tag | source |
|---|---|---|
| MotionSwitcher: service `motion_switcher`, API 1.0.0.1, ids 1001 CheckMode, 1002 SelectMode(name), 1003 ReleaseMode, 1004 SetSilent, 1005 GetSilent | OPENED | `unitree_sdk2py/comm/motion_switcher/motion_switcher_{client,api}.py` |
| G1 LocoClient: service **`sport`**, API 1.0.0.0; get ids 7001-7006 (FSM id, FSM mode, balance mode, swing height, stand height, phase-deprecated), set ids 7101 FSM id, 7102 balance mode, 7103 swing height, 7104 stand height, 7105 velocity, 7106 arm task, 7107 speed mode, 7110 SwitchToUserCtrl, 7111 SwitchToInternalCtrl(LAST/PASSIVE/WALKRUN) | OPENED | `unitree_sdk2py/g1/loco/g1_loco_{client,api}.py` |
| FSM ids used by the helpers: ZeroTorque 0, Damp 1, Sit 3, Start 500, Lie2StandUp 702, Squat2StandUp / StandUp2Squat 706; `Move(vx, vy, vyaw)` = SetVelocity with duration 1 s (or 864000 s continuous); High/LowStand via SetStandHeight(UINT32 max / 0); WaveHand / ShakeHand via arm task ids 0-3 | OPENED | same + `example/g1/high_level/g1_loco_client_example.py` |
| Arm control while the vendor walks: publish LowCmd on **`rt/arm_sdk`**; `motor_cmd[kNotUsedJoint0].q` is a weight w, executed arm command = motion·(1-w) + user·w; ramp w slowly | OPENED | xr wiki "Motion"; `example/g1/high_level/g1_arm7_sdk_dds_example.py` |
| Other command topics in the C++ examples: `rt/user_lowcmd` (g1_userctrl_dds_example), `rt/hand_sdk` (g1_hand_sdk_example) | OPENED (topic strings) | https://github.com/unitreerobotics/unitree_sdk2/tree/main/example/g1/high_level |
| Remote: current docs say L2+B damping, L2+UP locked standing, R1+X regular mode (older firmware: L1+A, L1+UP); "debug mode" is the low-level mode (vendor controller off) | OPENED (second-hand, xr wiki quoting Unitree docs) | xr wiki "Motion" |
| Example programs default to `rt/lowcmd` in debug mode and `rt/arm_sdk` in motion mode (`kTopicLowCommand_Debug`, `kTopicLowCommand_Motion`); `G1_29_Num_Motors = 35` | OPENED | xr wiki "Motion" (robot_arm.py excerpt) |

## 5. Dex3 hands

| fact | tag | source |
|---|---|---|
| Command `rt/dex3/{left,right}/cmd` (hg `HandCmd_`), state `rt/dex3/{left,right}/state` (hg `HandState_`), 7 motors per hand | OPENED | xr_teleoperate `teleop/robot_control/robot_hand_unitree.py` |
| The C++ example subscribes to **`rt/lf/dex3/{left,right}/state`** instead (low-frequency variant?) and sizes 7 motors, 9 pressure sensors | OPENED | unitree_sdk2 `example/g1/dex3/g1_dex3_example.cpp` |
| `HandState`: motor_state[], press_sensor_state[] (`PressSensorState`: pressure[12], temperature[12], lost), imu_state, power_v, power_a, system_v, device_v, error[2] | OPENED | unitree_ros2 `HandState.msg`, `PressSensorState.msg` |
| Dex3 joint limits in that example equal `config/g1_limits.yaml` (URDF) for the left hand; for the right hand the example writes thumb_1 max 0.742 where our URDF has 0.72431 (likely a typo in the example; check at B3.3) | OPENED + compared | example vs `config/g1_limits.yaml` |
| Which state topic this robot publishes, rate, whether tactile arrays are fitted (pressure values all zero?) | UNKNOWN | `sdk_probe.py` (U20) |

## 6. Cameras

| fact | tag | source |
|---|---|---|
| Unitree's teleimager image server (ZMQ PUB-SUB and WebRTC, port 60001 in examples) handles UVC, OpenCV and RealSense cameras; for a physical robot it must be **installed and started manually on PC2**; its sample discovery output lists "USB HDR Camera" (640x480 MJPG) and "Cherry Dual Camera" (stereo 1280x480 MJPG) | OPENED | teleimager README; xr_teleoperate README §3.1 |
| xr_teleoperate's documentation describes the head camera on PC2 (`cam_config_server.yaml → head_camera`) | OPENED | xr_teleoperate README |
| LeRobot's G1 server streams a `head_camera` 640x480 @ 30 over ZMQ port 5555 (`run_g1_server.py --camera`) | OPENED | LeRobot G1 page |
| `unitree_sdk2_python` has `example/front_camera/camera_opencv.py` (a video client) — that README section is generic, not G1-specific | OPENED | sdk2py README |
| Head camera model on this robot (RealSense D435i or a UVC stereo head), USB port, modes, intrinsics, whether a vendor service already holds it | UNKNOWN | `camera_probe.py`, `inventory.sh` lsusb / v4l2 / rs-enumerate (U22, U23) |

## 7. LiDAR

| fact | tag | source |
|---|---|---|
| The product page says "3D LiDAR", model not named | OPENED / UNKNOWN model | unitree.com/g1 |
| Livox ROS Driver 2 supports ROS 2 Humble and has MID360 launch files; its sample config uses host 192.168.1.5 and LiDAR 192.168.1.1xx, so a LiDAR on 192.168.123.x needs its own config | OPENED | https://github.com/Livox-SDK/livox_ros_driver2 README |
| unitree_ros2's RViz example uses `utlidar/cloud`: that is the Go2's built-in LiDAR, **not evidence for the G1** | OPENED | unitree_ros2 README |
| LiDAR model, IP, whether a driver runs on PC2, topic | UNKNOWN | `inventory.sh` `ip neigh` (other hosts on 192.168.123.x), ROS topic list (U24); the label (M02) |

## 8. ROS 2 bridge

| fact | tag | source |
|---|---|---|
| `unitree_ros2`: ROS 2 talks to the robot directly over DDS, no wrapper; Humble on 22.04 recommended; `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`; `CYCLONEDDS_URI` names the robot NIC; with Humble the cyclonedds 0.10.x build step "can be skipped" | OPENED | https://github.com/unitreerobotics/unitree_ros2 README |
| ROS topic `lowstate` / `lf/lowstate` (DDS `rt/lowstate`, `rt/lf/lowstate`); hg reader example `read_low_state_hg`; G1 examples under `example/src/src/g1/{audio_client,dex3,high_level,lowlevel}`; message packages `unitree_hg`, `unitree_go`, `unitree_api` in `cyclonedds_ws` | OPENED | unitree_ros2 README and tree |
| Without the `unitree_ros2` workspace sourced, `ros2 topic list` still shows the topics, but `ros2 topic echo` / our ROS probe cannot decode them | from the ROS 2 type system; not specific to Unitree | — |
| Whether ROS 2 on PC2 is already configured for the robot (RMW, URI, workspace), which topics appear | UNKNOWN | `inventory.sh` section 7, `ros_topic_probe.py` (U05, U13, U30) |

---

## 9. UNKNOWN register

`report_to_facts.py` marks each id **resolved / partial / open** from the scripts' output (the ids and the wording
below are checked against the script by a test).

| id | question | answered by |
|---|---|---|
| U01 | Jetson module variant and RAM | inventory.sh 01_model, 03_memory |
| U02 | L4T / JetPack release on the dev computer | 01_nv_tegra_release, 01_dpkg_nvidia |
| U03 | Ubuntu release and kernel | 01_os_release, 01_uname |
| U04 | CUDA / TensorRT / cuDNN versions installed | 01_cuda_trt, 01_dpkg_nvidia |
| U05 | ROS 2 distro installed and the RMW configured | 07_ros_env, 07_env |
| U06 | unitree_sdk2py and cyclonedds (python) installed, versions | 08_python_* |
| U07 | onnxruntime / TensorRT python bindings and their GPU providers | 08_python_* |
| U08 | pyrealsense2 / OpenCV in python | 08_python_* |
| U09 | power mode (nvpmodel) and clocks | 02_nvpmodel, 02_jetson_clocks_show (--sudo) |
| U10 | disk: NVMe model, size, free space | 03_df, 03_nvme |
| U11 | robot-network interface name and the dev computer's IP | 04_ip_addr |
| U12 | motion computer 192.168.123.161 reachable from the dev computer | 04_ping_192.168.123.161 |
| U13 | DDS domain id and CycloneDDS config used on the dev computer | 07_env, 07_cyclonedds_uri |
| U14 | rt/lowstate rate and jitter seen on the dev computer | sdk_probe |
| U15 | LowState motor slots and live motors (29-DoF body?) | sdk_probe |
| U16 | mode_machine / mode_pr values reported by LowState | sdk_probe |
| U17 | IMU fields: quaternion order/norm, accel magnitude, rpy | sdk_probe |
| U18 | torso secondary IMU topic (rt/secondary_imu) present | sdk_probe |
| U19 | battery state topic and fields | sdk_probe (guessed names) |
| U20 | Dex3 state topic names, motor and pressure-sensor counts | sdk_probe |
| U21 | SportModeState / odometry topic on the G1 | sdk_probe |
| U22 | head camera: model, interface, resolution, fps | camera_probe |
| U23 | RealSense depth camera present, intrinsics, IMU | camera_probe |
| U24 | LiDAR model, address and topic | 04_ip_neigh, 07_ros2_topics (model needs the label: at most partial) |
| U25 | clock sync on the dev computer (NTP / chrony) | 05_timedatectl, 05_time_services |
| U26 | vendor services running on the dev computer | 06_services_running |
| U27 | who publishes rt/lowcmd / rt/arm_sdk while the vendor controller runs | sdk_probe --watch-command-topics |
| U28 | microphone array capture device | 09_audio |
| U29 | real-time kernel and CPU governor | 01_uname, 01_cpu_governor |
| U30 | ROS 2 topics the robot exposes on the dev computer | 07_ros2_topics, ros_topic_probe |

Not answerable by a read-only script (manual, label, or a Hold step):

| id | question | how |
|---|---|---|
| M01 | serial number, variant label, hand variant (tactile arrays fitted?) | B0.1 photos |
| M02 | LiDAR model, mounting pose | label / manual (B0.6) |
| M03 | head camera model and mounting pose relative to the pelvis | label / URDF / manual (B0.6) |
| M04 | motion computer (PC1) firmware / OTA version | manual or app; PC1 is not logged into |
| M05 | active motion-switcher mode and loco FSM id | RPC call = a request publish: Hold step (B4.0) |
| M06 | the official G1 SDK Development Guide version and its topic list | B0.7 / B2.1 (support.unitree.com did not render for the scraper) |
| M07 | warranty position on reflashing PC2 | B2.1 |

---

## 10. Run sheet (you at the robot; nothing here moves it)

**Robot state for all steps:** powered on per the manual, in the vendor's default state after boot (do **not**
press any remote combination; no stand-up, no damping change), on its stand or seated, nobody else connected to
it. If it is hanging in the gantry that is fine too.

**What "read-only" means here.** No script creates a DDS writer, calls an RPC client, publishes a ROS message,
calls a service or changes a setting. Unavoidable side effects: each DDS participant (`sdk_probe`, `ros2 ... list`,
the ROS probe) sends discovery announcements on the network; `ping` sends 3 packets to .161 and .164; the camera
probe streams the camera for a few seconds (if the vendor already streams it, the probe reports "busy" and moves
on). Each script refuses to start without `--i-understand-read-only`.

### Step 0. Laptop on the robot network (on the laptop)

```
ip -br a                                   # find the USB-Ethernet NIC, e.g. enx...
sudo ip addr add 192.168.123.222/24 dev <nic>     # laptop only, never on the robot
ping -c 3 192.168.123.164; ping -c 3 192.168.123.161
```
Expected: both answer. If .161 does not answer, note it and continue (U12).

### Step 1. Copy the scripts to the dev computer (on the workstation or laptop)

```
cd ~/isaacsim/humanoid_arc
scp -r deploy/scripts/inventory unitree@192.168.123.164:~/inventory_scripts
ssh unitree@192.168.123.164
cd ~/inventory_scripts
```
(The LeRobot page gives `unitree` / `123` as the default login; if it differs, note it, do not reset it.)

### Step 2. System inventory (on PC2), about 1-2 minutes

```
bash inventory.sh --i-understand-read-only --out ~/inventory_reports
# optional, for the clock and NVMe queries that need root (prompts once, then runs only `--show` / list queries):
sudo -v && bash inventory.sh --i-understand-read-only --out ~/inventory_reports --sudo --sections 2,3
```
Expected: `[inventory] done: ~70 files, N commands returned non-zero` and the path of
`inv_<host>_<utc>.tar.gz`. Non-zero codes for missing tools (127) are normal. A timeout (124) on the ros2 lines
means discovery is slow: rerun with `--sections 7 --timeout 40`.

### Step 3. DDS state probe through the SDK (on PC2), 10 s + 10 s

Needs `unitree_sdk2py` importable by the python you use (step 2's `08_python_*.txt` tells which one has it; if none
has it, skip to step 4 and send step 2 back first: installing the SDK is a separate, reviewed step).

```
python3 sdk_probe.py --i-understand-read-only --iface eth0 --seconds 10 --out ~/inventory_reports
python3 sdk_probe.py --i-understand-read-only --iface eth0 --seconds 10 --out ~/inventory_reports --watch-command-topics
```
Use the interface that step 2 shows with a 192.168.123.x address (`--iface auto` finds it). Expected:
`rt/lowstate ... ~500 Hz` (the number is what we measure; the example's 2 ms loop suggests 500 Hz but no source
states the publish rate), a list of silent topics (normal: several names are guesses) and
`wrote .../sdk_probe_<utc>.json`. If **nothing** is received: rerun once with `--iface none`, then with
`--domain 1`, and note which worked.

### Step 4. ROS 2 topic probe (on PC2), 15 s

```
source /opt/ros/humble/setup.bash
[ -f ~/unitree_ros2/setup.sh ] && source ~/unitree_ros2/setup.sh     # only if it exists (step 2 shows it)
python3 ros_topic_probe.py --i-understand-read-only --seconds 15 --out ~/inventory_reports
```
Expected: `N topics discovered, M subscribed, K type-missing` and per-topic rates. Big topics (images, point clouds)
are received serialized, so the CPU cost is small. If ROS 2 shows no robot topics while step 3 did, the RMW or
`CYCLONEDDS_URI` is not set for the robot NIC (that is a finding, U05/U13).

### Step 5. Cameras (on PC2), under 1 minute

```
python3 camera_probe.py --i-understand-read-only --out ~/inventory_reports
```
Expected: the RealSense device list (or `none`), measured fps, and `camera_probe_<utc>/` with
`realsense_<serial>_color.jpg`, `_depth.npy`, `_depth16.png`, and a JPEG per other V4L2 camera that opened.
"could not open (busy)" means another process owns the camera: note which process (`06_ps.txt` has the list); do
not stop it.

### Step 6. Facts and package (on PC2 or back on the workstation)

```
cd ~/inventory_reports
python3 ~/inventory_scripts/report_to_facts.py --i-understand-read-only inv_*Z     # newest folder; probes are found next to it
tar czf inventory_bundle_$(date -u +%Y%m%dT%H%M%SZ).tar.gz inv_*Z sdk_probe_*.json ros_topic_probe_*.json camera_probe_*
exit
scp unitree@192.168.123.164:~/inventory_reports/inventory_bundle_*.tar.gz ~/isaacsim/humanoid_arc/outputs/B0/
```
Expected: `UNKNOWNs resolved X, partial Y, open Z` and `robot_facts.candidates.yaml` +
`robot_facts_summary.md` inside the inventory folder. Do not copy the YAML to `config/robot_facts.yaml` yourself
yet; the assistant reviews it first (`--write-config` exists for that later step).

### What to send back to the assistant

1. `outputs/B0/inventory_bundle_<utc>.tar.gz` (or its path on the workstation).
2. The console output of steps 3-6 (copy-paste), especially any ERROR line.
3. Photos for M01-M03 (label, head sensor area, hands, back panel) into `outputs/B0/`.
4. Anything you did differently from this sheet (another interface, domain, python, a skipped step).

### If something goes wrong

- The robot moves, beeps or changes mode during any step: stop the script (Ctrl-C), note the time and step. None
  of these scripts can command it, so report it as a finding.
- A script prints `REFUSING TO RUN`: the flag is missing; nothing was done.
- `sdk_probe.py` prints `unitree_sdk2py not importable`: expected if the SDK is not installed for that python.
