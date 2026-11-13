#!/usr/bin/env bash
# READ-ONLY inventory of the G1 dev computer (Jetson Orin NX). Run it ON the Jetson, as the normal user.
#
# What it does: runs only query commands (versions, lists, status, sysfs reads) and writes their output into one
# timestamped folder plus a .tar.gz of it. It changes no setting, starts/stops no service, sends nothing to the robot.
# What it never does: publish or call anything on DDS/ROS 2, change power mode or clocks, touch network config,
# start or stop services or containers. The only network traffic it creates: optional `ping` to the documented robot
# addresses (skip with --no-ping) and the DDS discovery packets any `ros2 ... list` creates (skip with --no-ros).
#
# Usage:
#   bash inventory.sh --i-understand-read-only [--out DIR] [--no-ros] [--no-ping] [--sudo] [--python PATH]... [--timeout S]
#     --out DIR     parent folder for the report (default: ./inventory_reports)
#     --no-ros      skip every ros2 / cyclonedds CLI call
#     --no-ping     skip ping of 192.168.123.161 / .164
#     --sudo        also run the few root-only QUERIES with `sudo -n` (never prompts; run `sudo -v` first)
#     --python P    extra python interpreter to inspect (repeatable; conda envs under ~/miniconda3 are found automatically)
#     --timeout S   per-command timeout in seconds (default 20)
#     --sections L  only these sections, comma list of 1-9 (default all): 1 system 2 power 3 disk 4 network 5 time
#                   6 services 7 DDS/ROS 8 python 9 devices
#
# Tagging: everything in the report is MEASURED on the machine it ran on. docs/robot-inventory.md has the run sheet.

VERSION="inventory.sh 1.0 (2026-10-30)"

ACK=0; OUTROOT="./inventory_reports"; SECTIONS=all; DO_ROS=1; DO_PING=1; DO_SUDO=0; T=20; EXTRA_PY=()
ARGS=("$@")
while [ $# -gt 0 ]; do
  case "$1" in
    --i-understand-read-only) ACK=1 ;;
    --out) OUTROOT="$2"; shift ;;
    --no-ros) DO_ROS=0 ;;
    --no-ping) DO_PING=0 ;;
    --sudo) DO_SUDO=1 ;;
    --python) EXTRA_PY+=("$2"); shift ;;
    --timeout) T="$2"; shift ;;
    --sections) SECTIONS="$2"; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ "$ACK" != "1" ]; then
  echo "REFUSING TO RUN: pass --i-understand-read-only to confirm you want the read-only inventory." >&2
  echo "This script only reads; see the header of $0 for exactly what it does." >&2
  exit 2
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
HOST="$(hostname 2>/dev/null || echo unknown)"
OUT="$OUTROOT/inv_${HOST}_${TS}"
mkdir -p "$OUT" || { echo "cannot create $OUT" >&2; exit 1; }
: > "$OUT/_exit_codes.txt"

# run NAME 'COMMAND STRING'   -> $OUT/NAME.txt with the command, its stdout+stderr; exit code into _exit_codes.txt
run() {
  local name="$1"; shift
  local f="$OUT/$name.txt"
  { echo "\$ $*"; echo "---"; } > "$f"
  timeout "$T" bash -c "$*" >> "$f" 2>&1 < /dev/null
  local rc=$?
  echo "$name $rc" >> "$OUT/_exit_codes.txt"
  return 0
}
have() { command -v "$1" > /dev/null 2>&1; }
want() { [ "$SECTIONS" = all ] || [[ ",$SECTIONS," == *",$1,"* ]]; }
say() { echo "[inventory] $*"; }

say "writing $OUT"
{
  echo "version: $VERSION"
  echo "utc_start: $TS"
  echo "host: $HOST"
  echo "user: $(id -un 2>/dev/null)"
  echo "args: ${ARGS[*]}"
  echo "script_sha256: $(sha256sum "$0" 2>/dev/null | cut -d' ' -f1)"
  echo "ros: $DO_ROS ping: $DO_PING sudo: $DO_SUDO timeout: $T sections: $SECTIONS"
} > "$OUT/_meta.txt"

# ---------------------------------------------------------------- 1. system, module, JetPack / L4T
if want 1; then
say "system"
run 01_model            'cat /proc/device-tree/model; echo; cat /proc/device-tree/compatible | tr "\0" " "; echo'
run 01_nv_tegra_release 'cat /etc/nv_tegra_release'
run 01_os_release       'cat /etc/os-release; lsb_release -a'
run 01_uname            'uname -a; cat /proc/version; cat /sys/kernel/realtime 2>/dev/null || echo "realtime: no /sys/kernel/realtime"'
run 01_cmdline          'cat /proc/cmdline'
run 01_hostnamectl      'hostnamectl'
run 01_uptime           'uptime; who -b'
run 01_cpu              'nproc; lscpu'
run 01_cpu_governor     'for c in /sys/devices/system/cpu/cpu[0-9]*; do echo "$(basename $c) online=$(cat $c/online 2>/dev/null || echo 1) gov=$(cat $c/cpufreq/scaling_governor 2>/dev/null) cur=$(cat $c/cpufreq/scaling_cur_freq 2>/dev/null) max=$(cat $c/cpufreq/scaling_max_freq 2>/dev/null)"; done'
run 01_limits           'ulimit -a; echo; id; groups'
run 01_dpkg_nvidia      'dpkg-query -W -f="\${Package} \${Version}\n" 2>/dev/null | grep -E "^(nvidia-jetpack|nvidia-l4t-core|nvidia-l4t-kernel|nvidia-l4t-cuda|cuda-toolkit|cuda-runtime|libcudnn|cudnn|tensorrt|libnvinfer|python3-libnvinfer|nvidia-container|docker)" | sort'
run 01_dpkg_ros         'dpkg-query -W -f="\${Package} \${Version}\n" 2>/dev/null | grep -E "^ros-|^python3-rclpy|^python3-colcon|cyclonedds|fastrtps|realsense|librealsense|livox|unitree" | sort'
run 01_apt_sources      'grep -hE "^deb" /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null; ls /etc/apt/sources.list.d/'
run 01_cuda_trt         'ls -d /usr/local/cuda* 2>/dev/null; /usr/local/cuda/bin/nvcc --version 2>/dev/null || nvcc --version; ls /usr/src/tensorrt/bin 2>/dev/null; command -v trtexec; ls -d $HOME/TensorRT* 2>/dev/null; find / -maxdepth 4 -name trtexec -type f 2>/dev/null | head'
run 01_jtop             'command -v jtop; pip3 show jetson-stats 2>/dev/null | head -3'
fi

# ---------------------------------------------------------------- 2. power mode, clocks, temperatures
if want 2; then
say "power and thermal"
run 02_nvpmodel         'nvpmodel -q; echo; nvpmodel -q --verbose 2>/dev/null | tail -40; echo; ls /etc/nvpmodel* 2>/dev/null; grep -E "POWER_MODEL|PM_CONFIG" /etc/nvpmodel.conf 2>/dev/null | head -30'
if [ "$DO_SUDO" = "1" ]; then
  run 02_jetson_clocks_show 'sudo -n jetson_clocks --show'
else
  echo "skipped (needs root; rerun with --sudo after sudo -v)" > "$OUT/02_jetson_clocks_show.txt"
fi
run 02_thermal          'for z in /sys/class/thermal/thermal_zone*; do echo "$(basename $z) $(cat $z/type 2>/dev/null) $(cat $z/temp 2>/dev/null)"; done; for f in /sys/class/hwmon/hwmon*/name; do echo "hwmon $(cat $f)"; done'
run 02_fan              'for f in /sys/devices/platform/*pwm-fan*/hwmon/hwmon*/pwm1 /sys/class/hwmon/hwmon*/rpm; do [ -e "$f" ] && echo "$f $(cat $f)"; done; true'
if have tegrastats; then
  run 02_tegrastats     'timeout 6 tegrastats --interval 1000'
else
  echo "tegrastats not found" > "$OUT/02_tegrastats.txt"
fi
fi

# ---------------------------------------------------------------- 3. memory and disks
if want 3; then
say "memory and disks"
run 03_memory           'free -h; echo; cat /proc/meminfo | head -20; echo; swapon --show; zramctl 2>/dev/null'
run 03_df               'df -hT -x tmpfs -x devtmpfs -x squashfs -x overlay'
run 03_lsblk            'lsblk -o NAME,SIZE,TYPE,FSTYPE,MODEL,SERIAL,MOUNTPOINT'
run 03_nvme             'ls -l /dev/nvme* 2>/dev/null; for d in /sys/class/nvme/nvme*; do echo "$(basename $d) model=$(cat $d/model 2>/dev/null) fw=$(cat $d/firmware_rev 2>/dev/null)"; done; grep -E "nvme|mmcblk" /proc/mounts'
if [ "$DO_SUDO" = "1" ]; then run 03_nvme_list 'sudo -n nvme list'; fi
run 03_home_usage       'du -sh $HOME 2>/dev/null; ls -la $HOME'
fi

# ---------------------------------------------------------------- 4. network
if want 4; then
say "network"
run 04_ip_addr          'ip -br addr; echo; ip addr'
run 04_ip_route         'ip route; echo; ip -6 route'
run 04_ip_neigh         'ip neigh'
run 04_link_speed       'for n in /sys/class/net/*; do echo "$(basename $n) operstate=$(cat $n/operstate 2>/dev/null) speed=$(cat $n/speed 2>/dev/null) mtu=$(cat $n/mtu) mac=$(cat $n/address)"; done'
run 04_ethtool          'for n in $(ls /sys/class/net); do [ "$n" = lo ] && continue; echo "== $n"; ethtool "$n" 2>&1 | grep -E "Speed|Duplex|Link detected"; done'
run 04_wifi             'rfkill list; echo; iw dev; echo; nmcli -t -f DEVICE,TYPE,STATE,CONNECTION device 2>/dev/null'
run 04_sockets          'ss -tulpn'
run 04_resolv           'cat /etc/resolv.conf; cat /etc/hosts'
run 04_tailscale        'command -v tailscale && tailscale status'
if [ "$DO_PING" = "1" ]; then
  for ip in 192.168.123.161 192.168.123.164; do run "04_ping_${ip}" "ping -c 3 -W 1 $ip"; done
fi
fi

# ---------------------------------------------------------------- 5. time
if want 5; then
say "time"
run 05_date             'date -u +%Y-%m-%dT%H:%M:%S.%NZ; date'
run 05_timedatectl      'timedatectl status; echo; timedatectl show'
run 05_chrony           'chronyc tracking; echo; chronyc sources -v; echo; chronyc sourcestats'
run 05_time_services    'for s in chrony chronyd systemd-timesyncd ntp ntpd ptp4l phc2sys; do echo "$s $(systemctl is-active $s 2>/dev/null) $(systemctl is-enabled $s 2>/dev/null)"; done; ls /dev/ptp* 2>/dev/null'
fi

# ---------------------------------------------------------------- 6. processes, services, vendor software
if want 6; then
say "services"
run 06_services_running 'systemctl list-units --type=service --state=running --no-pager --plain'
run 06_services_enabled 'systemctl list-unit-files --type=service --state=enabled --no-pager --plain'
run 06_services_interest 'for u in $(systemctl list-units --type=service --all --no-pager --plain --no-legend | awk "{print \$1}" | grep -iE "unitree|ros|dds|docker|realsense|livox|lidar|teleimager|image|camera|chrony|ntp|nvargus|jtop|nvpmodel|motion|robot|lowlevel|sport|ota|tailscale"); do echo "=================== $u"; systemctl status "$u" --no-pager -n 5 2>&1 | head -20; systemctl cat "$u" --no-pager 2>&1 | grep -E "^(ExecStart|User|WorkingDirectory|Environment|After|Restart)=" ; done'
run 06_ps               'ps -eo pid,user,%cpu,%mem,etime,comm,args --sort=-%cpu | head -60'
run 06_docker           'docker ps -a; echo; docker images'
run 06_cron             'crontab -l; ls -la /etc/cron.d'
run 06_vendor_paths     'ls -la / /opt /opt/ros /opt/unitree_robotics /unitree 2>/dev/null; find / -xdev -maxdepth 4 \( -iname "*unitree*" -o -iname "*livox*" -o -iname "*teleimager*" -o -iname "*realsense*" -o -iname "*cyclonedds*" \) -not -path "/proc/*" -not -path "/sys/*" 2>/dev/null | head -100'
run 06_home_dirs        'ls -d $HOME/*/ 2>/dev/null; for d in $HOME/unitree_ros2 $HOME/unitree_sdk2 $HOME/unitree_sdk2_python $HOME/teleimager $HOME/xr_teleoperate; do [ -d "$d" ] && (echo "== $d"; git -C "$d" log -1 --format="%H %cd %s" 2>/dev/null; git -C "$d" remote -v 2>/dev/null | head -1); done; true'
run 06_unitree_ros2_setup 'for f in $HOME/unitree_ros2/setup.sh $HOME/unitree_ros2/setup_local.sh; do [ -f "$f" ] && (echo "== $f"; cat "$f"); done; true'
fi

# ---------------------------------------------------------------- 7. DDS / ROS 2 environment
if want 7; then
say "DDS / ROS environment"
run 07_env              'env | grep -E "^(ROS_|RMW_|CYCLONEDDS|CYCLONE|FASTRTPS|FASTDDS|AMENT|COLCON|LD_LIBRARY_PATH|PYTHONPATH|CUDA|TensorRT|CONDA)" | sort'
run 07_bashrc_dds       'grep -nE "ROS|RMW|CYCLONE|DDS|source |conda" $HOME/.bashrc 2>/dev/null'
run 07_cyclonedds_uri   'echo "CYCLONEDDS_URI=$CYCLONEDDS_URI"; case "$CYCLONEDDS_URI" in file://*) cat "${CYCLONEDDS_URI#file://}";; /*) cat "$CYCLONEDDS_URI";; esac; true'

if [ "$DO_ROS" = "1" ]; then
  if [ -z "$ROS_DISTRO" ]; then
    for d in humble jazzy foxy; do
      if [ -f "/opt/ros/$d/setup.bash" ]; then source "/opt/ros/$d/setup.bash"; echo "sourced /opt/ros/$d/setup.bash" >> "$OUT/_meta.txt"; break; fi
    done
  fi
  run 07_ros_env        'echo "ROS_DISTRO=$ROS_DISTRO ROS_DOMAIN_ID=$ROS_DOMAIN_ID RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY"; command -v ros2; ros2 pkg list 2>/dev/null | grep -iE "unitree|livox|realsense|rmw|cyclone|foxglove" '
  if have ros2; then
    say "ros2 lists (DDS discovery only)"
    run 07_ros2_nodes     'ros2 node list --no-daemon --spin-time 5 2>/dev/null || ros2 node list'
    run 07_ros2_topics    'ros2 topic list -t --no-daemon --spin-time 5 2>/dev/null || ros2 topic list -t'
    run 07_ros2_services  'ros2 service list -t --no-daemon --spin-time 5 2>/dev/null || ros2 service list -t'
    run 07_ros2_actions   'ros2 action list -t'
    # topic info for every discovered topic (publisher/subscriber counts, QoS); capped at 80 topics
    TOPICS="$(sed -n '3,$p' "$OUT/07_ros2_topics.txt" | awk '{print $1}' | grep '^/' | head -80)"
    : > "$OUT/07_ros2_topic_info.txt"
    for tp in $TOPICS; do
      { echo "=================== $tp"; timeout 10 ros2 topic info -v "$tp" --no-daemon 2>&1 || timeout 10 ros2 topic info -v "$tp" 2>&1; } >> "$OUT/07_ros2_topic_info.txt"
    done
    echo "07_ros2_topic_info 0" >> "$OUT/_exit_codes.txt"
  else
    echo "ros2 not found after sourcing" > "$OUT/07_ros2_nodes.txt"
  fi
  if have cyclonedds; then
    run 07_cyclonedds_ps 'cyclonedds ps'
  else
    echo "cyclonedds CLI not found (pip package cyclonedds provides it)" > "$OUT/07_cyclonedds_ps.txt"
  fi
fi
fi

# ---------------------------------------------------------------- 8. python and packages
if want 8; then
say "python packages"
PYSNIP="$OUT/_pyprobe.py"
cat > "$PYSNIP" <<'PYEOF'
import importlib, importlib.metadata as md, json, sys, time
out = {"executable": sys.executable, "version": sys.version, "dists": {}, "modules": {}}
for d in ["cyclonedds", "unitree_sdk2py", "unitree-sdk2py", "onnxruntime", "onnxruntime-gpu", "tensorrt", "pyrealsense2",
          "opencv-python", "opencv-contrib-python", "opencv-python-headless", "numpy", "torch", "pyzmq", "foxglove-sdk",
          "PyYAML", "jetson-stats", "lerobot", "teleimager", "pinocchio", "mujoco"]:
    try:
        out["dists"][d] = md.version(d)
    except Exception:
        pass
for m in ["rclpy", "cyclonedds", "unitree_sdk2py", "onnxruntime", "tensorrt", "pyrealsense2", "cv2", "numpy", "torch", "zmq", "yaml"]:
    t0 = time.time()
    try:
        mod = importlib.import_module(m)
        info = {"ok": True, "version": str(getattr(mod, "__version__", "")), "file": str(getattr(mod, "__file__", ""))}
        if m == "onnxruntime":
            info["providers"] = list(mod.get_available_providers())
        if m == "cv2":
            bi = mod.getBuildInformation()
            info["cuda_in_build"] = "CUDA:" in bi and "NVIDIA CUDA:                   YES" in bi
            info["gstreamer"] = [l.strip() for l in bi.splitlines() if "GStreamer" in l][:1]
        if m == "torch":
            info["cuda_available"] = bool(mod.cuda.is_available())
    except Exception as e:
        info = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    info["import_s"] = round(time.time() - t0, 2)
    out["modules"][m] = info
print(json.dumps(out, indent=1))
PYEOF
PYS=()
for p in python3 /usr/bin/python3 "${EXTRA_PY[@]}" $HOME/miniconda3/envs/*/bin/python $HOME/anaconda3/envs/*/bin/python $HOME/miniforge3/envs/*/bin/python; do
  if [ -x "$(command -v "$p" 2>/dev/null)" ]; then PYS+=("$(command -v "$p")"); fi
done
i=0
for p in $(printf "%s\n" "${PYS[@]}" | awk '!seen[$0]++'); do
  i=$((i+1))
  T_SAVE="$T"; T=$(( T > 90 ? T : 90 ))
  # not -I: user-site and PYTHONPATH installs (ROS rclpy, user-site pip packages) must stay visible
  run "08_python_${i}" "'$p' '$PYSNIP' 2>&1"
  T="$T_SAVE"
done
run 08_pip_list         'python3 -m pip list 2>/dev/null | head -300'
run 08_conda            'command -v conda && conda env list; ls $HOME/miniconda3/envs $HOME/anaconda3/envs 2>/dev/null'
fi

# ---------------------------------------------------------------- 9. USB, cameras, audio, serial
if want 9; then
say "devices"
run 09_lsusb            'lsusb; echo; lsusb -t'
run 09_dev_nodes        'ls -l /dev/video* /dev/media* /dev/ttyUSB* /dev/ttyACM* /dev/ttyTHS* /dev/i2c-* /dev/spidev* 2>/dev/null; ls -l /dev/serial/by-id 2>/dev/null; true'
run 09_v4l_sysfs        'for v in /sys/class/video4linux/*; do echo "$(basename $v) name=$(cat $v/name 2>/dev/null) index=$(cat $v/index 2>/dev/null) dev=$(readlink -f $v/device)"; done'
if have v4l2-ctl; then
  run 09_v4l2_list      'v4l2-ctl --list-devices'
  run 09_v4l2_formats   'for d in /dev/video*; do echo "=================== $d"; v4l2-ctl -d "$d" --info --list-formats-ext 2>&1 | head -80; done'
else
  echo "v4l2-ctl not found (apt package v4l-utils)" > "$OUT/09_v4l2_list.txt"
fi
if have rs-enumerate-devices; then
  run 09_rs_enumerate_short 'rs-enumerate-devices -s'
  run 09_rs_enumerate_calib 'rs-enumerate-devices -c'
  run 09_rs_enumerate_full  'rs-enumerate-devices'
else
  echo "rs-enumerate-devices not found (librealsense2-utils); camera_probe.py uses pyrealsense2 instead" > "$OUT/09_rs_enumerate_short.txt"
fi
run 09_audio            'arecord -l; echo; aplay -l'
run 09_dmesg_devices    'dmesg 2>&1 | grep -iE "realsense|uvcvideo|usb .*new|livox|eth|r8169|igb|nvme" | tail -60'
run 09_i2c_gpio         'ls /sys/bus/i2c/devices 2>/dev/null | head -50'
fi

# ---------------------------------------------------------------- done
echo "utc_end: $(date -u +%Y%m%dT%H%M%SZ)" >> "$OUT/_meta.txt"
nfail=$(awk '$2 != 0' "$OUT/_exit_codes.txt" | wc -l)
tar -C "$OUTROOT" -czf "$OUT.tar.gz" "$(basename "$OUT")"
say "done: $(ls "$OUT" | wc -l) files, $nfail commands returned non-zero (normal for tools that are not installed)"
say "report folder: $OUT"
say "archive:       $OUT.tar.gz   <- send this file back"
