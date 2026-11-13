"""Web UI API over real HTTP on 127.0.0.1 (ephemeral port): token auth, STOP without token, arming flow, owner rule,
log download whitelist, camera PNG, static files without CDN, default bind address."""

import json
import re
import struct
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import pytest

from humanoid_arc_bme.onboard.runtime.camera import FakeCamera
from humanoid_arc_bme.onboard.runtime.cli import build_operator, main as cli_main
from humanoid_arc_bme.onboard.runtime.web.server import STATIC, WebApp, png_from_frame

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "t" * 24


class Args:
    io, iface, models, shadow, lan, port, token, allow_lowcmd, config_dir, camera, foxglove = "fake", None, None, None, False, 0, TOKEN, False, None, "fake", None

    def __init__(self, tmp):
        self.log_root = str(tmp)


@pytest.fixture()
def server(tmp_path):
    rt, app, token = build_operator(Args(tmp_path))
    app.make_server()
    rt.start()
    app.start()
    yield rt, app, f"http://127.0.0.1:{app.port}"
    rt.close()
    app.close()


def req(url, body=None, token=TOKEN, client="c1"):
    h = {"X-Client": client}
    if token:
        h["Authorization"] = f"Bearer {token}"
    data = None
    if body is not None:
        data, h["Content-Type"] = json.dumps(body).encode(), "application/json"
    r = urllib.request.Request(url, data=data, headers=h, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(r, timeout=5) as resp:
            return resp.status, resp.read(), resp.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


def wait_green(base, hb_client="c1", seconds=3.0):
    t0 = time.time()
    while time.time() - t0 < seconds:
        req(base + "/api/heartbeat", {"mode": "teleop", "vx": 0, "vy": 0, "wz": 0}, client=hb_client)
        s = json.loads(req(base + "/api/status")[1])
        if s["preflight_green"]:
            return s
        time.sleep(0.12)  # stay under the per-IP rate limit (UI-07)
    return s


def test_default_bind_is_localhost(tmp_path):
    rt, app, token = build_operator(Args(tmp_path))
    assert app.host == "127.0.0.1" and len(token) >= 16
    a = Args(tmp_path)
    a.lan = True
    with pytest.raises(SystemExit):  # UI-05: --lan alone is refused (plain HTTP on 0.0.0.0)
        build_operator(a)
    a.lan_plain_http_ack = True
    rt2, app2, _ = build_operator(a)
    assert app2.host == "0.0.0.0"
    rt.close()
    rt2.close()
    with pytest.raises(ValueError):
        WebApp(rt, "short")


def test_static_page_has_no_external_urls_and_is_served(server):
    _rt, _app, base = server
    for f in ("index.html", "app.js", "style.css"):
        text = (STATIC / f).read_text()
        assert not re.search(r"(src|href)\s*=\s*[\"']?(https?:)?//", text), f  # no CDN / external resources
    code, body, h = req(base + "/", token=None)
    assert code == 200 and b"<title>G1 Operator</title>" in body and "default-src 'self'" in h["Content-Security-Policy"]


def test_api_requires_token(server):
    rt, _app, base = server
    t = [0.0]

    def slow_clock():  # every request is 1 s after the last: this test is about the token, not the rate limit
        t[0] += 1.0
        return t[0]
    _app.clock = slow_clock
    for path in ("/api/status", "/api/plots", "/api/logs"):
        assert req(base + path, token=None)[0] == 401
        assert req(base + path, token="wrong" * 5)[0] == 401
    for path in ("/api/heartbeat", "/api/arm/request", "/api/disarm", "/api/damp", "/api/controller", "/api/reset_stop"):
        assert req(base + path, {}, token=None)[0] == 401
    assert req(base + "/api/status?token=" + TOKEN, token=None)[0] == 200  # GET may carry ?token= (img src, downloads)
    assert req(base + "/api/heartbeat?token=" + TOKEN, {}, token=None)[0] == 401  # POST may not
    assert req(base + "/api/status")[0] == 200


def test_stop_works_without_token_and_latches(server):
    rt, _app, base = server
    code, body, _ = req(base + "/api/stop", {}, token=None)
    assert code == 200 and rt.sup.stop_latched
    assert "stop:ui_noauth" in rt.sup.reason
    code, body, _ = req(base + "/api/arm/request", {})
    assert code == 409 and b"latched" in body


def test_arm_flow_typed_confirmation_and_owner(server):
    rt, _app, base = server
    s = wait_green(base)
    assert s["preflight_green"], [c for c in s["preflight"] if c["status"] != "ok"]
    code, body, _ = req(base + "/api/arm/request", {})
    phrase = json.loads(body)["type_this"]
    code, body, _ = req(base + "/api/arm/confirm", {"text": "ARM XXXX"})
    assert code == 409 and not rt.sup.armed
    req(base + "/api/arm/request", {})
    code, body, _ = req(base + "/api/heartbeat", {"mode": "teleop", "vx": 0, "vy": 0, "wz": 0})
    phrase = json.loads(req(base + "/api/arm/request", {})[1])["type_this"]
    code, body, _ = req(base + "/api/arm/confirm", {"text": phrase})
    assert code == 200 and json.loads(body)["ok"] and rt.sup.armed
    code, body, _ = req(base + "/api/heartbeat", {"mode": "teleop", "vx": 0.1}, client="intruder")
    assert code == 409  # another client cannot drive
    code, body, _ = req(base + "/api/controller", {"name": "vendor"})
    assert code == 409  # no controller switch while armed
    code, body, _ = req(base + "/api/heartbeat", {"mode": "teleop", "vx": 9.0, "vy": 0, "wz": 0, "held": True})
    assert code == 400 and rt.sup.armed  # out of the envelope: rejected (UI-04), not clipped
    code, body, _ = req(base + "/api/heartbeat", {"mode": "teleop", "vx": "nan", "vy": 0, "wz": 0, "held": True})
    assert code == 400
    code, body, _ = req(base + "/api/heartbeat", {"mode": "teleop", "vx": rt.lim.vx_max, "vy": 0, "wz": 0})  # not held
    assert code == 200 and json.loads(body)["target"] == [0.0, 0.0, 0.0]  # heartbeat alone never authorises motion (UI-03)
    time.sleep(0.6)  # no more heartbeats: deadman
    assert not rt.sup.armed and rt.sup.reason == "deadman_timeout", (rt.sup.reason, rt.fault, rt.fault_detail, rt.state_age)


def test_bad_json_and_unknown_mode_are_400(server):
    _rt, _app, base = server
    r = urllib.request.Request(base + "/api/heartbeat", data=b"{not json", method="POST",
                               headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(r, timeout=5)
    assert e.value.code == 400
    assert req(base + "/api/heartbeat", {"mode": "fly"})[0] == 400
    if _rt.nav is None:
        assert req(base + "/api/heartbeat", {"mode": "waypoints"})[0] == 400  # no nav module
    else:
        assert req(base + "/api/waypoints", {"waypoints": []})[0] == 400
        assert req(base + "/api/waypoints", {"waypoints": [[1.0, 0.0]]})[0] == 200


def test_logs_whitelist_and_download(server):
    rt, _app, base = server
    files = json.loads(req(base + "/api/logs")[1])
    assert {"events.jsonl", "telemetry.mcap"} <= {f["name"] for f in files}
    code, body, h = req(base + "/api/logs/events.jsonl")
    assert code == 200 and b'"startup"' in body and h["Content-Disposition"] == "attachment"
    assert req(base + "/api/logs/..%2F..%2Fetc%2Fpasswd")[0] == 404
    assert req(base + "/api/logs/../../config/g1_limits.yaml")[0] == 404


def test_camera_png(server):
    rt, _app, base = server
    t0 = time.time()
    while "head" not in rt.frames and time.time() - t0 < 2:
        time.sleep(0.05)
    code, body, h = req(base + "/api/camera/head.png")
    assert code == 200 and body.startswith(b"\x89PNG") and h["Content-Type"] == "image/png"
    assert req(base + "/api/camera/nope.png")[0] == 404


def test_png_encoder_decodes():
    fr = FakeCamera(lambda: 0.5, width=640, height=48).read()
    png = png_from_frame(fr, max_width=320)
    w, h = struct.unpack(">II", png[16:24])
    assert (w, h) == (320, 24)
    idat_len = struct.unpack(">I", png[33:37])[0]
    raw = zlib.decompress(png[41:41 + idat_len])
    assert len(raw) == h * (1 + 3 * w) and raw[1:4] == fr.data[0:3]


def test_preflight_cli_runs_read_only(capsys, tmp_path):
    code = cli_main(["preflight", "--log-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "limits" in out and "required not ok" in out and code in (0, 1)  # here: unitree_sdk2py missing -> 1


def test_rate_limit_and_stop_exempt(server):
    rt, app, base = server
    codes = [req(base + "/api/status")[0] for _ in range(80)]
    assert 429 in codes and codes[0] == 200
    assert req(base + "/api/stop", {}, token=None)[0] == 200  # STOP is never rate limited


def test_stop_confirmed_only_when_disarmed_by_the_server(server):
    rt, app, base = server
    code, body, _ = req(base + "/api/stop", {}, token=None)
    j = json.loads(body)
    assert code == 200 and j["confirmed"] is True and j["state"] == "disarmed" and j["stop_latched"]
