"""Command gateway: auth, the stop exemption, confirm/expiry/cancel, sentence routing, limits, and failure behaviour.

The robot is a fake CommandService that records what it was asked to do; the parser is the real LocalGrammarParser.
Clocks are injected and frozen unless a test advances them, so rate limits are deterministic.
"""

import math
import threading

import pytest
from fastapi.testclient import TestClient

from humanoid_arc_bme.onboard.command import LocalGrammarParser, Move, Stop, Turn
from humanoid_arc_bme.onboard.runtime import command_gateway as gw
from humanoid_arc_bme.onboard.runtime.command_gateway import check_bind_host, create_app, serve
from humanoid_arc_bme.onboard.stack import RobotStack
from humanoid_arc_bme.onboard.task.executor import FAILED, RUNNING, TaskStatus

TOKEN = "gateway-test-token-0123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
TTL = 20.0


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeService:
    """Records every call. `ready_reason` set means not ready; `stop_error`/`parse_error`/`submit_error` inject faults."""

    def __init__(self) -> None:
        self.parser = LocalGrammarParser()
        self.submitted: list = []
        self.stops: list[str] = []
        self.ready_reason: str | None = None
        self.stop_error: Exception | None = None
        self.parse_error: Exception | None = None
        self.submit_error: Exception | None = None

    def parse(self, text):
        if self.parse_error is not None:
            raise self.parse_error
        return self.parser.parse(text)

    def submit(self, cmd):
        if self.submit_error is not None:
            raise self.submit_error
        self.submitted.append(cmd)
        return TaskStatus(state=RUNNING, phase="walk", message="walking", command="move")

    def stop(self, why):
        self.stops.append(why)
        if self.stop_error is not None:
            raise self.stop_error

    def status(self):
        return {"state": "idle", "phase": "", "message": "", "command": "", "safety": "ok", "reasons": []}

    def scene(self):
        return [{"label": "cup", "color": "red", "range_m": 1.2, "confidence": 0.9, "position_body": [1.0, 0.5, 0.0]}]

    def ready(self):
        return (False, self.ready_reason) if self.ready_reason else (True, "")


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def service() -> FakeService:
    return FakeService()


def make_client(service, clock, rate_per_s: int = 20) -> TestClient:
    app = create_app(service, TOKEN, confirm_ttl_s=TTL, clock=clock, rate_per_s=rate_per_s)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client(service, clock) -> TestClient:
    return make_client(service, clock)


def command(client, text: str, headers=AUTH):
    return client.post("/api/command", json={"text": text}, headers=headers)


def pending_id(client) -> str:
    return command(client, "go forward 2 meters").json()["id"]


# ----------------------------------------------------------------- stop: no token, no limits

def test_stop_works_without_token(client, service):
    r = client.post("/api/stop")
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert service.stops


def test_stop_bypasses_rate_limit(service, clock):
    client = make_client(service, clock, rate_per_s=1)
    assert [client.post("/api/stop").status_code for _ in range(10)] == [200] * 10
    assert len(service.stops) == 10


def test_stop_answers_200_when_failed_auth_is_flooded(client, service):
    for _ in range(10):
        client.get("/api/status", headers={"Authorization": "Bearer wrong"})
    assert client.post("/api/stop").status_code == 200
    assert service.stops


def test_stop_answers_200_and_logs_when_service_fails(service, clock, caplog):
    service.stop_error = RuntimeError("bus down")
    r = make_client(service, clock).post("/api/stop")
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert "stop failed" in caplog.text


# ----------------------------------------------------------------- authentication

PROTECTED = [
    ("GET", "/api/status", None),
    ("GET", "/api/scene", None),
    ("POST", "/api/command", {"text": "pick up the red cup"}),
    ("POST", "/api/confirm", {"id": "abc"}),
    ("POST", "/api/cancel", None),
]


@pytest.mark.parametrize("method,path,body", PROTECTED)
def test_protected_routes_need_a_token(client, service, method, path, body):
    assert client.request(method, path, json=body).status_code == 401
    wrong = client.request(method, path, json=body, headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401
    no_scheme = client.request(method, path, json=body, headers={"Authorization": TOKEN})
    assert no_scheme.status_code == 401
    assert service.submitted == [] and service.stops == []


def test_status_with_token(client):
    r = client.get("/api/status", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"state": "idle", "phase": "", "message": "", "command": "", "safety": "ok", "reasons": [],
                        "pending": None}


def test_scene_with_token(client):
    r = client.get("/api/scene", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["objects"][0]["label"] == "cup"


# ----------------------------------------------------------------- command -> pending -> confirm

def test_command_becomes_pending_and_does_not_execute(client, service):
    r = command(client, "go forward 2 meters")
    body = r.json()
    assert r.status_code == 200
    assert body["accepted"] is True and body["needs_confirm"] is True
    assert body["command"] == {"intent": "move", "forward_m": 2.0, "left_m": 0.0}
    assert body["expires_in_s"] == TTL
    assert service.submitted == []
    pending = client.get("/api/status", headers=AUTH).json()["pending"]
    assert pending["id"] == body["id"] and pending["text"] == "go forward 2 meters"


def test_confirm_executes_exactly_once(client, service):
    cid = pending_id(client)
    r = client.post("/api/confirm", json={"id": cid}, headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"executed": True, "status": {"state": "running", "phase": "walk", "message": "walking",
                                                     "command": "move"}}
    assert service.submitted == [Move(forward_m=2.0, left_m=0.0)]
    again = client.post("/api/confirm", json={"id": cid}, headers=AUTH)
    assert again.status_code == 404
    assert len(service.submitted) == 1


def test_confirm_with_wrong_id_keeps_pending(client, service):
    cid = pending_id(client)
    r = client.post("/api/confirm", json={"id": "not-the-id"}, headers=AUTH)
    assert r.status_code == 404
    assert service.submitted == []
    assert client.get("/api/status", headers=AUTH).json()["pending"]["id"] == cid


def test_confirm_after_expiry_is_410_and_clears_pending(client, service, clock):
    cid = pending_id(client)
    clock.now += TTL
    r = client.post("/api/confirm", json={"id": cid}, headers=AUTH)
    assert r.status_code == 410
    assert service.submitted == []
    assert client.get("/api/status", headers=AUTH).json()["pending"] is None


def test_confirm_just_before_expiry_still_executes(client, service, clock):
    cid = pending_id(client)
    clock.now += TTL - 0.5
    assert client.post("/api/confirm", json={"id": cid}, headers=AUTH).status_code == 200
    assert len(service.submitted) == 1


def test_new_command_replaces_pending(client, service):
    old = pending_id(client)
    new = command(client, "turn left 90 degrees").json()["id"]
    assert client.post("/api/confirm", json={"id": old}, headers=AUTH).status_code == 404
    assert client.post("/api/confirm", json={"id": new}, headers=AUTH).status_code == 200
    assert len(service.submitted) == 1
    assert isinstance(service.submitted[0], Turn)
    assert math.isclose(service.submitted[0].angle_rad, math.pi / 2)


def test_cancel_drops_pending(client, service):
    cid = pending_id(client)
    assert client.post("/api/cancel", headers=AUTH).json() == {"ok": True}
    assert client.post("/api/confirm", json={"id": cid}, headers=AUTH).status_code == 404
    assert service.submitted == []


def test_pick_sentence_shows_readable_command(client):
    body = command(client, "pick up the red cup").json()
    assert body["command"] == {"intent": "pick", "label": "cup", "color": "red"}


# ----------------------------------------------------------------- sentence routing

@pytest.mark.parametrize("text", ["please stop", "STOP!!!"])
def test_stop_sentence_executes_immediately(client, service, text):
    r = command(client, text)
    assert r.status_code == 200
    assert r.json()["executed"] is True and r.json()["command"] == {"intent": "stop"}
    assert service.stops and service.submitted == []
    assert client.get("/api/status", headers=AUTH).json()["pending"] is None


def test_status_sentence_is_answered_not_queued(client, service):
    body = command(client, "what do you see").json()
    assert body["accepted"] is True and body["executed"] is True
    assert body["answer"]["state"] == "idle"
    assert service.submitted == []
    assert client.get("/api/status", headers=AUTH).json()["pending"] is None


def test_reject_is_not_accepted(client, service):
    body = command(client, "   ").json()
    assert body["accepted"] is False and body["reason"]
    assert service.submitted == []


def test_reject_does_not_replace_pending(client):
    cid = pending_id(client)
    command(client, "   ")
    assert client.get("/api/status", headers=AUTH).json()["pending"]["id"] == cid


# ----------------------------------------------------------------- request validation

@pytest.mark.parametrize("body", [
    {"text": "x" * 301},
    {"text": "stop", "extra": 1},
    {"text": 42},
    {"text": None},
    {},
])
def test_invalid_command_body_is_422(client, body):
    assert client.post("/api/command", json=body, headers=AUTH).status_code == 422


@pytest.mark.parametrize("body", [{"id": "a" * 65}, {"id": ""}, {"id": "x", "extra": True}, {"id": 7}, {}])
def test_invalid_confirm_body_is_422(client, body):
    assert client.post("/api/confirm", json=body, headers=AUTH).status_code == 422


def test_text_of_exactly_300_chars_is_accepted(client):
    text = "go forward 2 meters".ljust(300)
    assert len(text) == 300
    assert command(client, text).status_code == 200


# ----------------------------------------------------------------- rate limits

def test_authorised_requests_are_rate_limited(service, clock):
    client = make_client(service, clock, rate_per_s=3)
    codes = [client.get("/api/status", headers=AUTH).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    clock.now += 1.0
    assert client.get("/api/status", headers=AUTH).status_code == 200


def test_failed_auth_flood_is_429_after_five(client, clock):
    wrong = {"Authorization": "Bearer wrong"}
    codes = [client.get("/api/status", headers=wrong).status_code for _ in range(7)]
    assert codes == [401] * 5 + [429, 429]
    # While the failure window is full, even the right token is refused: guesses cannot be compared at request rate.
    assert client.get("/api/status", headers=AUTH).status_code == 429
    clock.now += 1.0
    assert client.get("/api/status", headers=AUTH).status_code == 200


def test_failed_auth_flood_does_not_stop_the_robot(client):
    for _ in range(10):
        client.get("/api/status", headers={"Authorization": "Bearer wrong"})
    assert client.post("/api/stop").status_code == 200


# ----------------------------------------------------------------- readiness and failures

def test_not_ready_gives_503_with_reason_but_stop_works(service, clock):
    service.ready_reason = "control loop is down"
    client = make_client(service, clock)
    assert command(client, "go forward 2 meters").json() == {"detail": "control loop is down"}
    assert command(client, "go forward 2 meters").status_code == 503
    assert client.post("/api/confirm", json={"id": "x"}, headers=AUTH).status_code == 503
    assert client.post("/api/stop").status_code == 200
    assert client.get("/api/status", headers=AUTH).status_code == 200


def test_parse_failure_is_500_and_robot_not_commanded(service, clock):
    service.parse_error = RuntimeError("grammar broke")
    r = command(make_client(service, clock), "go forward 2 meters")
    assert r.status_code == 500
    assert r.json() == {"detail": "internal error"}
    assert "grammar broke" not in r.text
    assert service.submitted == []


def test_submit_failure_is_500_and_pending_is_consumed(service, clock):
    client = make_client(service, clock)
    cid = pending_id(client)
    service.submit_error = RuntimeError("arm fault")
    r = client.post("/api/confirm", json={"id": cid}, headers=AUTH)
    assert r.status_code == 500 and r.json() == {"detail": "internal error"}
    service.submit_error = None
    assert client.post("/api/confirm", json={"id": cid}, headers=AUTH).status_code == 404
    assert service.submitted == []


def test_responses_carry_security_headers(client):
    for r in (client.post("/api/stop"), client.get("/api/status"), client.get("/api/status", headers=AUTH)):
        assert r.headers["Cache-Control"] == "no-store"
        assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_token_never_appears_in_a_response(client, service):
    cid = pending_id(client)
    responses = [
        client.get("/api/status", headers=AUTH),
        client.get("/api/status", headers={"Authorization": f"Bearer {TOKEN}x"}),
        client.get("/api/status"),
        client.post("/api/command", json={"text": TOKEN * 3}, headers=AUTH),
        client.post("/api/command", json={"text": "x" * 400}, headers=AUTH),
        client.post("/api/confirm", json={"id": "nope"}, headers=AUTH),
        client.post("/api/confirm", json={"id": cid}, headers=AUTH),
        client.post("/api/stop"),
    ]
    for r in responses:
        assert TOKEN not in r.text
        assert TOKEN not in " ".join(r.headers.values())


def test_stack_command_service_without_sensor_data():
    service = gw.StackCommandService(RobotStack(), lambda: None, threading.Lock())
    assert service.ready() == (False, "no sensor data yet")
    assert service.scene() == []
    assert service.submit(Move(forward_m=1.0, left_m=0.0)).state == FAILED
    assert set(service.status()) == {"state", "phase", "message", "command", "safety", "reasons"}
    assert service.parse("stop") == Stop()


# ----------------------------------------------------------------- app construction and bind checks

@pytest.mark.parametrize("kwargs", [{"rate_per_s": 0}, {"confirm_ttl_s": 0.0}])
def test_create_app_rejects_bad_limits(service, kwargs):
    with pytest.raises(ValueError):
        create_app(service, TOKEN, **kwargs)


def test_create_app_rejects_empty_token(service):
    with pytest.raises(ValueError):
        create_app(service, "")


@pytest.mark.parametrize("host", ["127.0.0.1", "127.8.9.10", "localhost", "::1", "100.64.0.1", "100.127.255.254",
                                  "100.100.100.100"])
def test_check_bind_host_accepts_loopback_and_tailscale(host):
    check_bind_host(host)


@pytest.mark.parametrize("host", ["192.168.1.5", "0.0.0.0", "8.8.8.8", "100.128.0.1", "100.63.255.255", "example.org",
                                  "fd7a:115c:a1e0::1"])
def test_check_bind_host_refuses_other_hosts(host):
    with pytest.raises(ValueError):
        check_bind_host(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "8.8.8.8"])
def test_check_bind_host_allow_any_host_overrides(host):
    check_bind_host(host, allow_any_host=True)


def test_serve_refuses_a_bad_host_without_starting_uvicorn(monkeypatch, service):
    def fail(*args, **kwargs):
        raise AssertionError("uvicorn must not start for a refused host")

    monkeypatch.setattr(gw.uvicorn, "run", fail)
    with pytest.raises(ValueError):
        serve(service, TOKEN, host="192.168.1.5")
    with pytest.raises(ValueError):
        serve(service, TOKEN, host="0.0.0.0")
