// G1 operator UI. No external libraries, no inline styles (CSP). One page for two servers:
//   "robot": onboard/runtime/web/server.py (GET /api/status, typed-phrase arming)   "sim": isaac/ui/server.py (heartbeat returns the state)
// Safety behaviour kept from the previous UI: STOP needs no token and is retried until confirmed; the robot moves only while an input is
// held; a silent browser tab stops the heartbeat and the deadman disarms; arming needs a deliberate action (slide) and green checks.
"use strict";
(function () {
  const $ = (id) => document.getElementById(id);
  const clientId = "ui-" + Math.random().toString(36).slice(2, 10);
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const fmt = (x, d = 2) => (x === null || x === undefined || !isFinite(x) ? "-" : Number(x).toFixed(d));
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };

  // ------------------------------------------------------------ token + api
  let token = "";
  const vm = location.hash.match(/view=(drive|sensors|safety)/); // deep link, e.g. #token=...&view=sensors
  const m = location.hash.match(/token=([^&]+)/);
  if (m) {
    token = decodeURIComponent(m[1]);
    try { sessionStorage.setItem("g1token", token); } catch (e) { /* storage blocked */ }
    history.replaceState(null, "", location.pathname);
  } else {
    try { token = sessionStorage.getItem("g1token") || ""; } catch (e) { token = ""; }
  }
  if (!token) $("tokencard").hidden = false;
  $("tokenset").onclick = () => {
    token = $("tokenin").value.trim();
    try { sessionStorage.setItem("g1token", token); } catch (e) { /* ignore */ }
    $("tokencard").hidden = !token;
  };

  async function api(path, body, timeoutMs) {
    const opt = { method: body === undefined ? "GET" : "POST", headers: { "Authorization": "Bearer " + token, "X-Client": clientId }, cache: "no-store" };
    if (body !== undefined) { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), timeoutMs || 2000);
    opt.signal = ac.signal;
    try {
      const r = await fetch(path, opt);
      let j = null;
      try { j = await r.json(); } catch (e) { j = null; }
      if (r.status === 401) $("tokencard").hidden = false;
      return { status: r.status, ok: r.ok, data: j };
    } catch (e) {
      return { status: 0, ok: false, data: null };
    } finally { clearTimeout(timer); }
  }

  // ------------------------------------------------------------ backend detection: the robot server answers /api/ping, the simulator does not
  let backend = null; // "robot" | "sim"
  async function detect() {
    try { const r = await fetch("/api/ping", { cache: "no-store" }); if (r.ok) return "robot"; } catch (e) { /* fall through */ }
    return "sim";
  }

  // ------------------------------------------------------------ STOP: always works, also without the token
  let stopSeq = 0;
  const sm = $("stopmsg");
  function stopNote(text, cls) { sm.hidden = false; sm.textContent = text; sm.className = "stopmsg " + cls; }
  async function stop() {
    pad = null; keys.clear(); holdBtn = false; testWalk = null;
    const seq = ++stopSeq;
    stopNote("Sending STOP...", "stopwait");
    const warn = setTimeout(() => { if (seq === stopSeq && !sm.className.includes("stopok")) stopNote("STOP NOT CONFIRMED: use the remote", "stopfail"); }, 300);
    for (let i = 0; i < 5 && seq === stopSeq; i++) {
      const ac = new AbortController(); const t = setTimeout(() => ac.abort(), 300);
      try {
        const r = await fetch("/api/stop", { method: "POST", signal: ac.signal, headers: token ? { "Authorization": "Bearer " + token, "X-Client": clientId } : {} });
        const j = await r.json();
        const confirmed = backend === "sim" ? r.ok : (r.ok && j && j.confirmed);
        if (confirmed) { clearTimeout(warn); stopNote("STOP confirmed. Robot is stopped and latched.", "stopok"); return; }
      } catch (e) { /* retry */ } finally { clearTimeout(t); }
    }
    clearTimeout(warn);
    if (seq === stopSeq) stopNote("STOP NOT CONFIRMED: use the remote", "stopfail");
  }
  $("stop").onclick = stop;

  // ------------------------------------------------------------ tabs
  let view = "drive";
  for (const b of document.querySelectorAll("#tabs .tab")) {
    b.onclick = () => {
      view = b.dataset.view;
      for (const o of document.querySelectorAll("#tabs .tab")) o.classList.toggle("on", o === b);
      for (const v of ["drive", "sensors", "safety"]) $("view-" + v).hidden = v !== view;
      if (view === "sensors") pullSensors();
      if (view === "safety") renderTests();
    };
  }

  if (vm) document.querySelector('#tabs [data-view="' + vm[1] + '"]').click();

  // ------------------------------------------------------------ status normalisation (both backends -> one shape)
  let S = null, lastOk = 0, needNeutral = false, loopDown = false;
  let simLimits = null;
  function normRobot(s) {
    const sup = s.supervisor || {};
    return { armed: sup.state === "armed", held: !!sup.held, latched: !!sup.stop_latched, supState: sup.state, controller: s.controller, controllers: s.controllers || [],
      preflight: s.preflight || [], green: !!s.preflight_green, tilt: s.tilt_deg, loopHz: s.loop_hz, loopAge: s.loop_age_ms, cmd: s.cmd || [0, 0, 0],
      limits: s.limits || { vx_min: 0, vx_max: 0, vy_max: 0, wz_max: 0, deadman_timeout_s: null }, obstacle: s.obstacle, cameras: s.cameras || [], events: s.events || [],
      fault: s.fault, imu: s.imu, nav: s.nav, hbHz: sup.heartbeat_hz, stateAge: s.state_age_ms, pose: null, faults: {}, raw: s };
  }
  function normSim(s) {
    const L = s.limits || [0.4, 0.2, 0.5];
    return { armed: !!s.armed, held: !!s.held, latched: !!s.stop_latched, supState: s.supervisor, controller: "GR00T SONIC (stand-in)", controllers: [],
      preflight: [], green: true, tilt: s.truth ? s.truth.tilt_deg : null, loopHz: 50, loopAge: 0, cmd: s.cmd || [0, 0, 0],
      limits: { vx_min: -L[0] / 2, vx_max: L[0], vy_max: L[1], wz_max: L[2], deadman_timeout_s: null }, obstacle: s.teleop_guard ? { warning: s.teleop_guard } : null, cameras: ["d435_rgb"],
      events: [], fault: s.fault, imu: null, nav: s.nav, hbHz: null, stateAge: 0, pose: s.pose, faults: s.faults || {}, simState: s, raw: s };
  }
  const env = () => (S && S.limits) || { vx_min: 0, vx_max: 0, vy_max: 0, wz_max: 0 };

  // ------------------------------------------------------------ inputs: joystick, keys, hold button (motion only while held)
  const keys = new Set();
  let pad = null, holdBtn = false, testWalk = null;
  const scale = () => clamp(Number($("scale").value) / 100, 0, 1);
  $("scale").oninput = () => { $("scaleval").textContent = $("scale").value + "%"; };
  document.addEventListener("keydown", (e) => {
    const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName);
    if (e.key === "Escape" || (e.code === "Space" && !typing)) { e.preventDefault(); stop(); return; }
    if (!typing) keys.add(e.key.toLowerCase());
  });
  document.addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
  const dropInputs = () => { keys.clear(); pad = null; holdBtn = false; $("holdbtn").classList.remove("on"); };
  window.addEventListener("blur", dropInputs);
  document.addEventListener("visibilitychange", dropInputs);

  function driveCmd() {
    const e = env(), s = scale();
    let fx = 0, fy = 0, fw = 0, src = "none";
    if (pad) { fx = pad.y; fw = -pad.x; src = "pad"; }
    else if (holdBtn) { fx = 1; src = "button"; }
    else if (["w", "s", "a", "d", "q", "e"].some((k) => keys.has(k))) {
      fx = (keys.has("w") ? 1 : 0) - (keys.has("s") ? 1 : 0); fw = (keys.has("a") ? 1 : 0) - (keys.has("d") ? 1 : 0); fy = (keys.has("q") ? 1 : 0) - (keys.has("e") ? 1 : 0); src = "keys";
    }
    const held = src !== "none";
    return { mode: "teleop", held: held, src: src, vx: held ? clamp((fx >= 0 ? fx * e.vx_max : -fx * e.vx_min) * s, e.vx_min, e.vx_max) : 0,
      vy: held ? clamp(fy * e.vy_max * s, -e.vy_max, e.vy_max) : 0, wz: held ? clamp(fw * e.wz_max * s, -e.wz_max, e.wz_max) : 0 };
  }

  const padEl = $("pad"), pctx = padEl.getContext("2d");
  function padPos(ev) {
    const r = padEl.getBoundingClientRect();
    let x = ((ev.clientX - r.left) / r.width) * 2 - 1, y = -(((ev.clientY - r.top) / r.height) * 2 - 1);
    const n = Math.hypot(x, y); if (n > 1) { x /= n; y /= n; }
    return { x: x, y: y };
  }
  padEl.addEventListener("pointerdown", (ev) => { if (!S || !S.armed) return; padEl.setPointerCapture(ev.pointerId); pad = padPos(ev); });
  padEl.addEventListener("pointermove", (ev) => { if (pad) pad = padPos(ev); });
  for (const t of ["pointerup", "pointercancel", "lostpointercapture"]) padEl.addEventListener(t, () => { pad = null; });
  function drawPad() {
    const w = padEl.width, h = padEl.height;
    pctx.clearRect(0, 0, w, h);
    pctx.strokeStyle = "rgba(255,255,255,.18)"; pctx.lineWidth = 1;
    pctx.beginPath(); pctx.moveTo(w / 2, 8); pctx.lineTo(w / 2, h - 8); pctx.moveTo(8, h / 2); pctx.lineTo(w - 8, h / 2); pctx.stroke();
    pctx.beginPath(); pctx.arc(w / 2, h / 2, w / 2 - 6, 0, 6.2832); pctx.stroke();
    const p = pad || { x: 0, y: 0 };
    pctx.fillStyle = pad ? "#38bdf8" : (S && S.armed ? "rgba(255,255,255,.7)" : "rgba(255,255,255,.25)");
    pctx.beginPath(); pctx.arc(w / 2 + p.x * w / 2 * 0.8, h / 2 - p.y * h / 2 * 0.8, 24, 0, 6.2832); pctx.fill();
  }
  const hb = $("holdbtn");
  hb.addEventListener("pointerdown", (ev) => { if (hb.disabled) return; hb.setPointerCapture(ev.pointerId); holdBtn = true; hb.classList.add("on"); });
  for (const t of ["pointerup", "pointercancel", "lostpointercapture"]) hb.addEventListener(t, () => { holdBtn = false; hb.classList.remove("on"); });

  // ------------------------------------------------------------ heartbeat (the deadman): 10 Hz while this tab is visible
  let hbBusy = false;
  async function heartbeatTick() {
    drawPad();
    if (!token || !backend || hbBusy || document.visibilityState !== "visible") return;
    const body = testWalk ? { mode: "teleop", held: true, src: "test", vx: testWalk.vx, vy: 0, wz: 0 } : driveCmd();
    hbBusy = true;
    try {
      const r = await api("/api/heartbeat", body, 250);
      onHeartbeat(r);
    } finally { hbBusy = false; }
  }
  function onHeartbeat(r) {
    if (backend === "robot") {
      if (r.status === 503) loopDown = true;
      else if (r.ok) { loopDown = false; needNeutral = !!(r.data && r.data.need_neutral); }
      if (!r.ok && r.data && r.data.error && !$("armmsg").dataset.keep) $("armmsg").textContent = r.data.error;
    } else if (r.ok && r.data) {
      S = normSim(r.data); lastOk = Date.now(); render();
      needNeutral = false;
    }
    sample();
  }
  setInterval(heartbeatTick, 100);

  // ------------------------------------------------------------ arming: slide to arm (the typed phrase of the robot server is sent for you)
  const slide = $("slide"), knob = $("slideKnob"), fill = $("slideFill");
  let drag = null;
  const travel = () => slide.clientWidth - knob.clientWidth - 8;
  function setSlide(f) { const x = f * travel(); knob.style.transform = "translateX(" + x + "px)"; fill.style.width = (x + 56) + "px"; slide.setAttribute("aria-valuenow", String(Math.round(f * 100))); }
  knob.addEventListener("pointerdown", (ev) => { if (slide.classList.contains("off")) return; knob.setPointerCapture(ev.pointerId); drag = { x0: ev.clientX, f0: 0 }; });
  knob.addEventListener("pointermove", (ev) => { if (drag) setSlide(clamp((ev.clientX - drag.x0) / travel(), 0, 1)); });
  const slideEnd = async () => {
    if (!drag) return; drag = null;
    const f = Number(slide.getAttribute("aria-valuenow")) / 100;
    setSlide(0);
    if (f >= 0.95) await toggleArm();
  };
  for (const t of ["pointerup", "pointercancel"]) knob.addEventListener(t, slideEnd);
  slide.addEventListener("keydown", (e) => { if (e.key === "Enter" && !slide.classList.contains("off")) toggleArm(); });
  function say(text, cls) { const m = $("armmsg"); m.textContent = text; m.className = "msg" + (cls ? " " + cls : ""); }
  async function toggleArm() {
    if (S && S.armed) { await api(backend === "sim" ? "/api/disarm" : "/api/disarm", {}); say("Disarmed."); return; }
    if (backend === "sim") { const r = await api("/api/arm", {}); say(r.ok && r.data && r.data.ok !== false ? "Armed. Release every input, then hold to walk." : "Arm refused: " + ((r.data && (r.data.error || r.data.reason)) || "no answer")); return; }
    const q = await api("/api/arm/request", {});
    if (!q.ok) { say("Arm refused: " + ((q.data && q.data.error) || "no answer")); return; }
    const c = await api("/api/arm/confirm", { text: q.data.type_this });
    say(c.data && c.data.ok ? "Armed. Hold the physical stop. Release every input before the first move." : "Arm refused: " + ((c.data && (c.data.reason || c.data.error)) || "no answer"));
  }
  $("resetbtn").onclick = async () => { await api(backend === "sim" ? "/api/reset" : "/api/reset_stop", {}, 35000); stopSeq++; sm.hidden = true; };
  $("dampbtn").onclick = async () => {
    if (backend === "sim") { say("Not available in the simulator."); return; }
    if (!window.confirm("The robot goes limp and falls unless it is supported (gantry or rope). Send?")) return;
    const r = await api("/api/damp", {}); say(r.data ? String(r.data.result || r.data.error || "sent") : "no answer");
  };

  // ------------------------------------------------------------ rendering
  const BRAIN = [["groot", "GR00T SONIC", "NVIDIA whole-body walking policy (TensorRT on the robot)."], ["stand", "Stand only", "Keeps the robot standing. It will not walk."], ["vendor", "Unitree built-in", "The vendor's own walking controller."]];
  const brainInfo = (n) => BRAIN.find((b) => n.toLowerCase().includes(b[0])) || [n, n, ""];
  function stateWords() {
    if (!S) return ["Connecting", "info", "Waiting for the robot."];
    if (backend === "robot" && (loopDown || S.loopHz < 45 || (S.loopAge !== null && S.loopAge > 200))) return ["No control loop", "bad", "The control program is not running. The robot cannot be commanded."];
    if (S.fault) return ["Fault", "bad", "Fault: " + S.fault + ". The robot stopped itself."];
    if (S.latched) return ["Stopped", "bad", "STOP is latched. Press Reset STOP, then arm again."];
    if (S.armed) return S.held ? ["Walking", "ok", "Moving while you hold. Let go and it stands."] : ["Armed", "warn", needNeutral ? "Release every input once, then hold to walk." : "Ready. Hold the walk button or drag the joystick."];
    return ["Disarmed", "warn", S.green ? "The robot cannot move. Slide to arm." : "The robot cannot move. Checks are not all green: arming is refused."];
  }
  function render() {
    if (!S) return;
    const [word, tone, help] = stateWords();
    $("bigState").textContent = word; $("stateHelp").textContent = help;
    $("stateTxt").textContent = word; $("chipState").querySelector(".dot").className = "dot " + tone;
    $("targetTxt").textContent = backend === "sim" ? "Connected to simulator, not the robot" : "Connected to the robot";
    $("chipTarget").querySelector(".dot").className = "dot " + (backend === "sim" ? "info" : "ok");
    $("slideLabel").textContent = S.armed ? "Slide to disarm" : "Slide to arm";
    slide.classList.toggle("off", !S.armed && !S.green);
    $("resetbtn").hidden = !S.latched;
    $("dampbtn").hidden = backend === "sim";
    const canWalk = S.armed && !S.latched && !needNeutral;
    hb.disabled = !canWalk;
    $("holdHelp").textContent = canWalk ? "Hold the button, drag the joystick, or hold W." : (S.armed ? "Release every input once first." : "Arm first. Keys: W S forward and back, A D turn, Q E sideways.");
    // brain
    const box = $("ctlChoices");
    const names = S.controllers.length ? S.controllers : [S.controller];
    if (box.dataset.sig !== names.join("|") + S.controller + S.armed) {
      box.dataset.sig = names.join("|") + S.controller + S.armed; box.innerHTML = "";
      for (const n of names) {
        const info = brainInfo(n), b = el("button", "choice" + (n === S.controller ? " on" : "")); b.appendChild(el("b", "", info[1])); b.appendChild(el("span", "help", info[2]));
        b.disabled = S.armed || backend === "sim";
        b.onclick = async () => { const r = await api("/api/controller", { name: n }); if (!r.ok) say((r.data && (r.data.reason || r.data.error)) || "refused"); };
        box.appendChild(b);
      }
    }
    $("ctlHelp").textContent = S.armed ? "Disarm to change the walking brain." : (backend === "sim" ? "The simulator runs a stand-in for the walking policy." : "");
    // checks
    const pf = $("preflight"); pf.innerHTML = "";
    for (const c of S.preflight) {
      const li = el("li"); li.appendChild(el("i", "dot " + (c.status === "ok" ? "ok" : c.status === "fail" ? "bad" : "warn")));
      li.appendChild(el("span", "nm", c.name)); li.appendChild(el("span", "dt", c.detail + (c.required ? "" : " (advisory)"))); pf.appendChild(li);
    }
    $("pfsum").textContent = backend === "sim" ? "simulator" : (S.green ? "all green" : "not green"); $("pfsum").className = "pfsum " + (S.green ? "ok" : "fail");
    // camera + obstacle
    const o = S.obstacle || null;
    let oTone = "warn", oTxt = "No obstacle sensor";
    if (o && o.distance !== null && o.distance !== undefined) { oTxt = "Nearest obstacle " + fmt(o.distance, 1) + " m"; oTone = o.reason === "blocked" ? "bad" : o.reason === "capped" ? "warn" : "ok"; }
    else if (o && o.warning) oTxt = o.warning.length > 60 ? "No obstacle sensor" : o.warning;
    $("obsTxt").textContent = oTxt; $("obsChip").querySelector(".dot").className = "dot " + oTone;
    const e = env(), scl = o && o.scale !== undefined && o.scale !== null ? o.scale : 1;
    $("limitFill").style.width = Math.round(clamp(scl, 0, 1) * 100) + "%"; $("limitFill").style.background = scl < 0.4 ? "var(--bad)" : scl < 1 ? "var(--warn)" : "var(--ok)";
    $("limitTxt").textContent = fmt(e.vx_max * scl, 2) + " m/s";
    // S1 card
    $("envF").textContent = fmt(e.vx_max, 2) + " m/s"; $("envW").textContent = fmt(e.wz_max, 2) + " rad/s";
    $("envD").textContent = e.deadman_timeout_s ? fmt(e.deadman_timeout_s, 1) + " s" : "-";
    $("loopTxt").textContent = backend === "sim" ? "simulated" : fmt(S.loopHz, 0) + " Hz";
    // sensors tab cards
    renderSensors();
    renderSim();
    const ev = $("events"); ev.innerHTML = "";
    const tNow = S.events.length ? S.events[S.events.length - 1].t : 0;
    for (const x of S.events.slice().reverse()) {
      const ago = fmt(tNow - x.t, 0) + " s before latest  ";
      const t = x.event === "transition" ? `${ago}${x.from} to ${x.to} (${x.reason})` : `${ago}${x.event}`;
      ev.appendChild(el("li", "", t));
    }
    const img = $("camimg");
    if (S.cameras.length && img.dataset.src !== "1") { img.dataset.src = "1"; }
  }
  function kv(parent, k, v, cls) { const r = el("div", "kv"); r.appendChild(el("span", "", k)); r.appendChild(el("b", cls || "", v)); parent.appendChild(r); }
  function renderSensors() {
    if (!S) return;
    const od = $("odom"); od.innerHTML = "";
    const p = S.pose;
    kv(od, "Position x, y", p ? fmt(p.x) + " m, " + fmt(p.y) + " m" : "no pose source");
    kv(od, "Heading", p ? fmt(p.yaw * 57.2958, 0) + "°" : "-");
    kv(od, "Speed", fmt(Math.hypot(S.cmd[0] || 0, S.cmd[1] || 0), 2) + " m/s");
    kv(od, "Pose source", p ? (p.healthy ? p.source : p.source + " (lost)") : "none", p && !p.healthy ? "bad" : "");
    const bd = $("body"); bd.innerHTML = "";
    kv(bd, "Tilt", S.tilt === null || S.tilt === undefined ? "-" : fmt(S.tilt, 1) + "°");
    const r = S.raw || {}; const tmp = r.meta && r.meta.motor_temp_max_c;
    kv(bd, "Hottest joint", tmp ? fmt(tmp, 0) + " °C" : "-");
    kv(bd, "Battery", "unknown (not in the robot's state message)");
    const al = $("alive"); al.innerHTML = "";
    const row = (name, ok, txt) => { const r2 = el("div", "kv"); const s = el("span"); s.appendChild(el("i", "dot " + (ok ? "ok" : "bad"))); s.appendChild(document.createTextNode(" " + name)); r2.appendChild(s); r2.appendChild(el("b", "", txt)); al.appendChild(r2); };
    row("Robot state", S.stateAge !== null && S.stateAge < 300, S.stateAge === null ? "none" : fmt(S.stateAge, 0) + " ms old");
    row("Depth camera", S.cameras.length > 0, S.cameras.length ? "streaming" : "none");
    row("Obstacle sensor", !!(S.obstacle && S.obstacle.distance !== null && S.obstacle.distance !== undefined), S.obstacle && S.obstacle.distance != null ? "ok" : "none");
    row("Rear sensor", false, "none");
    const o = S.obstacle;
    $("obsBig").textContent = o && o.distance != null ? fmt(o.distance, 1) + " m" : "no reading";
    $("obsHelp").textContent = o && o.distance != null ? (o.reason === "blocked" ? "Blocked: the robot will not move forward." : o.reason === "capped" ? "Slowed to " + fmt((env().vx_max || 0) * (o.scale || 0), 2) + " m/s." : "Clear. Full speed allowed.") : (o && o.warning) || "";
  }

  // ------------------------------------------------------------ camera
  setInterval(async () => {
    if (!S || !token || view !== "drive" || !S.cameras.length) return;
    const im = $("camimg");
    if (backend === "robot") { im.src = `/api/camera/${encodeURIComponent(S.cameras[0])}.png?token=${encodeURIComponent(token)}&t=${Date.now()}`; im.hidden = false; $("nocam").hidden = true; }
    else {
      try {
        const r = await fetch("/api/img/d435_rgb.jpg", { headers: { "Authorization": "Bearer " + token }, cache: "no-store" });
        if (r.ok) { const u = URL.createObjectURL(await r.blob()), old = im.src; im.src = u; if (old.startsWith("blob:")) URL.revokeObjectURL(old); im.hidden = false; $("nocam").hidden = true; }
      } catch (e) { /* no image yet */ }
    }
  }, 500);

  // ------------------------------------------------------------ natural-language box (the stack's command gateway; simulator mounts it under /gw)
  function bubble(text, cls) { const b = el("div", "bubble " + (cls || ""), text); $("sayLog").appendChild(b); $("sayLog").scrollTop = 1e6; return b; }
  async function sendSentence() {
    const t = $("sayin").value.trim(); if (!t) return;
    bubble(t, "me"); $("sayin").value = "";
    if (backend !== "sim") { bubble("The robot's command gateway (port 8090) is a separate service and is not connected to this page yet.", "err"); return; }
    const r = await api("/gw/api/command", { text: t }, 8000);
    const j = r.data;
    if (!r.ok || !j) { bubble("Could not understand: " + ((j && j.detail) || "no answer"), "err"); return; }
    const b = bubble((j.summary || j.reply || j.text || JSON.stringify(j)).toString(), "");
    if (j.id) {
      const row = el("div", "row"), ok = el("button", "btn primary", "Confirm"), no = el("button", "btn", "Cancel");
      ok.onclick = async () => { row.remove(); const c = await api("/gw/api/confirm", { id: j.id }); const s = c.data && c.data.status; bubble(s ? "Started: " + s.state + ". Hold the walk button to move." : "Could not start: " + ((c.data && c.data.detail) || "no answer"), s ? "" : "err"); };
      no.onclick = async () => { row.remove(); await api("/gw/api/cancel", {}); bubble("Cancelled."); };
      row.appendChild(ok); row.appendChild(no); b.appendChild(row);
    }
  }
  $("saybtn").onclick = sendSentence;
  $("sayin").addEventListener("keydown", (e) => { if (e.key === "Enter") sendSentence(); });

  // ------------------------------------------------------------ 3D map (own projection on a canvas, no WebGL needed)
  const cv = $("map3d"), cx = cv.getContext("2d");
  const cam = { yaw: -0.6, pitch: 0.9, zoom: 55 }; // zoom = pixels per metre
  const layers = { cloud: true, trail: true, obstacle: true, plan: true };
  let CL = null, PLAN = null, MAPC = null, trailLocal = [];
  for (const b of document.querySelectorAll("#layers .pill")) b.onclick = () => { layers[b.dataset.layer] = !layers[b.dataset.layer]; b.classList.toggle("on", layers[b.dataset.layer]); };
  let md = null;
  cv.addEventListener("pointerdown", (e) => { cv.setPointerCapture(e.pointerId); md = { x: e.clientX, y: e.clientY }; });
  cv.addEventListener("pointermove", (e) => { if (!md) return; cam.yaw += (e.clientX - md.x) * 0.008; cam.pitch = clamp(cam.pitch + (e.clientY - md.y) * 0.008, 0.15, 1.5); md = { x: e.clientX, y: e.clientY }; });
  for (const t of ["pointerup", "pointercancel"]) cv.addEventListener(t, () => { md = null; });
  cv.addEventListener("wheel", (e) => { e.preventDefault(); cam.zoom = clamp(cam.zoom * (e.deltaY < 0 ? 1.1 : 0.9), 15, 250); }, { passive: false });
  const BAND_Z = [0.1, 0.5, 0.9, 1.3, 1.7]; // metres per height band (display only)
  function proj(x, y, z, c0) {
    const dx = x - c0[0], dy = y - c0[1];
    const cy_ = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const rx = dx * cy_ - dy * sy, ry = dx * sy + dy * cy_;
    return [cv.width / 2 + rx * cam.zoom, cv.height * 0.58 - (ry * sp + z * cp) * cam.zoom];
  }
  function drawMap() {
    const w = cv.width, h = cv.height; cx.clearRect(0, 0, w, h);
    const pose = S && S.pose ? S.pose : { x: 0, y: 0, yaw: 0 };
    const c0 = [pose.x, pose.y];
    cx.strokeStyle = "rgba(255,255,255,.12)"; cx.lineWidth = 1;
    const gx = Math.round(pose.x), gy = Math.round(pose.y);
    for (let i = -10; i <= 10; i++) {
      let a = proj(gx + i, gy - 10, 0, c0), b = proj(gx + i, gy + 10, 0, c0); cx.beginPath(); cx.moveTo(a[0], a[1]); cx.lineTo(b[0], b[1]); cx.stroke();
      a = proj(gx - 10, gy + i, 0, c0); b = proj(gx + 10, gy + i, 0, c0); cx.beginPath(); cx.moveTo(a[0], a[1]); cx.lineTo(b[0], b[1]); cx.stroke();
    }
    if (layers.plan && PLAN) {
      cx.strokeStyle = "rgba(255,255,255,.3)";
      for (const r of PLAN.rects || []) {
        if (r.length < 4) continue;
        const [x0, y0, x1, y1] = r; const pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]].map((q) => proj(q[0], q[1], 0, c0));
        cx.beginPath(); pts.forEach((q, i) => (i ? cx.lineTo(q[0], q[1]) : cx.moveTo(q[0], q[1]))); cx.stroke();
      }
    }
    if (layers.cloud) {
      cx.fillStyle = "rgba(103,232,249,.85)";
      const draw = (arr, a) => { for (let i = 0; i + 2 < arr.length; i += 3) { const q = proj(arr[i] / 100, arr[i + 1] / 100, BAND_Z[clamp(arr[i + 2], 0, 4)], c0); cx.globalAlpha = a; cx.fillRect(q[0] - 1.5, q[1] - 1.5, 3, 3); } cx.globalAlpha = 1; };
      if (MAPC) draw(MAPC, 0.35);
      if (CL && CL.scan) draw(CL.scan, 0.9);
    }
    const trail = CL && CL.trail && CL.trail.length ? CL.trail : trailLocal.flat();
    if (layers.trail && trail.length > 3) {
      cx.strokeStyle = "#a78bfa"; cx.lineWidth = 3; cx.beginPath();
      for (let i = 0; i + 1 < trail.length; i += 2) { const q = proj(trail[i], trail[i + 1], 0.02, c0); if (i) cx.lineTo(q[0], q[1]); else cx.moveTo(q[0], q[1]); }
      cx.stroke();
    }
    const o = S && S.obstacle;
    if (layers.obstacle && o && o.distance != null) {
      const d = o.distance, wd = 0.45, c = Math.cos(pose.yaw), s = Math.sin(pose.yaw);
      const pts = [[0.2, -wd], [d, -wd], [d, wd], [0.2, wd]].map((q) => proj(pose.x + c * q[0] - s * q[1], pose.y + s * q[0] + c * q[1], 0.02, c0));
      cx.fillStyle = "rgba(251,191,36,.22)"; cx.strokeStyle = "#fbbf24"; cx.lineWidth = 2; cx.setLineDash([8, 6]);
      cx.beginPath(); pts.forEach((q, i) => (i ? cx.lineTo(q[0], q[1]) : cx.moveTo(q[0], q[1]))); cx.closePath(); cx.fill(); cx.stroke(); cx.setLineDash([]);
    }
    const rp = proj(pose.x, pose.y, 0.05, c0), tip = proj(pose.x + Math.cos(pose.yaw) * 0.5, pose.y + Math.sin(pose.yaw) * 0.5, 0.05, c0);
    cx.fillStyle = "#38bdf8"; cx.shadowColor = "#38bdf8"; cx.shadowBlur = 24; cx.beginPath(); cx.arc(rp[0], rp[1], 12, 0, 6.2832); cx.fill(); cx.shadowBlur = 0;
    cx.strokeStyle = "#fff"; cx.lineWidth = 3; cx.beginPath(); cx.moveTo(rp[0], rp[1]); cx.lineTo(tip[0], tip[1]); cx.stroke();
  }
  setInterval(() => { if (view === "sensors") drawMap(); }, 100);
  let lastPose = null;
  function sample() { // local path for the robot server (it has no map feed): the pose is only known in the simulator
    if (S && S.pose && (!lastPose || Math.hypot(S.pose.x - lastPose[0], S.pose.y - lastPose[1]) > 0.05)) { lastPose = [S.pose.x, S.pose.y]; trailLocal.push(lastPose); trailLocal = trailLocal.slice(-600); }
  }
  async function pullSensors() {
    if (!token || !backend) return;
    if (backend === "sim") {
      if (!PLAN) { const p = await api("/api/plan"); if (p.ok) PLAN = p.data; }
      const c = await api("/api/cloud", undefined, 1500); if (c.ok) CL = c.data;
      const mp = await api("/api/map", undefined, 2500); if (mp.ok && mp.data) MAPC = mp.data.cells;
    }
    $("mapNote").textContent = backend === "sim" ? "Lidar points and floor plan come from the simulator." : "The robot server does not publish lidar or odometry yet. The robot is shown at the origin; this view fills in when that feed exists.";
  }
  setInterval(() => { if (view === "sensors") pullSensors(); }, 600);

  // ------------------------------------------------------------ safety tests (simulator only): break one thing, measure the time to a stop
  const TESTS = [
    { id: "camera", name: "Camera freezes", desc: "Depth frames stop. Must stop within 0.5 s.", limit: 0.5, kind: "fault", fault: "stale_frames" },
    { id: "depth", name: "Camera goes blank", desc: "Depth image drops out. Must stop within 0.5 s.", limit: 0.5, kind: "fault", fault: "depth_dropout" },
    { id: "pose", name: "Position lost", desc: "The localizer loses the robot. Must stop within 0.5 s.", limit: 0.5, kind: "fault", fault: "localizer_lost" },
    { id: "state", name: "Robot state lost", desc: "The robot stops reporting its state. Must stop within 0.5 s.", limit: 0.5, kind: "fault", fault: "lowstate_lost" },
    { id: "stopbtn", name: "STOP button", desc: "Press STOP mid-walk and measure the delay. Must stop within 0.3 s.", limit: 0.3, kind: "stop" },
  ];
  const results = {};
  let runningTest = null, lastCmd = [0, 0, 0];
  function renderSim() {
    if (!S) return;
    lastCmd = S.cmd;
    const box = $("simNow"); box.innerHTML = "";
    if (backend !== "sim") { box.appendChild(el("p", "help", "Not connected to a simulator.")); return; }
    const s = S.simState || {};
    kv(box, "Simulated time", fmt(s.t, 1) + " s"); kv(box, "Speed vs real time", fmt(s.rtf, 2) + "x");
    kv(box, "Robot fell", s.truth && s.truth.fallen ? "yes" : "no", s.truth && s.truth.fallen ? "bad" : "");
    kv(box, "Active faults", Object.keys(S.faults).filter((k) => S.faults[k]).join(", ") || "none");
  }
  function renderTests() {
    const list = $("testList"); list.innerHTML = "";
    const sim = backend === "sim";
    const canRun = sim && S && S.armed && !S.latched && !runningTest;
    const head = el("div", "row gap");
    const all = el("button", "btn primary", "Run all"); all.disabled = !canRun; all.onclick = runAll; head.appendChild(all);
    list.appendChild(head); list.lastChild.style.marginBottom = "12px";
    for (const t of TESTS) {
      const r = results[t.id], row = el("div", "test"), txt = el("div", "grow");
      txt.appendChild(el("div", "nm", t.name)); txt.appendChild(el("div", "ds", t.desc)); row.appendChild(txt);
      let chip = el("span", "res", "Not run yet");
      if (runningTest === t.id) chip = el("span", "res", "Running...");
      else if (r) { chip = el("span", "res " + (r.pass ? "pass" : "fail"), (r.pass ? "Passed" : "Failed") + " · " + r.text); }
      row.appendChild(chip);
      const b = el("button", "btn", "Run"); b.disabled = !canRun; b.onclick = () => runTest(t); row.appendChild(b);
      list.appendChild(row);
    }
    $("testNote").textContent = !sim ? "Tests run only in the simulator. This page is connected to the real robot, and nothing here injects faults into it."
      : (S && S.armed ? "Each test walks the simulated robot at 0.2 m/s, breaks one thing, and times the stop. Timing is accurate to about 0.1 s (heartbeat rate)." : "Arm the simulated robot first (Drive tab).");
  }
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const moving = () => Math.hypot(lastCmd[0] || 0, lastCmd[1] || 0) > 0.05;
  async function runTest(t) {
    if (runningTest || backend !== "sim") return;
    runningTest = t.id; delete results[t.id]; renderTests();
    try {
      testWalk = { vx: Math.min(0.2, env().vx_max || 0.2) };
      let w = 0; while (!moving() && w < 40) { await sleep(100); w++; } // up to 4 s to get walking
      if (!moving()) { results[t.id] = { pass: false, text: "the robot did not start walking" }; return; }
      await sleep(500);
      const t0 = performance.now();
      if (t.kind === "fault") await api("/api/fault", { name: t.fault, on: true });
      else if (t.kind === "stop") { await fetch("/api/stop", { method: "POST" }); testWalk = null; }
      let dt = null;
      while (performance.now() - t0 < 4000) {
        await sleep(50);
        if (!moving()) { dt = (performance.now() - t0) / 1000; break; }
      }
      if (dt === null) results[t.id] = { pass: false, text: "still moving after 4 s" };
      else results[t.id] = { pass: dt <= t.limit, text: fmt(dt, 2) + " s (limit " + fmt(t.limit, 1) + " s)" };
    } finally {
      testWalk = null;
      if (t.kind === "fault") await api("/api/fault", { name: t.fault, on: false });
      if (t.kind === "stop") await api("/api/reset", {}, 35000);
      runningTest = null; renderTests();
    }
  }
  async function runAll() { for (const t of TESTS) { if (!(S && S.armed && !S.latched)) break; await runTest(t); await sleep(500); } }
  setInterval(() => { if (view === "safety") renderTests(); }, 500);

  // ------------------------------------------------------------ status polling (robot server) and logs
  async function pollStatus() {
    if (!token || !backend) return;
    if (backend === "robot") {
      const r = await api("/api/status");
      if (r.ok && r.data) { S = normRobot(r.data); lastOk = Date.now(); sample(); render(); }
    }
    const lost = Date.now() - lastOk > 1500;
    if (lost && backend) { $("stateTxt").textContent = "No link"; $("chipState").querySelector(".dot").className = "dot bad"; $("bigState").textContent = "No link"; $("stateHelp").textContent = "The page lost the robot. It stops by itself without heartbeats."; }
  }
  setInterval(pollStatus, 250);
  async function pollLogs() {
    if (!token || backend !== "robot") return;
    const r = await api("/api/logs"); if (!r.ok || !r.data) return;
    const ul = $("logs"); ul.innerHTML = "";
    for (const f of r.data) {
      const li = el("li"), a = el("a", "", `${f.name} (${(f.bytes / 1024).toFixed(0)} kB)`);
      a.href = `/api/logs/${encodeURIComponent(f.name)}?token=${encodeURIComponent(token)}`; a.download = f.name; li.appendChild(a); ul.appendChild(li);
    }
  }
  setInterval(pollLogs, 5000);

  (async () => { backend = await detect(); setSlide(0); $("targetTxt").textContent = backend === "sim" ? "Connecting to simulator" : "Connecting to the robot"; pollLogs(); })();
})();
