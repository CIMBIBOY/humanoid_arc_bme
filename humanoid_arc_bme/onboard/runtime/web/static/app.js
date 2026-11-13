// G1 operator UI. No external libraries. Talks to g1_runtime/web/server.py by JSON polling.
"use strict";
(function () {
  const $ = (id) => document.getElementById(id);
  const clientId = "ui-" + Math.random().toString(36).slice(2, 10);
  let token = "";
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
    const opt = { method: body === undefined ? "GET" : "POST", headers: { "Authorization": "Bearer " + token, "X-Client": clientId } };
    if (body !== undefined) { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), timeoutMs || 2000);
    opt.signal = ac.signal;
    let r;
    try { r = await fetch(path, opt); } finally { clearTimeout(timer); }
    let j = null;
    try { j = await r.json(); } catch (e) { j = null; }
    if (r.status === 401) $("tokencard").hidden = false;
    return { status: r.status, ok: r.ok, data: j };
  }

  // ------------------------------------------------------------ STOP (always works, also without the token)
  // "STOP CONFIRMED" is shown only after the server answered that the supervisor latched it; no answer within 300 ms
  // (or an unconfirmed answer) shows "STOP NOT CONFIRMED: use the remote". STOP is re-sent until confirmed (max 5 tries).
  let stopSeq = 0;
  async function stop() {
    pad = null; keys.clear(); holdBtn = false;
    const seq = ++stopSeq;
    const msg = $("armmsg");
    msg.textContent = "STOP sending..."; msg.className = "msg stopwait";
    const warn = setTimeout(() => { if (seq === stopSeq && msg.className !== "msg stopok") { msg.textContent = "STOP NOT CONFIRMED: use the remote"; msg.className = "msg stopfail"; } }, 300);
    for (let i = 0; i < 5 && seq === stopSeq; i++) {
      const ac = new AbortController(); const t = setTimeout(() => ac.abort(), 300);
      try {
        const r = await fetch("/api/stop", { method: "POST", signal: ac.signal, headers: token ? { "Authorization": "Bearer " + token, "X-Client": clientId } : {} });
        const j = await r.json();
        if (r.ok && j && j.confirmed) { clearTimeout(warn); msg.textContent = "STOP CONFIRMED (" + j.state + ", latched)"; msg.className = "msg stopok"; return; }
      } catch (e) { /* retry */ } finally { clearTimeout(t); }
    }
    clearTimeout(warn);
    if (seq === stopSeq) { msg.textContent = "STOP NOT CONFIRMED: use the remote"; msg.className = "msg stopfail"; }
  }
  $("stop").onclick = stop;
  document.addEventListener("keydown", (e) => {
    const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName);
    if (e.key === "Escape" || (e.code === "Space" && !typing)) { e.preventDefault(); stop(); return; }
    if (!typing) keys.add(e.key.toLowerCase());
  });
  document.addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
  window.addEventListener("blur", () => { keys.clear(); pad = null; holdBtn = false; });
  document.addEventListener("visibilitychange", () => { keys.clear(); pad = null; holdBtn = false; });

  // ------------------------------------------------------------ inputs
  const keys = new Set();
  let status = null, lastStatusOk = 0, mode = "teleop";
  let pad = null; // [-1..1] x (turn), y (forward) while the pad is pressed (hold to run)
  let holdBtn = false; // HOLD TO RUN button pressed (curvature / waypoints / ros)
  const env = () => (status && status.limits) || { vx_min: 0, vx_max: 0, vy_max: 0, wz_max: 0 };
  const scale = () => Math.min(1, Math.max(0, Number($("scale").value) / 100));
  for (const id of ["scale", "cv", "ck"]) {
    $(id).oninput = () => { $(id + "val").textContent = id === "scale" ? $(id).value + "%" : (id === "ck" ? (Number($(id).value) / 100).toFixed(2) : $(id).value); };
  }
  const clampv = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

  // Velocity is non-zero ONLY while the pad is pressed or a drive key is held (no persistent slider as a motion source).
  function teleopCmd() {
    const e = env(), s = scale();
    let fx = 0, fy = 0, fw = 0, src = "none";
    if (pad) { fx = pad.y; fw = -pad.x; src = "pad"; }
    else if (["w", "s", "a", "d", "q", "e"].some((k) => keys.has(k))) {
      fx = (keys.has("w") ? 1 : 0) - (keys.has("s") ? 1 : 0);
      fw = (keys.has("a") ? 1 : 0) - (keys.has("d") ? 1 : 0);
      fy = (keys.has("q") ? 1 : 0) - (keys.has("e") ? 1 : 0);
      src = "keys";
    }
    const held = src !== "none";
    const vx = held ? clampv((fx >= 0 ? fx * e.vx_max : -fx * e.vx_min) * s, e.vx_min, e.vx_max) : 0;
    return { mode: "teleop", held: held, vx: vx, vy: held ? clampv(fy * e.vy_max * s, -e.vy_max, e.vy_max) : 0,
             wz: held ? clampv(fw * e.wz_max * s, -e.wz_max, e.wz_max) : 0, src: src };
  }

  function curvatureCmd() {
    const e = env(), c = Number($("cv").value) / 100;
    const v = holdBtn ? clampv(c * (c >= 0 ? e.vx_max : -e.vx_min), e.vx_min, e.vx_max) : 0;
    return { mode: "curvature", held: holdBtn, v: v, kappa: Number($("ck").value) / 100 };
  }

  const hb = $("holdbtn");
  hb.addEventListener("pointerdown", (ev) => { hb.setPointerCapture(ev.pointerId); holdBtn = true; hb.classList.add("on"); });
  const holdUp = () => { holdBtn = false; hb.classList.remove("on"); };
  hb.addEventListener("pointerup", holdUp); hb.addEventListener("pointercancel", holdUp); hb.addEventListener("lostpointercapture", holdUp);

  // joystick pad
  const padEl = $("pad"), pctx = padEl.getContext("2d");
  function padPos(ev) {
    const r = padEl.getBoundingClientRect();
    let x = ((ev.clientX - r.left) / r.width) * 2 - 1, y = -(((ev.clientY - r.top) / r.height) * 2 - 1);
    const n = Math.hypot(x, y); if (n > 1) { x /= n; y /= n; }
    return { x: x, y: y };
  }
  padEl.addEventListener("pointerdown", (ev) => { padEl.setPointerCapture(ev.pointerId); pad = padPos(ev); });
  padEl.addEventListener("pointermove", (ev) => { if (pad) pad = padPos(ev); });
  const padUp = () => { pad = null; };
  padEl.addEventListener("pointerup", padUp); padEl.addEventListener("pointercancel", padUp); padEl.addEventListener("lostpointercapture", padUp);
  function drawPad() {
    const w = padEl.width, h = padEl.height;
    pctx.clearRect(0, 0, w, h);
    pctx.strokeStyle = "#2c333b"; pctx.beginPath(); pctx.moveTo(w / 2, 0); pctx.lineTo(w / 2, h); pctx.moveTo(0, h / 2); pctx.lineTo(w, h / 2); pctx.stroke();
    const p = pad || { x: 0, y: 0 };
    pctx.fillStyle = pad ? "#6aa9ff" : "#3a434e";
    pctx.beginPath(); pctx.arc(w / 2 + p.x * w / 2 * 0.9, h / 2 - p.y * h / 2 * 0.9, 18, 0, 2 * Math.PI); pctx.fill();
  }

  // mode tabs
  for (const b of document.querySelectorAll("#modetabs button")) {
    b.onclick = () => {
      mode = b.dataset.mode;
      for (const o of document.querySelectorAll("#modetabs button")) o.classList.toggle("on", o === b);
      for (const p of ["teleop", "curvature", "waypoints", "ros"]) $("pane-" + p).hidden = p !== mode;
      $("holdbtn").hidden = mode === "teleop"; holdUp(); pad = null;
    };
  }

  // ------------------------------------------------------------ heartbeat (the deadman): 10 Hz while this tab is visible
  let hbBusy = false, loopDown = false, needNeutral = false;
  setInterval(async () => {
    drawPad();
    if (!token || hbBusy || document.visibilityState !== "visible") return;
    let body = mode === "teleop" ? teleopCmd() : mode === "curvature" ? curvatureCmd() : { mode: mode, held: holdBtn };
    hbBusy = true;
    try {
      const r = await api("/api/heartbeat", body, 250);
      if (r.status === 503) loopDown = true;
      else if (r.ok) { loopDown = false; needNeutral = !!(r.data && r.data.need_neutral); }
      if (!r.ok && r.data && !String($("armmsg").className).includes("stop")) $("armmsg").textContent = r.data.error || "";
    } catch (e) { /* link indicator handles it */ }
    hbBusy = false;
  }, 100);

  // ------------------------------------------------------------ arm / disarm / damp / controller
  $("armbtn").onclick = async () => {
    const r = await api("/api/arm/request", {});
    if (!r.ok) { $("armmsg").textContent = (r.data && r.data.error) || "arm request refused"; return; }
    $("armphrase").textContent = r.data.type_this; $("armctl").textContent = status ? status.controller : "";
    $("armbox").hidden = false; $("armtext").value = ""; $("armtext").focus();
  };
  $("armcancel").onclick = () => { $("armbox").hidden = true; };
  $("armconfirm").onclick = async () => {
    const r = await api("/api/arm/confirm", { text: $("armtext").value });
    $("armbox").hidden = true;
    $("armmsg").textContent = r.data ? (r.data.ok ? "ARMED" : "refused: " + (r.data.reason || r.data.error)) : "no answer";
  };
  $("armtext").addEventListener("keydown", (e) => { if (e.key === "Enter") $("armconfirm").click(); });
  $("disarmbtn").onclick = () => api("/api/disarm", {});
  $("resetbtn").onclick = () => api("/api/reset_stop", {});
  $("dampbtn").onclick = async () => {
    if (!confirm("Damp: the robot goes limp and falls unless it is supported (gantry / rope). Send?")) return;
    const r = await api("/api/damp", {});
    $("armmsg").textContent = r.data ? (r.data.result || r.data.error) : "";
  };
  $("ctlsel").onchange = async () => {
    const r = await api("/api/controller", { name: $("ctlsel").value });
    if (!r.ok) $("armmsg").textContent = (r.data && (r.data.reason || r.data.error)) || "refused";
  };
  $("relbtn").onclick = async () => {
    const r = await api("/api/release_vendor", { text: $("reltext").value });
    $("armmsg").textContent = r.data ? (r.data.reason || r.data.error) : "";
  };
  $("wpsend").onclick = async () => {
    const wps = $("wps").value.split("\n").map((l) => l.trim().split(/[\s,]+/).map(Number)).filter((a) => a.length >= 2 && a.every(isFinite));
    const r = await api("/api/waypoints", { waypoints: wps });
    $("armmsg").textContent = r.data ? (r.data.ok ? `${wps.length} waypoints sent` : r.data.reason) : "";
  };

  // ------------------------------------------------------------ status
  const fmt = (x, d = 2) => (x === null || x === undefined ? "-" : Number(x).toFixed(d));
  function li(text, cls) { const e = document.createElement("li"); e.textContent = text; if (cls) e.className = cls; return e; }
  function render(s) {
    const sup = s.supervisor, armed = sup.state === "armed";
    const b = $("badge");
    const noLoop = loopDown || s.loop_hz < 45 || (s.loop_age_ms !== null && s.loop_age_ms > 200);
    b.textContent = noLoop ? "NO CONTROL LOOP" : s.fault ? "FAULT: " + s.fault : armed ? (needNeutral ? "ARMED: release input" : sup.held ? "ARMED: MOVING" : "ARMED") : sup.stop_latched ? "STOP LATCHED" : "DISARMED";
    b.className = "badge " + (noLoop || s.fault ? "lost" : armed ? "armed" : sup.stop_latched ? "latched" : "disarmed");
    $("ctl").textContent = s.controller; $("age").textContent = s.state_age_ms === null ? "none" : fmt(s.state_age_ms, 0) + " ms";
    $("loophz").textContent = fmt(s.loop_hz, 0) + " Hz"; $("hbhz").textContent = fmt(sup.heartbeat_hz, 0) + " Hz";
    $("cmdtxt").textContent = s.cmd.map((x) => fmt(x)).join(" ");
    const sel = $("ctlsel");
    if (sel.options.length !== s.controllers.length) {
      sel.innerHTML = "";
      for (const n of s.controllers) { const o = document.createElement("option"); o.value = o.textContent = n; sel.appendChild(o); }
    }
    if (document.activeElement !== sel) sel.value = s.controller;
    sel.disabled = armed; $("armbtn").disabled = armed || sup.stop_latched; $("disarmbtn").disabled = !armed;
    $("resetbtn").disabled = !sup.stop_latched;
    const pf = $("preflight"); pf.innerHTML = "";
    for (const c of s.preflight) {
      const e = li("", c.status); const sp = document.createElement("span"); sp.className = "s"; sp.textContent = c.status.toUpperCase();
      e.appendChild(sp); e.appendChild(document.createTextNode(c.name + ": " + c.detail));
      if (!c.required) { const a = document.createElement("span"); a.className = "adv"; a.textContent = " (advisory)"; e.appendChild(a); }
      pf.appendChild(e);
    }
    $("pfsum").textContent = s.preflight_green ? "GREEN" : "NOT GREEN: arming refused";
    $("pfsum").className = s.preflight_green ? "ok" : "fail";
    const L = s.limits;
    $("envelope").textContent = `vx ${L.vx_min}..${L.vx_max} m/s, vy +-${L.vy_max} m/s, wz +-${L.wz_max} rad/s, ramp ${L.ax}/${L.ay}/${L.awz}, deadman ${L.deadman_timeout_s} s`;
    $("robot").textContent = JSON.stringify({ tilt_deg: s.tilt_deg, imu: s.imu, meta: s.meta, controller: s.controller_status,
      shadow: s.shadow, owner: s.owner, mode: s.mode, run_dir: s.run_dir, battery: "UNKNOWN (not in rt/lowstate)" }, null, 1);
    $("lowcmd").hidden = !(s.lowcmd && s.lowcmd.enabled);
    $("navinfo").textContent = s.nav ? JSON.stringify(s.nav) : "No navigation module loaded (config nav_module): waypoints disabled.";
    $("wpsend").disabled = !s.nav;
    const ev = $("events"); ev.innerHTML = "";
    for (const e of s.events.slice().reverse()) {
      const t = e.event === "transition" ? `${fmt(e.t, 2)} ${e.from} -> ${e.to} (${e.reason}${e.detail ? ": " + e.detail : ""})` : `${fmt(e.t, 2)} ${e.event} ${JSON.stringify(Object.fromEntries(Object.entries(e).filter(([k]) => !["t", "event"].includes(k))))}`;
      ev.appendChild(li(t));
    }
    const cams = $("cams");
    for (const n of s.cameras) {
      if (!document.getElementById("cam-" + n)) { const im = document.createElement("img"); im.id = "cam-" + n; im.alt = n; cams.appendChild(im); }
    }
  }
  async function pollStatus() {
    if (token) {
      try {
        const r = await api("/api/status");
        if (r.ok) { status = r.data; lastStatusOk = Date.now(); render(status); }
      } catch (e) { /* link lost */ }
    }
    const lost = Date.now() - lastStatusOk > 1000;
    $("link").textContent = lost ? "LOST" : "ok";
    if (lost) { $("badge").textContent = "NO LINK"; $("badge").className = "badge lost"; }
  }
  setInterval(pollStatus, 250);

  // ------------------------------------------------------------ plots
  let rows = [], since = 0;
  function plot(cv, series) {
    const ctx = cv.getContext("2d"), w = cv.width, h = cv.height;
    ctx.clearRect(0, 0, w, h);
    if (rows.length < 2) return;
    const t1 = rows[rows.length - 1].wall, t0 = t1 - 60;
    let lo = Infinity, hi = -Infinity;
    for (const s of series) for (const r of rows) { const v = r[s.k]; if (v !== null && v !== undefined) { const y = v * (s.f || 1); lo = Math.min(lo, y); hi = Math.max(hi, y); } }
    if (!isFinite(lo)) return;
    if (hi - lo < 1e-6) { hi += 0.5; lo -= 0.5; }
    const pad = (hi - lo) * 0.1; lo -= pad; hi += pad;
    const X = (t) => ((t - t0) / 60) * w, Y = (v) => h - ((v - lo) / (hi - lo)) * h;
    ctx.strokeStyle = "#2c333b"; ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(w, Y(0)); ctx.stroke();
    ctx.fillStyle = "#9aa3ad"; ctx.font = "11px sans-serif"; ctx.fillText(hi.toFixed(2), 2, 11); ctx.fillText(lo.toFixed(2), 2, h - 3);
    for (const r of rows) if (r.armed) { ctx.fillStyle = "rgba(224,72,62,0.08)"; ctx.fillRect(X(r.wall), 0, w / 600 + 1, h); }
    for (const s of series) {
      ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue(s.c); ctx.beginPath();
      let pen = false;
      for (const r of rows) { const v = r[s.k]; if (v === null || v === undefined) { pen = false; continue; } const x = X(r.wall), y = Y(v * (s.f || 1)); if (pen) ctx.lineTo(x, y); else ctx.moveTo(x, y); pen = true; }
      ctx.stroke();
    }
  }
  async function pollPlots() {
    if (!token) return;
    try {
      const r = await api("/api/plots?since=" + since);
      if (r.ok && r.data.length) { rows = rows.concat(r.data); since = rows[rows.length - 1].wall; const cut = since - 60; rows = rows.filter((x) => x.wall >= cut); }
    } catch (e) { return; }
    plot($("plotv"), [{ k: "vx_target", c: "--c1" }, { k: "vx", c: "--c2" }, { k: "wz", c: "--c3" }, { k: "gyro_z", c: "--c4" }]);
    plot($("plots"), [{ k: "state_age_ms", c: "--c5" }, { k: "tilt_deg", c: "--c6" }, { k: "loop_hz", c: "--c7", f: 0.1 }]);
  }
  setInterval(pollPlots, 500);

  // ------------------------------------------------------------ camera and logs
  setInterval(() => {
    if (!status || !token) return;
    for (const n of status.cameras) { const im = document.getElementById("cam-" + n); if (im) im.src = `/api/camera/${encodeURIComponent(n)}.png?token=${encodeURIComponent(token)}&t=${Date.now()}`; }
  }, 500);
  async function pollLogs() {
    if (!token) return;
    const r = await api("/api/logs"); if (!r.ok) return;
    const ul = $("logs"); ul.innerHTML = "";
    for (const f of r.data) {
      const e = document.createElement("li"), a = document.createElement("a");
      a.href = `/api/logs/${encodeURIComponent(f.name)}?token=${encodeURIComponent(token)}`; a.textContent = `${f.name} (${(f.bytes / 1024).toFixed(0)} kB)`;
      a.download = f.name; e.appendChild(a); ul.appendChild(e);
    }
  }
  setInterval(pollLogs, 5000); setTimeout(pollLogs, 500);
})();
