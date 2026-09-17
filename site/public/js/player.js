const $ = id => document.getElementById(id);
const FPS = 30;
const CART = '/1CELESTE.PNG';
const SKIP = 10 * FPS;
const BITS = ["LEFT", "RIGHT", "UP", "DOWN", "O", "X"];
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
const buttons = m => m === 0 ? "release" : BITS.filter((_, i) => m & (1 << i)).join("+");
const actionName = a => a.wait ? "wait" : buttons(a.buttons);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi));
const time = f => `${Math.floor(f / FPS / 60)}:${String(Math.floor(f / FPS) % 60).padStart(2, "0")}`;
const decisionFrame = d => d.frame_start ?? Math.round(d.from * FPS);

const ctx = $("screen").getContext("2d");
const tick = () => new Promise(r => setTimeout(r, 0));
const paint = () => new Promise(r => requestAnimationFrame(r));

let M, fbPtr, imageData, cartPtr;
let input = new Uint8Array(0), decisions = [], shown = null, cursor = 0;
let playing = false, speed = 1, raf = null, seekSeq = 0;

const boot = createOpen8({ locateFile: f => "/player/" + f, canvas: $("screen") }).then(m => {
  M = m;
  fbPtr = M._malloc(128 * 128 * 4);
  imageData = ctx.createImageData(128, 128);
  cartPtr = M.stringToNewUTF8(CART);
  if (M._shim_init() !== 0 || M._shim_load_cart(cartPtr) !== 0) throw Error("open8 failed to start");
});

const status = d => {
  if (!d.actions.length) return (d.status || "not executed").replaceAll("_", " ");
  if (d.status && !["played", "executed"].includes(d.status)) return d.status.replaceAll("_", " ");
  return cursor < decisionFrame(d) ? "upcoming" : "played";
};

function draw() {
  M._shim_framebuffer(fbPtr);
  imageData.data.set(M.HEAPU8.subarray(fbPtr, fbPtr + 128 * 128 * 4));
  ctx.putImageData(imageData, 0, 0);
}

function renderCursor() {
  $("playhead").style.left = Math.min(100, cursor / (input.length || 1) * 100) + "%";
  $("clock").textContent = time(cursor) + " / " + time(input.length);
  const state = $("state");
  if (state && shown) state.textContent = status(shown);
}

function show(d) {
  shown = d;
  document.querySelectorAll("#list .row").forEach((r, i) => r.classList.toggle("current", decisions[i] === d));
  if (d) d.node.scrollIntoView({ block: "nearest" });
  const p = $("panel");
  if (!d) { p.innerHTML = ""; return; }
  const acts = d.actions.map(a => `<span class="act">${actionName(a)}</span> &times; ${a.frames}f`).join(", ") || "none";
  let html = `<div class="card"><h3>decision ${d.decision} @ ${(decisionFrame(d) / FPS).toFixed(1)}s &middot; think ${Number(d.latency || 0).toFixed(1)}s &middot; <span class="meta" id="state">${esc(status(d))}</span></h3>
    <p class="meta">played: ${acts}</p></div>`;
  if (d.thinking) html += `<div class="card"><h3>CoT</h3><div class="thinking">${esc(d.thinking)}</div></div>`;
  if (d.error) html += `<div class="card"><pre>${esc(d.error)}</pre></div>`;
  if (d.text) html += `<div class="card"><h3>reply</h3><div class="reply">${esc(d.text)}</div></div>`;
  if (d.tool) html += `<div class="card"><h3>tool call</h3><pre>${esc(JSON.stringify(d.tool))}</pre></div>`;
  p.innerHTML = html;
}

function syncPanel() {
  const d = decisions.findLast(x => decisionFrame(x) <= cursor) || null;
  if (d !== shown) show(d);
}

function buildList() {
  const list = $("list");
  list.innerHTML = "";
  for (const d of decisions) {
    const row = document.createElement("div");
    row.className = "row";
    row.textContent = `#${d.decision} @ ${(decisionFrame(d) / FPS).toFixed(1)}s  ${d.actions.map(actionName).join(",") || status(d)}`;
    row.onclick = () => seekTo(decisionFrame(d), true);
    d.node = row;
    list.appendChild(row);
  }
}

function buildTimeline() {
  const tl = $("timeline");
  tl.querySelectorAll(".bar").forEach(b => b.remove());
  for (const d of decisions) for (const a of (d.actions.length ? d.actions : [{ frame_start: decisionFrame(d), frame_end: decisionFrame(d) }])) {
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.left = Math.min(99.6, a.frame_start / input.length * 100) + "%";
    bar.style.width = Math.max((a.frame_end - a.frame_start) / input.length * 100, 0.4) + "%";
    bar.style.background = !d.actions.length ? "#986c3a" : a.buttons ? `hsl(${a.buttons * 137 % 360} 55% 45%)` : "#3a3a3a";
    bar.title = `#${d.decision} @ ${(decisionFrame(d) / FPS).toFixed(1)}s  ${d.actions.length ? actionName(a) + " × " + a.frames + "f" : status(d)}`;
    bar.onclick = e => { e.stopPropagation(); seekTo(decisionFrame(d), true); };
    tl.appendChild(bar);
  }
}

// Replaying is the emulator's only way to move: it fast-forwards from the
// current frame, reloading the cartridge when we go backwards. Any seek
// resumes playback, so the run keeps moving without another click on play.
async function seekTo(frame, autoplay = false) {
  const target = clamp(Math.round(frame), 0, input.length);
  const seq = ++seekSeq;
  if (playing) stop();
  const distant = Math.abs(target - cursor) > 90;
  if (distant) { $("buffering").hidden = false; await paint(); }
  if (target < cursor) { M._shim_load_cart(cartPtr); cursor = 0; }
  while (cursor < target) {
    let j = cursor;
    while (j < target && input[j] === input[cursor]) j++;
    M._shim_step(j - cursor, input[cursor]);
    cursor = j;
    if ((cursor & 1023) === 0) {
      renderCursor();
      await tick();
      if (seq !== seekSeq) return;
    }
  }
  if (seq !== seekSeq) return;
  draw(); syncPanel(); renderCursor();
  $("buffering").hidden = true;
  if (autoplay) start();
}

function start() {
  if (playing) return;
  if (cursor >= input.length) { seekTo(0, true); return; }
  playing = true;
  playLoop.last = 0; playLoop.acc = 0;
  $("playPause").textContent = "pause";
  raf = requestAnimationFrame(playLoop);
}

function stop() {
  playing = false;
  if (raf) cancelAnimationFrame(raf);
  raf = null;
  $("playPause").textContent = "play";
}

function playLoop(now) {
  if (!playing) return;
  if (!playLoop.last) playLoop.last = now;
  let acc = (playLoop.acc || 0) + (now - playLoop.last);
  playLoop.last = now;
  const interval = 1000 / (FPS * speed);
  while (acc >= interval && cursor < input.length) { M._shim_step(1, input[cursor]); cursor++; acc -= interval; }
  playLoop.acc = acc > interval ? 0 : acc;
  draw(); syncPanel(); renderCursor();
  raf = cursor >= input.length ? (stop(), null) : requestAnimationFrame(playLoop);
}

async function loadRun(name) {
  stop();
  seekSeq++;
  $("buffering").hidden = true;
  $("meta").textContent = name;
  $("list").innerHTML = ""; $("panel").innerHTML = ""; shown = null;
  const url = name.split("/").map(encodeURIComponent).join("/");
  const response = await fetch(`/data/runs/${url}.json`);
  if (!response.ok) { $("meta").textContent = `no replay for ${name}`; return; }
  const data = await response.json();
  input = Uint8Array.from(atob(data.input), c => c.charCodeAt(0));
  decisions = data.decisions;
  buildList(); buildTimeline();
  await boot;
  M._shim_load_cart(cartPtr);
  cursor = 0;
  $("meta").textContent = `${data.label} · ${data.harness_label || data.harness} · ${data.mode}`
    + (data.progress == null ? "" : ` · ${Number(data.progress).toFixed(2)}%`);
  draw(); show(decisions[0] || null); renderCursor();
}

$("playPause").onclick = () => playing ? stop() : start();
$("toStart").onclick = () => seekTo(0, true);
$("back").onclick = () => seekTo(cursor - SKIP, true);
$("fwd").onclick = () => seekTo(cursor + SKIP, true);
$("toEnd").onclick = () => seekTo(input.length, true);
$("speed").onchange = e => { speed = Number(e.target.value); e.target.blur(); };
$("timeline").onclick = e => seekTo(e.offsetX / $("timeline").clientWidth * input.length, true);
window.onpopstate = () => {
  const name = new URLSearchParams(location.search).get("run");
  if (name) loadRun(name).catch(error => $("pollError").textContent = error.message);
};
document.onkeydown = e => {
  if (e.target.closest?.("input, select")) return;
  const jump = { ArrowLeft: -5 * FPS, ArrowRight: 5 * FPS, KeyJ: -SKIP, KeyL: SKIP }[e.code];
  if (e.code === "Space") { e.preventDefault(); playing ? stop() : start(); }
  else if (jump) { e.preventDefault(); seekTo(cursor + jump, true); }
  else if (e.code === "Home") { e.preventDefault(); seekTo(0, true); }
  else if (e.code === "End") { e.preventDefault(); seekTo(input.length, true); }
};

// Debug surface: lets us check a published replay against the native emulator.
window.player = {
  seek: seekTo, play: start, pause: stop,
  get frames() { return input.length; },
  get cursor() { return cursor; },
  get speed() { return speed; },
  set speed(v) { speed = v; },
  pixels: () => M.HEAPU8.slice(fbPtr, fbPtr + 128 * 128 * 4),
};

(async () => {
  try {
    const name = new URLSearchParams(location.search).get("run");
    if (name) await loadRun(name);
    else $("meta").textContent = "pick a run from the leaderboard";
  } catch (error) { $("pollError").textContent = error.message; }
})();
