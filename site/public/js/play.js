const $ = id => document.getElementById(id);
const FPS = 30;
const CART = '/1CELESTE.PNG';
let budget = 300 * FPS;   // replaced by the leaderboard's budget once it loads
// Matched on the labelled key (e.key, layout-aware) and on the physical
// position (e.code, US layout), so AZERTY, QWERTZ and friends find Z and X
// where they are printed on their own keyboard.
const BINDINGS = {
  ArrowLeft: 1, ArrowRight: 2, ArrowUp: 4, ArrowDown: 8,
  KeyZ: 16, KeyY: 16, KeyC: 16, KeyN: 16, Space: 16,
  KeyX: 32, KeyV: 32, KeyM: 32,
  z: 16, y: 16, c: 16, n: 16, " ": 16,
  x: 32, v: 32, m: 32,
};

const ctx = $("screen").getContext("2d");
const time = f => `${Math.floor(f / FPS / 60)}:${String(Math.floor(f / FPS) % 60).padStart(2, "0")}`;
const finite = v => typeof v === "number" && Number.isFinite(v) ? v : null;

// Mirrors celestebench/scoring.py (grounded_height_v1): progress only moves up.
class Progress {
  constructor() { this.rooms = 0; this.room_progress = 0; this.room = null; this.deaths = null; this.grounded = 0; }
  get percent() { return 100 * (this.rooms + this.room_progress) / 30; }
  update(s) {
    if (!s || finite(s.room) === null || finite(s.deaths) === null) { this.grounded = 0; return; }
    if (s.room !== this.room || s.deaths !== this.deaths) this.grounded = 0;
    if (this.room === this.rooms && s.room === this.rooms + 1 && s.room <= 30) {
      this.rooms = s.room;
      this.room_progress = 0;
    }
    this.room = s.room;
    this.deaths = s.deaths;
    const feet = finite(s.feet_y);
    if (!(s.room >= 0 && s.room < 30 && s.alive && s.grounded && feet !== null)) { this.grounded = 0; return; }
    this.grounded++;
    if (this.grounded >= 3 && s.room === this.rooms) {
      const start = finite(s.spawn_feet_y), end = finite(s.exit_feet_y);
      if (start !== null && end !== null && start > end) {
        const fraction = Math.max(0, Math.min(0.999999, (start - feet) / (start - end)));
        this.room_progress = Math.max(this.room_progress, fraction);
      }
    }
  }
}

let M, fbPtr, imageData, cartPtr;
let running = false, raf = null, acc = 0, last = 0, frames = 0, progress = new Progress();
let board = null;
const down = new Map();

const boot = createOpen8({ locateFile: f => "/player/" + f, canvas: $("screen") }).then(m => {
  M = m;
  fbPtr = M._malloc(128 * 128 * 4);
  imageData = ctx.createImageData(128, 128);
  cartPtr = M.stringToNewUTF8(CART);
  if (M._shim_init() !== 0 || M._shim_load_cart(cartPtr) !== 0) throw Error("open8 failed to start");
});

function draw() {
  M._shim_framebuffer(fbPtr);
  imageData.data.set(M.HEAPU8.subarray(fbPtr, fbPtr + 128 * 128 * 4));
  ctx.putImageData(imageData, 0, 0);
}

// shim_game_state_t; keep the offsets in sync with csrc/shim.c and open8.py.
function gameState() {
  const p = M._shim_game_state();
  if (!p) return null;
  return {
    room: M.HEAP32[p >> 2], alive: M.HEAP32[(p + 4) >> 2] !== 0,
    feet_y: M.HEAPF32[(p + 8) >> 2], grounded: M.HEAP32[(p + 12) >> 2] !== 0,
    spawn_feet_y: M.HEAPF32[(p + 16) >> 2], exit_feet_y: M.HEAPF32[(p + 20) >> 2],
    deaths: M.HEAP32[(p + 24) >> 2],
  };
}

function binding(e) {
  const byCode = BINDINGS[e.code];
  if (byCode !== undefined) return { id: e.code, bit: byCode };
  const key = (e.key || "").toLowerCase();
  return BINDINGS[key] === undefined ? null : { id: "key:" + key, bit: BINDINGS[key] };
}

// Standard-mapping gamepad: the face buttons jump and dash, the D-pad or the
// left stick move. A and Y jump, B and X dash, so every common pad works.
function padMask() {
  let m = 0;
  for (const pad of navigator.getGamepads?.() || []) {
    if (!pad?.connected) continue;
    const held = i => pad.buttons[i]?.pressed || pad.buttons[i]?.value > 0.5;
    if (held(0) || held(3)) m |= 16;
    if (held(1) || held(2)) m |= 32;
    if (held(14) || pad.axes[0] < -0.4) m |= 1;
    if (held(15) || pad.axes[0] > 0.4) m |= 2;
    if (held(12) || pad.axes[1] < -0.4) m |= 4;
    if (held(13) || pad.axes[1] > 0.4) m |= 8;
  }
  return m;
}

const mask = () => { let m = padMask(); for (const bit of down.values()) m |= bit; return m; };

function hud() {
  $("progress").textContent = progress.percent.toFixed(2) + "%";
  $("rooms").textContent = progress.rooms;
  $("deaths").textContent = progress.deaths ?? 0;
  $("time").textContent = time(Math.max(0, budget - frames));
}

function loop(now) {
  if (!running) return;
  if (!last) last = now;
  acc += now - last;
  last = now;
  const interval = 1000 / FPS;
  while (acc >= interval && frames < budget) {
    M._shim_step(1, mask());
    frames++;
    progress.update(gameState());
    acc -= interval;
  }
  if (acc > interval) acc = 0;
  draw();
  hud();
  if (frames >= budget) return finish();
  raf = requestAnimationFrame(loop);
}

async function begin() {
  await boot;
  down.clear();
  M._shim_load_cart(cartPtr);
  progress = new Progress();
  frames = 0; acc = 0; last = 0;
  running = true;
  $("overlay").hidden = true;
  $("result").hidden = true;
  $("stop").disabled = false;
  $("go").textContent = "play again";
  draw();
  hud();
  raf = requestAnimationFrame(loop);
}

function finish() {
  running = false;
  if (raf) cancelAnimationFrame(raf);
  raf = null;
  $("stop").disabled = true;
  $("overlay").hidden = false;
  $("overlayText").textContent = `time! you reached ${progress.percent.toFixed(2)}%`
    + (frames < budget ? ` (stopped at ${time(frames)})` : "");
  renderResult();
  $("result").hidden = false;
}

function renderResult() {
  const your = progress.percent;
  const result = $("result");
  if (!board) {
    result.innerHTML = `<div class="card"><h3>your score</h3><p style="font-size:20px">${your.toFixed(2)}%</p></div>`;
    return;
  }
  // One line per model × harness, keeping the best setup of each, so the same
  // model never shows up three times for three thinking levels. Rank against
  // each setup's best run, not the mean the chart plots.
  const top = g => g.scored_runs[0].progress;
  const best = new Map();
  for (const g of board.groups) {
    if (!(g.scored > 0) || !Number.isFinite(g.progress)) continue;
    const key = g.model + "|" + ((g.settings || {}).harness || "tau");
    if (!best.has(key) || top(g) > top(best.get(key))) best.set(key, g);
  }
  const ranked = [...best.values()].sort((a, b) => top(b) - top(a));
  const rank = 1 + ranked.filter(g => top(g) > your).length;
  const rows = ranked.map(g => ({ model: g.model, progress: top(g) }));
  rows.splice(rank - 1, 0, { model: "you", progress: your, you: true });
  const from = Math.max(0, Math.min(rank - 4, rows.length - 8));
  const nearby = rows.slice(from, from + 8);
  result.innerHTML = `<div class="card">
    <h3>your score</h3>
    <p style="font-size:20px;margin:0">${your.toFixed(2)}%</p>
    <p class="meta">you'd rank ${rank} of ${ranked.length} models · RTC</p>
    <div class="table-wrap"><table><thead><tr><th>#</th><th>model</th><th>progress</th></tr></thead><tbody>
    ${nearby.map((r, i) => `<tr class="${r.you ? "you" : ""}"><td class="meta">${from + i + 1}</td><td>${r.model}</td><td>${r.progress.toFixed(3)}%</td></tr>`).join("")}
    </tbody></table></div></div>`;
}

$("go").onclick = () => begin().catch(error => $("pollError").textContent = error.message);
$("stop").onclick = () => finish();

// Debug surface: drive the emulator head-lessly to check the metric against scoring.py.
window.play = {
  ready: false,
  get percent() { return progress.percent; },
  get frames() { return frames; },
  get mask() { return mask(); },
  get state() { return gameState(); },
  reset() { M._shim_load_cart(cartPtr); progress = new Progress(); frames = 0; draw(); hud(); },
  step(buttons, count) {
    for (let i = 0; i < count; i++) { M._shim_step(1, buttons); frames++; progress.update(gameState()); }
    draw(); hud();
  },
};

document.onkeydown = e => {
  if (e.target.closest?.("input, select")) return;
  const hit = binding(e);
  if (!hit) return;
  e.preventDefault();
  if (!e.repeat) down.set(hit.id, hit.bit);
};
document.onkeyup = e => { const hit = binding(e); if (hit) down.delete(hit.id); };
window.onblur = () => down.clear();

(async () => {
  try {
    await boot;
    draw();
    window.play.ready = true;
    const data = await fetch("/data/leaderboard.json").then(r => r.json());
    const entry = data.entries.rtc || Object.values(data.entries)[0];
    const wanted = Number(data.defaultBudget ?? entry.budgets[0]);
    const key = Object.keys(entry.byBudget).find(k => Number(k) === wanted) || Object.keys(entry.byBudget)[0];
    board = entry.byBudget[key];
    budget = wanted * FPS;
    hud();
  } catch (error) { $("pollError").textContent = error.message; }
})();
