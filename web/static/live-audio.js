const liveSounds = new Map();

function syncLiveSound(button, name) {
  const sound = liveSounds.get(name);
  if (sound) sound.button = button;
  button.textContent = sound ? "mute" : "sound";
  button.setAttribute("aria-pressed", sound ? "true" : "false");
}

async function toggleLiveSound(button, name) {
  const current = liveSounds.get(name);
  if (current) {
    current.stopped = true;
    liveSounds.delete(name);
    await current.context.close();
    syncLiveSound(button, name);
    return;
  }
  const context = new AudioContext();
  await context.resume();
  const sound = {context, button, offset: "tail", next: 0, stopped: false};
  liveSounds.set(name, sound);
  syncLiveSound(button, name);

  async function pump() {
    if (sound.stopped) return;
    try {
      const response = await fetch(`/live-audio/${encodeURIComponent(name)}?offset=${sound.offset}`, {cache:"no-store"});
      if (!response.ok) throw Error(`audio stream ${response.status}`);
      sound.offset = response.headers.get("X-Audio-Offset") || sound.offset;
      const bytes = await response.arrayBuffer();
      if (bytes.byteLength >= 2) {
        const rate = +(response.headers.get("X-Audio-Rate") || 22050);
        const samples = new Int16Array(bytes.slice(0, bytes.byteLength & ~1));
        const buffer = context.createBuffer(1, samples.length, rate);
        const channel = buffer.getChannelData(0);
        for (let i = 0; i < samples.length; i++) channel[i] = samples[i] / 32768;
        const source = context.createBufferSource();
        source.buffer = buffer;
        source.connect(context.destination);
        const when = Math.max(context.currentTime + .06, sound.next);
        source.start(when);
        sound.next = when + buffer.duration;
      }
      setTimeout(pump, sound.next > context.currentTime + .25 ? 100 : 40);
    } catch (error) {
      sound.stopped = true;
      liveSounds.delete(name);
      context.close();
      button.textContent = "sound unavailable";
      button.setAttribute("aria-pressed", "false");
    }
  }
  pump();
}

function stopLiveSound(name) {
  const sound = liveSounds.get(name);
  if (!sound) return;
  sound.stopped = true;
  liveSounds.delete(name);
  sound.context.close();
}

function retainLiveSounds(names) {
  const kept = new Set(names);
  for (const name of liveSounds.keys()) if (!kept.has(name)) stopLiveSound(name);
}
