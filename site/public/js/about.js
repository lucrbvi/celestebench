const RUN = "gpt-6-astra/2026-09-15-14-58-57-326458ee";
const FPS = 30;

(async () => {
  const figure = document.getElementById("aboutDemo");
  try {
    const url = "/data/runs/" + RUN.split("/").map(encodeURIComponent).join("/") + ".json";
    const [run, M] = await Promise.all([
      fetch(url).then(r => { if (!r.ok) throw Error(`run fetch failed (${r.status})`); return r.json(); }),
      createOpen8({ locateFile: f => "/player/" + f, canvas: document.getElementById("demo") }),
    ]);
    const input = Uint8Array.from(atob(run.input), c => c.charCodeAt(0));
    const ctx = document.getElementById("demo").getContext("2d");
    const fb = M._malloc(128 * 128 * 4);
    const image = ctx.createImageData(128, 128);
    const cart = M.stringToNewUTF8("/1CELESTE.PNG");
    if (M._shim_init() !== 0 || M._shim_load_cart(cart) !== 0) throw Error("open8 failed to start");
    let i = 0, last = 0;
    requestAnimationFrame(function loop(now) {
      try {
        if (now - last >= 1000 / FPS) {
          last = now;
          M._shim_step(1, input[i]);
          if (++i >= input.length) { M._shim_load_cart(cart); i = 0; }
          M._shim_framebuffer(fb);
          image.data.set(M.HEAPU8.subarray(fb, fb + 128 * 128 * 4));
          ctx.putImageData(image, 0, 0);
        }
        requestAnimationFrame(loop);
      } catch (error) { figure.hidden = true; }
    });
  } catch (error) { figure.hidden = true; }
})();
