/*
 * Kit de widgets dibujados a mano para el Control Panel.
 *
 * Los widgets nativos de litegraph (el clasico "flecha etiqueta valor flecha")
 * son funcionales pero no se pueden estilar: litegraph los dibuja el mismo
 * segun su `type`. La unica salida es declarar widgets de tipo desconocido con
 * su propio `draw` y su propio `mouse`, que es lo que hay aqui.
 *
 * Cada fabrica devuelve el widget ya insertado en node.widgets.
 */

export const C = {
  bg: "#141b18",
  card: "#1b2622",
  cardHi: "#22302b",
  slot: "#0f1613",
  line: "#2c3a34",
  green: "#5ecf8f",
  greenDim: "#2f6b4c",
  blue: "#5aa9e6",
  amber: "#e8b04b",
  red: "#e8695f",
  text: "#dfe7e3",
  dim: "#8fa39a",
  faint: "#5d6f67",
};

const ROW = 30;

/* ------------------------------------------------------------ helpers */
export function rrect(ctx, x, y, w, h, r) {
  const rr = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

export function wrap(ctx, text, maxW) {
  const out = [];
  for (const para of String(text).split("\n")) {
    let line = "";
    for (const word of para.split(" ")) {
      const t = line ? line + " " + word : word;
      if (ctx.measureText(t).width > maxW && line) { out.push(line); line = word; }
      else line = t;
    }
    out.push(line);
  }
  return out;
}

export function ellipsis(ctx, text, maxW) {
  let t = String(text);
  if (ctx.measureText(t).width <= maxW) return t;
  while (t.length > 1 && ctx.measureText(t + "…").width > maxW) t = t.slice(0, -1);
  return t + "…";
}

/** Y local del raton dentro del nodo, para el estado hover. */
function localMouseY(app, node) {
  try {
    const gm = app.canvas?.graph_mouse;
    if (!gm) return -1;
    return gm[1] - node.pos[1];
  } catch (e) { return -1; }
}

function isDown(t) { return /down/i.test(t); }
function isMove(t) { return /move/i.test(t); }
function isUp(t) { return /up/i.test(t); }

function push(node, w) {
  w.serialize = false;
  w.options = Object.assign({ serialize: false }, w.options || {});
  node.widgets.push(w);
  return w;
}

/* ------------------------------------------------------------ estaticos */
export function cpHeader(node, num, title, subtitle, colour) {
  return push(node, {
    type: "cp_header", name: "", value: "", __cpKind: "art",
    computeSize: () => [0, subtitle ? 54 : 40],
    draw(ctx, n, W, y) {
      ctx.save();
      ctx.fillStyle = colour;
      rrect(ctx, 12, y + 7, 3, subtitle ? 38 : 24, 2);
      ctx.fill();
      let tx = 22;
      if (num) {
        ctx.beginPath();
        ctx.arc(34, y + 16, 10, 0, Math.PI * 2);
        ctx.fillStyle = colour;
        ctx.fill();
        ctx.fillStyle = C.bg;
        ctx.font = "bold 12px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(String(num), 34, y + 20);
        ctx.textAlign = "left";
        tx = 52;
      }
      ctx.fillStyle = colour;
      ctx.font = "bold 13px sans-serif";
      ctx.fillText(title, tx, y + 20);
      if (subtitle) {
        ctx.fillStyle = C.dim;
        ctx.font = "11px sans-serif";
        const l = wrap(ctx, subtitle, W - tx - 16);
        ctx.fillText(l[0], tx, y + 38);
        if (l[1]) ctx.fillText(l[1], tx, y + 50);
      }
      ctx.restore();
    },
  });
}

export function cpCaption(node, text) {
  return push(node, {
    type: "cp_caption", name: "", value: "", __cpKind: "art",
    computeSize: () => [0, 22],
    draw(ctx, n, W, y) {
      ctx.save();
      ctx.fillStyle = C.faint;
      ctx.font = "bold 9px sans-serif";
      const label = String(text).toUpperCase();
      ctx.fillText(label, 18, y + 15);
      const tw = ctx.measureText(label).width;
      ctx.strokeStyle = C.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(18 + tw + 10, y + 11);
      ctx.lineTo(W - 16, y + 11);
      ctx.stroke();
      ctx.restore();
    },
  });
}

export function cpHint(node, text) {
  const probe = document.createElement("canvas").getContext("2d");
  probe.font = "11px sans-serif";
  let cache = null, cw = 0;
  const lines = (w) => {
    if (!cache || cw !== w) { cache = wrap(probe, text, w); cw = w; }
    return cache;
  };
  return push(node, {
    type: "cp_hint", name: "", value: "", __cpKind: "art",
    computeSize: (width) => [0, lines((width || 320) - 34).length * 14 + 8],
    draw(ctx, n, W, y) {
      ctx.save();
      ctx.fillStyle = C.dim;
      ctx.font = "11px sans-serif";
      let yy = y + 12;
      for (const l of lines(W - 34)) { ctx.fillText(l, 18, yy); yy += 14; }
      ctx.restore();
    },
  });
}

/* ------------------------------------------------------------ interactivos */

/** Boton con relleno, icono opcional y realce al pasar por encima. */
export function cpButton(node, app, label, onClick, opts = {}) {
  const primary = !!opts.primary;
  return push(node, {
    type: "cp_button", name: label, value: "", __cpKind: "btn",
    computeSize: () => [0, 34],
    draw(ctx, n, W, y, H) {
      const hov = this.__hover;
      const x = 14, w = W - 28, h = 26;
      ctx.save();
      ctx.fillStyle = primary
        ? (hov ? "#2b7d55" : C.greenDim)
        : (hov ? C.cardHi : C.card);
      rrect(ctx, x, y + 3, w, h, 6);
      ctx.fill();
      ctx.strokeStyle = primary ? C.green : C.line;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = primary ? "#eafff3" : (hov ? C.text : C.dim);
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(label, x + w / 2, y + 20);
      ctx.textAlign = "left";
      ctx.restore();
      this.__hover = false;
      const my = localMouseY(app, n);
      if (my >= y && my <= y + H) this.__hover = true;
    },
    mouse(e, pos, n) {
      if (isDown(e.type)) { onClick(); return true; }
      return false;
    },
  });
}

/** Interruptor tipo switch, con contador de nodos afectados. */
export function cpToggle(node, app, label, count, get, set) {
  return push(node, {
    type: "cp_toggle", name: label, value: get(), __cpKind: "switch",
    computeSize: () => [0, 32],
    draw(ctx, n, W, y, H) {
      // se relee del grafo en cada frame: si alguien cambia el modo de un nodo
      // por fuera del panel, el interruptor lo refleja al instante
      if (!this.__drag) this.value = get();
      const on = this.value;
      const x = 14, w = W - 28, h = 26;
      const hov = this.__hover;
      ctx.save();
      ctx.fillStyle = hov ? C.cardHi : C.card;
      rrect(ctx, x, y + 3, w, h, 6);
      ctx.fill();
      ctx.strokeStyle = on ? C.greenDim : C.line;
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = on ? C.text : C.faint;
      ctx.font = "bold 11px sans-serif";
      ctx.fillText(ellipsis(ctx, label, w - 130), x + 12, y + 20);

      if (count) {
        ctx.font = "9px sans-serif";
        const badge = count + (count === 1 ? " nodo" : " nodos");
        const bw = ctx.measureText(badge).width + 12;
        ctx.fillStyle = C.slot;
        rrect(ctx, x + w - 62 - bw, y + 9, bw, 14, 7);
        ctx.fill();
        ctx.fillStyle = C.faint;
        ctx.fillText(badge, x + w - 56 - bw, y + 19);
      }

      const sx = x + w - 52, sy = y + 8, sw = 40, sh = 16;
      ctx.fillStyle = on ? C.green : "#39433d";
      rrect(ctx, sx, sy, sw, sh, 8);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(on ? sx + sw - 8 : sx + 8, sy + sh / 2, 6, 0, Math.PI * 2);
      ctx.fillStyle = on ? "#0f1c15" : C.dim;
      ctx.fill();
      ctx.restore();

      this.__hover = false;
      const my = localMouseY(app, n);
      if (my >= y && my <= y + H) this.__hover = true;
    },
    mouse(e, pos, n) {
      if (isDown(e.type)) {
        this.value = !this.value;
        set(this.value);
        return true;
      }
      return false;
    },
  });
}

/** Fila etiqueta + valor. Arrastra en horizontal para numeros, clic para editar. */
export function cpValue(node, app, label, get, set, cfg = {}) {
  const kind = cfg.kind || "text";           // text | number | combo
  const values = cfg.values || [];
  const step = cfg.step ?? 0.01;
  const min = cfg.min ?? -Infinity;
  const max = cfg.max ?? Infinity;
  const prec = cfg.precision ?? (Number.isInteger(step) ? 0 : 2);

  const fmt = (v) => {
    if (kind === "number") {
      const n = Number(v);
      return Number.isFinite(n) ? n.toFixed(prec) : String(v);
    }
    return String(v ?? "");
  };

  return push(node, {
    type: "cp_value", name: label, value: get(), __cpKind: "proxy",
    computeSize: () => [0, ROW],
    draw(ctx, n, W, y, H) {
      this.value = get();
      const x = 14, w = W - 28, h = ROW - 6;
      const hov = this.__hover;
      ctx.save();
      ctx.fillStyle = hov ? C.cardHi : C.card;
      rrect(ctx, x, y + 2, w, h, 6);
      ctx.fill();
      ctx.strokeStyle = hov ? C.line : "transparent";
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = C.dim;
      ctx.font = "11px sans-serif";
      const labW = Math.min(ctx.measureText(label).width, w * 0.45);
      ctx.fillText(ellipsis(ctx, label, w * 0.45), x + 12, y + 18);

      const vx = x + 16 + labW;
      const vw = w - (16 + labW) - 14;
      ctx.fillStyle = C.slot;
      rrect(ctx, vx, y + 5, vw, h - 6, 5);
      ctx.fill();

      ctx.fillStyle = kind === "number" ? C.green : C.text;
      ctx.font = kind === "number" ? "bold 11px sans-serif" : "11px sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(ellipsis(ctx, fmt(this.value), vw - 22), vx + vw - 10, y + 18);
      ctx.textAlign = "left";

      if (kind === "combo") {
        ctx.strokeStyle = C.faint;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(vx + 8, y + 12);
        ctx.lineTo(vx + 12, y + 16);
        ctx.lineTo(vx + 16, y + 12);
        ctx.stroke();
      } else if (kind === "number") {
        ctx.fillStyle = C.faint;
        ctx.font = "9px sans-serif";
        ctx.fillText("arrastra", vx + 8, y + 18);
      }
      ctx.restore();

      this.__hover = false;
      const my = localMouseY(app, n);
      if (my >= y && my <= y + H) this.__hover = true;
    },
    mouse(e, pos, n) {
      if (isDown(e.type)) {
        this.__drag = { x: e.canvasX ?? pos[0], v: Number(get()) || 0, moved: false };
        if (kind === "combo" && values.length) {
          try {
            new LiteGraph.ContextMenu(values, {
              event: e, className: "dark",
              callback: (v) => { set(v); n.setDirtyCanvas(true, true); },
            });
          } catch (err) { console.warn("[CP] menu", err); }
          this.__drag = null;
          return true;
        }
        return true;
      }
      if (isMove(e.type) && this.__drag && kind === "number") {
        const dx = (e.canvasX ?? pos[0]) - this.__drag.x;
        if (Math.abs(dx) > 2) {
          this.__drag.moved = true;
          let v = this.__drag.v + dx * step;
          v = Math.min(max, Math.max(min, v));
          set(Number(v.toFixed(Math.max(prec, 3))));
          n.setDirtyCanvas(true, true);
        }
        return true;
      }
      if (isUp(e.type) && this.__drag) {
        const wasDrag = this.__drag.moved;
        this.__drag = null;
        if (!wasDrag && kind !== "combo") {
          const cb = (txt) => {
            if (txt === null || txt === undefined) return;
            set(kind === "number" ? Number(txt) : txt);
            n.setDirtyCanvas(true, true);
          };
          try { app.canvas.prompt(label, get(), cb, e); }
          catch (err) { cb(window.prompt(label, String(get()))); }
        }
        return true;
      }
      return false;
    },
  });
}
