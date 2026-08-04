import { app } from "../../scripts/app.js";

/*
 * CP Control Panel v3
 * -------------------
 * Panel unico para artistas. Controla nodos REPARTIDOS por todo el grafo sin
 * obligar a moverlos: la pertenencia se marca en el TITULO del nodo.
 *
 *   #sw:nombre  -> interruptor que enciende/apaga ese conjunto
 *   #ui         -> los widgets de ese nodo se reflejan aqui
 *   #img        -> la imagen de ese nodo se previsualiza aqui
 *
 * Todo lo visual se dibuja con widgets personalizados (objetos con su propio
 * metodo draw + computeSize). Es la unica forma limpia de mezclar dibujo propio
 * con los widgets normales de litegraph: si se dibujara en onDrawForeground
 * habria que adivinar las posiciones y se solaparia al cambiar de tamano.
 */

const NODE_ID = "CP_ControlPanel";
const TAG_SW = /#sw:([A-Za-z0-9_\-]+)/g;
const TAG_UI = /#ui\b/;
const TAG_IMG = /#img\b/;

const MODE_ON = 0;
const MODE_MUTE = 2;
const MODE_BYPASS = 4;

const C = {
  bg: "#141b18",
  card: "#1b2622",
  cardAlt: "#182120",
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

const IMG_H = 230;

/* ------------------------------------------------------------ utilidades */
function allNodes() {
  const g = app.graph;
  if (!g) return [];
  return g._nodes || g.nodes || [];
}

function cleanTitle(n) {
  return String(n.title || n.type || "nodo")
    .replace(TAG_SW, "").replace(TAG_UI, "").replace(TAG_IMG, "")
    .replace(/\s+/g, " ").trim() || n.type;
}

function scan(self) {
  const switches = new Map();
  const uiNodes = [];
  let imgNode = null;
  for (const n of allNodes()) {
    if (n === self || n.type === NODE_ID) continue;
    const t = String(n.title || "");
    TAG_SW.lastIndex = 0;
    let m;
    while ((m = TAG_SW.exec(t)) !== null) {
      if (!switches.has(m[1])) switches.set(m[1], []);
      switches.get(m[1]).push(n);
    }
    if (TAG_UI.test(t)) uiNodes.push(n);
    if (!imgNode && TAG_IMG.test(t)) imgNode = n;
  }
  return { switches, uiNodes, imgNode };
}

function setMode(nodes, on, offMode) {
  for (const n of nodes) n.mode = on ? MODE_ON : offMode;
  app.graph.setDirtyCanvas(true, true);
}

function toast(msg, severity = "info") {
  try {
    app.extensionManager?.toast?.add({
      severity, summary: "Panel de control", detail: msg, life: 4500,
    });
  } catch (e) {
    console.log("[CP_ControlPanel]", msg);
  }
}

function focusNode(n) {
  try {
    app.canvas.selectNode(n);
    app.canvas.centerOnNode(n);
    app.graph.setDirtyCanvas(true, true);
  } catch (e) { /* noop */ }
}

function openMaskEditor(target) {
  if (!target) {
    toast("Marca tu nodo de imagen con #img en el titulo.", "warn");
    return;
  }
  focusNode(target);
  try {
    const cmd = app.extensionManager?.command;
    if (cmd?.execute) { cmd.execute("Comfy.MaskEditor.OpenMaskEditor"); return; }
  } catch (e) { /* siguiente */ }
  try {
    const CA = window.ComfyApp || app.constructor;
    if (CA?.copyToClipspace && CA?.open_maskeditor) {
      CA.copyToClipspace(target);
      CA.clipspace_return_node = target;
      CA.open_maskeditor();
      return;
    }
  } catch (e) { /* siguiente */ }
  toast("Nodo de imagen seleccionado: boton derecho -> Open in MaskEditor.", "warn");
}

/* ------------------------------------------------------------ dibujo base */
function rrect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

function wrap(ctx, text, maxW) {
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

/* ------------------------------------------------------- widgets propios */
function custom(node, height, draw) {
  const w = {
    type: "cp_custom", name: "", value: "", options: { serialize: false },
    serialize: false, __cpKind: "art",
    computeSize: () => [0, height],
    draw,
  };
  node.widgets.push(w);
  return w;
}

/** Cabecera de seccion: numero en circulo, titulo y subtitulo. */
function header(node, num, title, subtitle, colour) {
  custom(node, subtitle ? 52 : 38, (ctx, n, W, y) => {
    const x = 12, w = W - 24;
    ctx.save();
    ctx.fillStyle = colour;
    rrect(ctx, x, y + 6, 3, (subtitle ? 36 : 22), 2);
    ctx.fill();

    if (num) {
      ctx.beginPath();
      ctx.arc(x + 22, y + 15, 10, 0, Math.PI * 2);
      ctx.fillStyle = colour;
      ctx.fill();
      ctx.fillStyle = C.bg;
      ctx.font = "bold 12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(String(num), x + 22, y + 19);
      ctx.textAlign = "left";
    }

    ctx.fillStyle = colour;
    ctx.font = "bold 13px sans-serif";
    ctx.fillText(title, x + (num ? 40 : 12), y + 19);

    if (subtitle) {
      ctx.fillStyle = C.dim;
      ctx.font = "11px sans-serif";
      const lines = wrap(ctx, subtitle, w - (num ? 40 : 12));
      ctx.fillText(lines[0], x + (num ? 40 : 12), y + 36);
      if (lines[1]) ctx.fillText(lines[1], x + (num ? 40 : 12), y + 48);
    }
    ctx.restore();
  });
}

/** Texto de ayuda en gris. */
function hint(node, text) {
  let cache = null, cacheW = 0;
  const probe = document.createElement("canvas").getContext("2d");
  probe.font = "11px sans-serif";
  const measure = (w) => {
    if (!cache || cacheW !== w) { cache = wrap(probe, text, w); cacheW = w; }
    return cache;
  };
  const w = {
    type: "cp_custom", name: "", value: "", serialize: false, __cpKind: "art",
    computeSize: (width) => [0, measure((width || 300) - 30).length * 14 + 8],
    draw: (ctx, n, W, y) => {
      ctx.save();
      ctx.fillStyle = C.dim;
      ctx.font = "11px sans-serif";
      let yy = y + 12;
      for (const l of measure(W - 30)) { ctx.fillText(l, 16, yy); yy += 14; }
      ctx.restore();
    },
  };
  node.widgets.push(w);
  return w;
}

/** Tarjeta con la imagen del nodo #img + insignia de estado de la mascara. */
function imageCard(node) {
  custom(node, IMG_H, (ctx, n, W, y) => {
    const x = 12, w = W - 24, h = IMG_H - 12;
    ctx.save();
    ctx.fillStyle = C.cardAlt;
    rrect(ctx, x, y, w, h, 8);
    ctx.fill();
    ctx.strokeStyle = C.line;
    ctx.lineWidth = 1;
    ctx.stroke();

    const target = n.__cpImgNode;
    const img = target?.imgs?.[0];
    if (img && img.width) {
      const k = Math.min((w - 16) / img.width, (h - 30) / img.height);
      const dw = img.width * k, dh = img.height * k;
      const ix = x + (w - dw) / 2, iy = y + 8;
      ctx.save();
      rrect(ctx, ix, iy, dw, dh, 5);
      ctx.clip();
      ctx.drawImage(img, ix, iy, dw, dh);
      ctx.restore();

      const painted = String(target?.widgets?.[0]?.value || "").includes("clipspace");
      const label = painted ? "MASCARA PINTADA" : "sin mascara";
      const col = painted ? C.green : C.faint;
      ctx.font = "bold 10px sans-serif";
      const tw = ctx.measureText(label).width + 16;
      ctx.fillStyle = "rgba(10,16,13,0.82)";
      rrect(ctx, ix + 6, iy + 6, tw, 18, 9);
      ctx.fill();
      ctx.fillStyle = col;
      ctx.fillText(label, ix + 14, iy + 19);
    } else {
      ctx.fillStyle = C.faint;
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(target ? "lanza una vez para ver la imagen"
                          : "marca tu nodo de imagen con  #img",
                   x + w / 2, y + h / 2);
      ctx.textAlign = "left";
    }
    ctx.restore();
  });
}

/** Medidor del ajuste de denoise, con zonas de color. */
function denoiseGauge(node, getValue) {
  custom(node, 46, (ctx, n, W, y) => {
    const x = 16, w = W - 32, h = 8, by = y + 22;
    const v = getValue();
    if (v === null) return;
    ctx.save();
    ctx.fillStyle = C.dim;
    ctx.font = "11px sans-serif";
    ctx.fillText("respeta el original", x, y + 14);
    ctx.textAlign = "right";
    ctx.fillText("mas detalle nuevo", x + w, y + 14);
    ctx.textAlign = "left";

    const zones = [[0, 0.35, C.greenDim], [0.35, 0.72, C.green], [0.72, 1, C.red]];
    for (const [a, b, col] of zones) {
      ctx.fillStyle = col;
      ctx.fillRect(x + w * a, by, w * (b - a), h);
    }
    ctx.fillStyle = C.line;
    ctx.fillRect(x, by, w, 1);

    const t = Math.max(0, Math.min(1, (v + 0.2) / 0.4));
    const px = x + w * t;
    ctx.beginPath();
    ctx.arc(px, by + h / 2, 7, 0, Math.PI * 2);
    ctx.fillStyle = "#0d1512";
    ctx.fill();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = t > 0.72 ? C.red : (t > 0.35 ? C.green : C.blue);
    ctx.stroke();

    ctx.fillStyle = t > 0.72 ? C.red : C.text;
    ctx.font = "bold 11px sans-serif";
    ctx.textAlign = "center";
    const msg = t > 0.72 ? "zona de riesgo: la cara puede cambiar"
              : (t < 0.28 ? "muy conservador" : "zona util");
    ctx.fillText(msg, x + w / 2, by + 24);
    ctx.textAlign = "left";
    ctx.restore();
  });
}

/** Tarjeta de diagnostico: sintoma en color + solucion. */
function diagCard(node, getDiag) {
  const probe = document.createElement("canvas").getContext("2d");
  probe.font = "11px sans-serif";
  const w = {
    type: "cp_custom", name: "", value: "", serialize: false, __cpKind: "art",
    computeSize: (width) => {
      const d = getDiag();
      return [0, wrap(probe, d[1], (width || 320) - 46).length * 15 + 46];
    },
    draw: (ctx, n, W, y) => {
      const d = getDiag();
      const first = (n.__cpDiag ?? 0) === 0;
      const col = first ? C.blue : C.amber;
      ctx.save();
      ctx.font = "11px sans-serif";
      const lines = wrap(ctx, d[1], W - 46);
      const h = lines.length * 15 + 38;
      ctx.fillStyle = C.card;
      rrect(ctx, 12, y, W - 24, h, 8);
      ctx.fill();
      ctx.fillStyle = col;
      rrect(ctx, 12, y, 3, h, 2);
      ctx.fill();

      ctx.fillStyle = col;
      ctx.font = "bold 12px sans-serif";
      ctx.fillText(first ? "Elige tu sintoma arriba" : d[0], 24, y + 18);
      ctx.fillStyle = C.text;
      ctx.font = "11px sans-serif";
      let yy = y + 34;
      for (const l of lines) { ctx.fillText(l, 24, yy); yy += 15; }
      ctx.restore();
    },
  };
  node.widgets.push(w);
  return w;
}

/* ------------------------------------------------------------ diagnostico */
const DIAG = [
  ["— elige un sintoma —",
   "Selecciona arriba que le pasa a tu imagen y aqui aparece exactamente que tocar."],
  ["La cara o la pose CAMBIAN",
   "Baja el denoise: Calidad -> ajuste_denoise a -0.06. Si sigue, pon la calidad en 'Solo limpiar'. Por encima de 0.55 el modelo reinterpreta al personaje en vez de restaurarlo."],
  ["No recupera NADA de detalle",
   "Sube el denoise: ajuste_denoise a +0.05. Y comprueba que el interruptor 'vlm' este ACTIVO: si no, el prompt no dice que corregir y el modelo no sabe que anadir."],
  ["Sale con textura plastica",
   "Baja el LoRA de personaje del PASE 2 a 0.20. En el refinado el LoRA solo tiene que sostener el estilo, no reconstruir la identidad."],
  ["Se pierde la edicion en el resultado",
   "El PROMPT 2 debe DESCRIBIR lo editado, no ordenarlo otra vez. 'wearing a striped tie' si, 'add the tie' no. Y baja el LoRA de personaje del PASE 2."],
  ["El fondo cambia y no deberia",
   "Enciende el interruptor 'mascara' y pinta encima del personaje con el boton de arriba."],
  ["El resultado es IGUAL que el original",
   "El composite esta tapando el resultado con una mascara vacia. Mira la insignia de la imagen: si dice 'sin mascara', no llego. Apaga el interruptor 'mascara' o repinta."],
  ["Los colores se van",
   "Sube el strength del ColorMatch a 0.85."],
  ["Tarda demasiado",
   "Calidad -> 'Borrador (rapido)' mientras pruebas encuadre y prompt. Apaga los interruptores 'canon-auto' y 'pase-2' hasta el render final."],
  ["Da error y no lo entiendo",
   "Mira el cuadro ESTADO del flujo: traduce los errores a castellano. Los avisos marcados [!!] son los que hay que arreglar."],
];

/* ------------------------------------------------------------ construccion */
function build(self) {
  const keep = new Map();
  for (const w of self.widgets || []) {
    if (w.__cpKind === "switch") keep.set(w.__cpKey, w.value);
  }
  self.widgets = [];

  const { switches, uiNodes, imgNode } = scan(self);
  const offMode = self.properties?.["cp.off_mode"] === "mute" ? MODE_MUTE : MODE_BYPASS;
  self.__cpImgNode = imgNode;

  /* ---------- 1 · imagen ---------- */
  header(self, 1, "TU IMAGEN", "Carga el render y pinta encima del personaje.", C.green);
  imageCard(self);
  const b1 = self.addWidget("button", "Pintar mascara", null, () => openMaskEditor(self.__cpImgNode));
  b1.__cpKind = "btn"; b1.serialize = false;
  const b2 = self.addWidget("button", "Ir al nodo de imagen", null,
    () => self.__cpImgNode ? focusNode(self.__cpImgNode)
                           : toast("Marca tu nodo de imagen con #img.", "warn"));
  b2.__cpKind = "btn"; b2.serialize = false;

  /* ---------- 2 · que se ejecuta ---------- */
  if (switches.size) {
    header(self, 2, "QUE SE EJECUTA",
      "Cada interruptor enciende varios nodos a la vez, esten donde esten.", C.blue);
    for (const key of [...switches.keys()].sort()) {
      const targets = switches.get(key);
      const cur = keep.has(key) ? keep.get(key) : targets.every((n) => n.mode === MODE_ON);
      const w = self.addWidget("toggle", `${key}   ·   ${targets.length} nodos`, cur,
        (v) => setMode(scan(self).switches.get(key) || [], v, offMode),
        { on: "ACTIVO", off: "apagado" });
      w.__cpKind = "switch"; w.__cpKey = key; w.serialize = false;
      setMode(targets, cur, offMode);
    }
    hint(self, "Apagar lo que no uses acelera mucho las pruebas. Para el render final, enciendelo todo.");
    const bAll = self.addWidget("button", "Encender todo", null, () => {
      for (const nodes of scan(self).switches.values()) setMode(nodes, true, offMode);
      build(self);
    });
    bAll.__cpKind = "btn"; bAll.serialize = false;
  }

  /* ---------- 3 · valores ---------- */
  if (uiNodes.length) {
    header(self, 3, "AJUSTES", "Estos son los unicos numeros que hay que tocar.", C.amber);
    let adjWidget = null;
    for (const target of uiNodes) {
      const label = cleanTitle(target);
      for (const tw of target.widgets || []) {
        if (!tw || tw.__cpKind) continue;
        if (tw.type === "converted-widget" || tw.type === "hidden") continue;
        if (tw.name === "control_after_generate") continue;
        try {
          const p = self.addWidget(tw.type, `${label} · ${tw.name}`, tw.value, (v) => {
            tw.value = v;
            if (typeof tw.callback === "function") tw.callback(v, app.canvas, target);
            app.graph.setDirtyCanvas(true, true);
          }, tw.options || {});
          p.__cpKind = "proxy"; p.serialize = false;
          if (tw.name === "ajuste_denoise") adjWidget = tw;
        } catch (err) {
          console.warn("[CP_ControlPanel] widget no reflejado:", tw?.name, err);
        }
      }
    }
    if (adjWidget) denoiseGauge(self, () => Number(adjWidget.value) || 0);
  }

  /* ---------- 4 · diagnostico ---------- */
  header(self, 4, "ALGO NO SALE BIEN?", "Elige el sintoma y te digo que tocar.", C.red);
  const dw = self.addWidget("combo", "sintoma", DIAG[self.__cpDiag ?? 0][0], (v) => {
    const i = DIAG.findIndex((d) => d[0] === v);
    self.__cpDiag = i < 0 ? 0 : i;
    self.setSize(self.computeSize());
    app.graph.setDirtyCanvas(true, true);
  }, { values: DIAG.map((d) => d[0]) });
  dw.__cpKind = "diag"; dw.serialize = false;
  diagCard(self, () => DIAG[self.__cpDiag ?? 0]);

  if (!switches.size && !uiNodes.length && !imgNode) {
    hint(self, "Ningun nodo marcado todavia. Doble clic en el titulo de un nodo y anade #sw:nombre, #ui o #img.");
  }

  const s = self.computeSize();
  self.setSize([Math.max(390, s[0]), s[1]]);
  app.graph.setDirtyCanvas(true, true);
}

/* ------------------------------------------------------------ registro */
app.registerExtension({
  name: "character-pipeline.control-panel",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      this.properties = this.properties || {};
      if (!("cp.off_mode" in this.properties)) this.properties["cp.off_mode"] = "bypass";
      this.serialize_widgets = false;
      this.color = "#16281f";
      this.bgcolor = C.bg;
      this.__cpDiag = 0;
      setTimeout(() => build(this), 100);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      setTimeout(() => build(this), 200);
      return r;
    };

    // franja superior con el titulo, para que se lea de lejos
    const onDraw = nodeType.prototype.onDrawBackground;
    nodeType.prototype.onDrawBackground = function (ctx) {
      const r = onDraw?.apply(this, arguments);
      if (this.flags?.collapsed) return r;
      try {
        ctx.save();
        const g = ctx.createLinearGradient(0, 0, this.size[0], 0);
        g.addColorStop(0, "rgba(94,207,143,0.18)");
        g.addColorStop(1, "rgba(94,207,143,0)");
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, this.size[0], 3);
        ctx.restore();
      } catch (e) { /* nunca romper el canvas */ }
      return r;
    };

    nodeType.prototype.getExtraMenuOptions = function (_, options) {
      options.push(
        { content: "Refrescar panel", callback: () => build(this) },
        { content: "Pintar mascara", callback: () => openMaskEditor(this.__cpImgNode) },
        {
          content: "Apagar con mute en vez de bypass",
          callback: () => {
            this.properties["cp.off_mode"] =
              this.properties["cp.off_mode"] === "mute" ? "bypass" : "mute";
            toast("Modo apagado: " + this.properties["cp.off_mode"]);
            build(this);
          },
        },
      );
    };
  },

  async setup() {
    let last = -1;
    setInterval(() => {
      const nodes = allNodes();
      if (nodes.length === last) return;
      last = nodes.length;
      for (const n of nodes) if (n.type === NODE_ID) build(n);
    }, 1500);
  },
});
