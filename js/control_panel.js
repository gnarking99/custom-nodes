import { app } from "../../scripts/app.js";

/*
 * CP Control Panel v2
 * -------------------
 * Panel unico para artistas. Controla nodos REPARTIDOS por todo el grafo sin
 * obligar a moverlos: la pertenencia se marca en el TITULO del nodo.
 *
 *   #sw:nombre  -> interruptor que enciende/apaga ese conjunto
 *   #ui         -> los widgets de ese nodo se reflejan aqui
 *   #img        -> la imagen de ese nodo se previsualiza aqui (y se puede
 *                  abrir el editor de mascara desde el panel)
 */

const NODE_ID = "CP_ControlPanel";
const TAG_SW = /#sw:([A-Za-z0-9_\-]+)/g;
const TAG_UI = /#ui\b/;
const TAG_IMG = /#img\b/;

const MODE_ON = 0;
const MODE_MUTE = 2;
const MODE_BYPASS = 4;

const IMG_H = 210;
const DIAG_H = 190;

const COL = {
  head: "#7fd6a0",
  sub: "#6aa9d6",
  warn: "#e8b04b",
  text: "#cfd6dd",
  dim: "#8b949e",
  line: "#3a4048",
};

/* --------------------------------------------------------- diagnostico */
const DIAG = [
  ["— elige un sintoma —",
   "Elige abajo que le pasa a tu imagen y aqui aparece que tocar."],

  ["La cara o la pose CAMBIAN",
   "Baja el denoise. En el panel: Calidad -> ajuste_denoise a -0.06.\n" +
   "Si sigue, pon la calidad en 'Solo limpiar'.\n" +
   "Regla: por encima de 0.55 el modelo reinterpreta al personaje."],

  ["No recupera NADA de detalle",
   "Sube el denoise: ajuste_denoise a +0.05.\n" +
   "Comprueba tambien que el interruptor 'vlm' este ACTIVO: si no,\n" +
   "el prompt no dice que corregir y el modelo no sabe que anadir."],

  ["Sale con textura plastica",
   "Baja el LoRA de personaje del PASE 2 a 0.20.\n" +
   "En el pase de refinado el LoRA solo tiene que sostener el estilo."],

  ["Se pierde la edicion en el resultado final",
   "El PROMPT 2 debe DESCRIBIR lo editado, no ordenarlo otra vez.\n" +
   "'wearing a striped tie' si · 'add the tie' no.\n" +
   "Y baja el LoRA de personaje del PASE 2."],

  ["El fondo cambia y no deberia",
   "Enciende el interruptor 'mascara' y pinta encima del personaje.\n" +
   "Boton 'Abrir editor de mascara' aqui arriba."],

  ["El resultado es IGUAL que el original",
   "El ImageCompositeMasked esta tapando el resultado con una mascara\n" +
   "vacia. Mira el preview de la mascara: si esta todo negro, no llego.\n" +
   "Apaga el interruptor 'mascara' o repinta."],

  ["Los colores se van",
   "Sube el strength del ColorMatch a 0.85."],

  ["Tarda demasiado",
   "Calidad -> 'Borrador (rapido)' mientras pruebas encuadre y prompt.\n" +
   "Apaga los interruptores 'canon-auto' y 'pase-2' hasta el render final."],

  ["Da error y no entiendo el mensaje",
   "Mira el cuadro ESTADO del flujo: traduce los errores a castellano.\n" +
   "Los avisos marcados [!!] son los que hay que arreglar."],
];

/* --------------------------------------------------------- utilidades */
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
    const title = String(n.title || "");
    TAG_SW.lastIndex = 0;
    let m;
    while ((m = TAG_SW.exec(title)) !== null) {
      if (!switches.has(m[1])) switches.set(m[1], []);
      switches.get(m[1]).push(n);
    }
    if (TAG_UI.test(title)) uiNodes.push(n);
    if (!imgNode && TAG_IMG.test(title)) imgNode = n;
  }
  return { switches, uiNodes, imgNode };
}

function setMode(nodes, on, offMode) {
  for (const n of nodes) n.mode = on ? MODE_ON : offMode;
  app.graph.setDirtyCanvas(true, true);
}

function toast(msg, severity = "info") {
  try {
    app.extensionManager?.toast?.add({ severity, summary: "Panel de control", detail: msg, life: 4000 });
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

/* Abre el editor de mascara del nodo objetivo. Tres rutas, de mas a menos
 * moderna; la ultima siempre funciona porque solo lleva al nodo. */
function openMaskEditor(target) {
  if (!target) {
    toast("Marca tu nodo de imagen con #img en el titulo.", "warn");
    return;
  }
  focusNode(target);

  try {
    const cmd = app.extensionManager?.command;
    if (cmd?.execute) {
      cmd.execute("Comfy.MaskEditor.OpenMaskEditor");
      return;
    }
  } catch (e) { /* siguiente ruta */ }

  try {
    const CA = window.ComfyApp || app.constructor;
    if (CA?.copyToClipspace && CA?.open_maskeditor) {
      CA.copyToClipspace(target);
      CA.clipspace_return_node = target;
      CA.open_maskeditor();
      return;
    }
  } catch (e) { /* siguiente ruta */ }

  toast("He seleccionado tu nodo de imagen: boton derecho sobre el -> Open in MaskEditor.", "warn");
}

/* --------------------------------------------------------- construccion */
function sep(self, label) {
  const w = self.addWidget("text", label, "", () => {});
  w.__cpKind = "sep";
  w.serialize = false;
  w.disabled = true;
  return w;
}

function build(self) {
  const keep = new Map();
  for (const w of self.widgets || []) {
    if (w.__cpKind === "switch") keep.set(w.__cpKey, w.value);
  }
  const diagIdx = self.__cpDiag ?? 0;
  self.widgets = [];

  const { switches, uiNodes, imgNode } = scan(self);
  const offMode = self.properties?.["cp.off_mode"] === "mute" ? MODE_MUTE : MODE_BYPASS;
  self.__cpImgNode = imgNode;

  /* --- imagen y mascara --- */
  sep(self, "IMAGEN");
  const bMask = self.addWidget("button", "Abrir editor de mascara", null,
    () => openMaskEditor(self.__cpImgNode));
  bMask.__cpKind = "btn"; bMask.serialize = false;

  const bGo = self.addWidget("button", "Ir al nodo de imagen", null,
    () => self.__cpImgNode ? focusNode(self.__cpImgNode)
                           : toast("Marca tu nodo de imagen con #img.", "warn"));
  bGo.__cpKind = "btn"; bGo.serialize = false;

  /* --- interruptores --- */
  if (switches.size) {
    sep(self, "QUE SE EJECUTA");
    for (const key of [...switches.keys()].sort()) {
      const targets = switches.get(key);
      const cur = keep.has(key) ? keep.get(key) : targets.every((n) => n.mode === MODE_ON);
      const w = self.addWidget("toggle", `${key}  (${targets.length})`, cur,
        (v) => setMode(scan(self).switches.get(key) || [], v, offMode),
        { on: "ACTIVO", off: "apagado" });
      w.__cpKind = "switch"; w.__cpKey = key; w.serialize = false;
      setMode(targets, cur, offMode);
    }
    const bAll = self.addWidget("button", "Todo ACTIVO", null, () => {
      for (const nodes of scan(self).switches.values()) setMode(nodes, true, offMode);
      build(self);
    });
    bAll.__cpKind = "btn"; bAll.serialize = false;
  }

  /* --- valores reflejados --- */
  if (uiNodes.length) {
    sep(self, "VALORES");
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
        } catch (err) {
          console.warn("[CP_ControlPanel] widget no reflejado:", tw?.name, err);
        }
      }
    }
  }

  /* --- diagnostico --- */
  sep(self, "DIAGNOSTICO");
  const dw = self.addWidget("combo", "que le pasa", DIAG[diagIdx][0], (v) => {
    const i = DIAG.findIndex((d) => d[0] === v);
    self.__cpDiag = i < 0 ? 0 : i;
    app.graph.setDirtyCanvas(true, true);
  }, { values: DIAG.map((d) => d[0]) });
  dw.__cpKind = "diag"; dw.serialize = false;

  if (!switches.size && !uiNodes.length && !imgNode) {
    const w = self.addWidget("text", "sin nodos marcados",
      "pon #sw:nombre, #ui o #img en el titulo", () => {});
    w.__cpKind = "hint"; w.serialize = false;
  }

  const s = self.computeSize();
  self.setSize([Math.max(360, s[0]), s[1] + IMG_H + DIAG_H]);
  app.graph.setDirtyCanvas(true, true);
}

/* --------------------------------------------------------- dibujo */
function wrap(ctx, text, maxW) {
  const out = [];
  for (const para of String(text).split("\n")) {
    let line = "";
    for (const word of para.split(" ")) {
      const t = line ? line + " " + word : word;
      if (ctx.measureText(t).width > maxW && line) {
        out.push(line);
        line = word;
      } else {
        line = t;
      }
    }
    out.push(line);
  }
  return out;
}

function drawPanel(node, ctx) {
  if (node.flags?.collapsed) return;
  const w = node.size[0];
  const h = node.size[1];
  const pad = 10;

  const imgTop = h - IMG_H - DIAG_H;
  const diagTop = h - DIAG_H;

  /* imagen */
  ctx.save();
  ctx.strokeStyle = COL.line;
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, imgTop - 6); ctx.lineTo(w - pad, imgTop - 6); ctx.stroke();

  const img = node.__cpImgNode?.imgs?.[0];
  if (img && img.width) {
    const availW = w - pad * 2;
    const availH = IMG_H - 16;
    const k = Math.min(availW / img.width, availH / img.height);
    const dw = img.width * k, dh = img.height * k;
    ctx.drawImage(img, (w - dw) / 2, imgTop + (IMG_H - dh) / 2, dw, dh);
  } else {
    ctx.fillStyle = COL.dim;
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(node.__cpImgNode ? "(lanza una vez para ver la imagen)"
                                  : "marca tu nodo de imagen con #img",
                 w / 2, imgTop + IMG_H / 2);
    ctx.textAlign = "left";
  }

  /* diagnostico */
  ctx.beginPath(); ctx.moveTo(pad, diagTop - 6); ctx.lineTo(w - pad, diagTop - 6); ctx.stroke();
  const d = DIAG[node.__cpDiag ?? 0];
  ctx.font = "bold 12px sans-serif";
  ctx.fillStyle = (node.__cpDiag ?? 0) === 0 ? COL.sub : COL.warn;
  ctx.fillText(d[0], pad, diagTop + 12);
  ctx.font = "11px sans-serif";
  ctx.fillStyle = COL.text;
  let y = diagTop + 30;
  for (const line of wrap(ctx, d[1], w - pad * 2)) {
    if (y > h - 6) break;
    ctx.fillText(line, pad, y);
    y += 14;
  }
  ctx.restore();
}

/* --------------------------------------------------------- registro */
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
      this.color = "#1b3a2a";
      this.bgcolor = "#12251b";
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

    const onDraw = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      const r = onDraw?.apply(this, arguments);
      try { drawPanel(this, ctx); } catch (e) { /* nunca romper el canvas */ }
      return r;
    };

    nodeType.prototype.getExtraMenuOptions = function (_, options) {
      options.push(
        { content: "Refrescar panel", callback: () => build(this) },
        { content: "Abrir editor de mascara", callback: () => openMaskEditor(this.__cpImgNode) },
        {
          content: "Apagar: usar mute en vez de bypass",
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
