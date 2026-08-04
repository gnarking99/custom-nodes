import { app } from "../../scripts/app.js";
import {
  C, rrect, wrap, ellipsis,
  cpHeader, cpCaption, cpHint, cpButton, cpToggle, cpValue,
} from "./widgets.js";

/*
 * CP Control Panel v4
 * -------------------
 * Panel unico para artistas. Controla nodos REPARTIDOS por todo el grafo sin
 * obligar a moverlos: la pertenencia se marca en el TITULO del nodo.
 *
 *   #sw:nombre  -> interruptor que enciende/apaga ese conjunto
 *   #ui         -> los widgets de ese nodo se reflejan aqui
 *   #img        -> la imagen de ese nodo se previsualiza aqui
 *
 * Todo se dibuja con el kit de widgets.js: los nativos de litegraph no se
 * pueden estilar porque los pinta el propio motor segun su `type`.
 */

const NODE_ID = "CP_ControlPanel";
const TAG_SW = /#sw:([A-Za-z0-9_\-]+)/g;
const TAG_UI = /#ui\b/;
const TAG_IMG = /#img\b/;

const MODE_ON = 0;
const MODE_MUTE = 2;
const MODE_BYPASS = 4;
const IMG_H = 230;

/* ------------------------------------------------------------ utilidades */
function allNodes() {
  const g = app.graph;
  if (!g) return [];
  return g._nodes || g.nodes || [];
}

/** Nombre corto y legible de un nodo: sin etiquetas, sin flechas, sin "1)". */
function shortTitle(n) {
  let t = String(n.title || n.type || "nodo")
    .replace(TAG_SW, "").replace(TAG_UI, "").replace(TAG_IMG, "");
  t = t.split(/→|->|\bAQUI\b|:/)[0];
  t = t.replace(/^\s*\d+\)\s*/, "").replace(/[—–-]\s*$/, "");
  t = t.replace(/\s+/g, " ").trim();
  if (t.length > 26) t = t.slice(0, 25) + "…";
  return t || n.type;
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
  } catch (e) { console.log("[CP_ControlPanel]", msg); }
}

function focusNode(n) {
  try {
    app.canvas.selectNode(n);
    app.canvas.centerOnNode(n);
    app.graph.setDirtyCanvas(true, true);
  } catch (e) { /* noop */ }
}

function openMaskEditor(target) {
  if (!target) { toast("Marca tu nodo de imagen con #img en el titulo.", "warn"); return; }
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

/* ------------------------------------------------------- piezas visuales */
function imageCard(node) {
  node.widgets.push({
    type: "cp_img", name: "", value: "", serialize: false, __cpKind: "art",
    options: { serialize: false },
    computeSize: () => [0, IMG_H],
    draw(ctx, n, W, y) {
      const x = 14, w = W - 28, h = IMG_H - 12;
      ctx.save();
      ctx.fillStyle = C.slot;
      rrect(ctx, x, y, w, h, 8);
      ctx.fill();
      ctx.strokeStyle = C.line;
      ctx.lineWidth = 1;
      ctx.stroke();

      const target = n.__cpImgNode;
      const img = target?.imgs?.[0];
      if (img && img.width) {
        const k = Math.min((w - 16) / img.width, (h - 16) / img.height);
        const dw = img.width * k, dh = img.height * k;
        const ix = x + (w - dw) / 2, iy = y + (h - dh) / 2;
        ctx.save();
        rrect(ctx, ix, iy, dw, dh, 5);
        ctx.clip();
        ctx.drawImage(img, ix, iy, dw, dh);
        ctx.restore();

        const painted = String(target?.widgets?.[0]?.value || "").includes("clipspace");
        const label = painted ? "MASCARA PINTADA" : "SIN MASCARA";
        ctx.font = "bold 9px sans-serif";
        const tw = ctx.measureText(label).width + 18;
        ctx.fillStyle = "rgba(8,14,11,0.86)";
        rrect(ctx, ix + 8, iy + 8, tw, 18, 9);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(ix + 17, iy + 17, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = painted ? C.green : C.faint;
        ctx.fill();
        ctx.fillStyle = painted ? C.green : C.faint;
        ctx.fillText(label, ix + 25, iy + 21);
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
    },
  });
}

function denoiseGauge(node, getValue) {
  node.widgets.push({
    type: "cp_gauge", name: "", value: "", serialize: false, __cpKind: "art",
    options: { serialize: false },
    computeSize: () => [0, 52],
    draw(ctx, n, W, y) {
      const v = getValue();
      if (v === null || v === undefined) return;
      const x = 18, w = W - 36, h = 8, by = y + 24;
      ctx.save();
      ctx.fillStyle = C.faint;
      ctx.font = "9px sans-serif";
      ctx.fillText("RESPETA EL ORIGINAL", x, y + 14);
      ctx.textAlign = "right";
      ctx.fillText("MAS DETALLE NUEVO", x + w, y + 14);
      ctx.textAlign = "left";

      for (const [a, b, col] of [[0, .35, C.greenDim], [.35, .72, C.green], [.72, 1, C.red]]) {
        ctx.fillStyle = col;
        ctx.fillRect(x + w * a, by, w * (b - a), h);
      }
      const t = Math.max(0, Math.min(1, (v + 0.2) / 0.4));
      const px = x + w * t;
      ctx.beginPath();
      ctx.arc(px, by + h / 2, 7, 0, Math.PI * 2);
      ctx.fillStyle = "#0c1410";
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = t > .72 ? C.red : (t > .35 ? C.green : C.blue);
      ctx.stroke();

      ctx.fillStyle = t > .72 ? C.red : C.dim;
      ctx.font = "bold 10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(t > .72 ? "ZONA DE RIESGO · LA CARA PUEDE CAMBIAR"
                 : (t < .28 ? "MUY CONSERVADOR" : "ZONA UTIL"),
                   x + w / 2, by + 24);
      ctx.textAlign = "left";
      ctx.restore();
    },
  });
}

const DIAG = [
  ["— elige un sintoma —",
   "Selecciona arriba que le pasa a tu imagen y aqui aparece exactamente que tocar."],
  ["La cara o la pose CAMBIAN",
   "Baja el denoise: ajuste_denoise a -0.06. Si sigue, pon la calidad en 'Solo limpiar'. Por encima de 0.55 el modelo reinterpreta al personaje en vez de restaurarlo."],
  ["No recupera NADA de detalle",
   "Sube el denoise: ajuste_denoise a +0.05. Y comprueba que el interruptor 'vlm' este ACTIVO: si no, el prompt no dice que corregir y el modelo no sabe que anadir."],
  ["Sale con textura plastica",
   "Baja el LoRA de personaje del PASE 2 a 0.20. En el refinado el LoRA solo tiene que sostener el estilo, no reconstruir la identidad."],
  ["Se pierde la edicion en el resultado",
   "El PROMPT 2 debe DESCRIBIR lo editado, no ordenarlo otra vez. 'wearing a striped tie' si, 'add the tie' no. Y baja el LoRA de personaje del PASE 2."],
  ["El fondo cambia y no deberia",
   "Enciende el interruptor 'mascara' y pinta encima del personaje con el boton de arriba."],
  ["El resultado es IGUAL que el original",
   "El composite esta tapando el resultado con una mascara vacia. Mira la insignia de la imagen: si dice SIN MASCARA, no llego. Apaga el interruptor 'mascara' o repinta."],
  ["Los colores se van",
   "Sube el strength del ColorMatch a 0.85."],
  ["Tarda demasiado",
   "Calidad -> 'Borrador (rapido)' mientras pruebas encuadre y prompt. Apaga los interruptores 'canon-auto' y 'pase-2' hasta el render final."],
  ["Da error y no lo entiendo",
   "Mira el cuadro ESTADO del flujo: traduce los errores a castellano. Los avisos [!!] son los que hay que arreglar."],
];

function diagCard(node, getDiag) {
  const probe = document.createElement("canvas").getContext("2d");
  probe.font = "11px sans-serif";
  node.widgets.push({
    type: "cp_diag", name: "", value: "", serialize: false, __cpKind: "art",
    options: { serialize: false },
    computeSize: (width) => [0, wrap(probe, getDiag()[1], (width || 340) - 50).length * 15 + 46],
    draw(ctx, n, W, y) {
      const d = getDiag();
      const first = (n.__cpDiag ?? 0) === 0;
      const col = first ? C.blue : C.amber;
      ctx.save();
      ctx.font = "11px sans-serif";
      const lines = wrap(ctx, d[1], W - 50);
      const h = lines.length * 15 + 38;
      ctx.fillStyle = C.card;
      rrect(ctx, 14, y, W - 28, h, 8);
      ctx.fill();
      ctx.fillStyle = col;
      rrect(ctx, 14, y, 3, h, 2);
      ctx.fill();
      ctx.fillStyle = col;
      ctx.font = "bold 12px sans-serif";
      ctx.fillText(first ? "Elige tu sintoma arriba" : d[0], 26, y + 19);
      ctx.fillStyle = C.text;
      ctx.font = "11px sans-serif";
      let yy = y + 36;
      for (const l of lines) { ctx.fillText(l, 26, yy); yy += 15; }
      ctx.restore();
    },
  });
}

/* ------------------------------------------------------------ construccion */
function kindOf(tw) {
  const t = String(tw.type || "").toLowerCase();
  if (t === "combo" || Array.isArray(tw.options?.values)) return "combo";
  if (t === "number" || t === "slider" || typeof tw.value === "number") return "number";
  if (t === "toggle" || typeof tw.value === "boolean") return "bool";
  return "text";
}

function build(self) {
  const keep = new Map();
  for (const w of self.widgets || []) {
    if (w.__cpKind === "switch") keep.set(w.name, w.value);
  }
  self.widgets = [];

  const { switches, uiNodes, imgNode } = scan(self);
  const offMode = self.properties?.["cp.off_mode"] === "mute" ? MODE_MUTE : MODE_BYPASS;
  self.__cpImgNode = imgNode;

  /* 1 · imagen */
  cpHeader(self, 1, "TU IMAGEN", "Carga el render y pinta encima del personaje.", C.green);
  imageCard(self);
  cpButton(self, app, "PINTAR MASCARA", () => openMaskEditor(self.__cpImgNode), { primary: true });
  cpButton(self, app, "Ir al nodo de imagen",
    () => self.__cpImgNode ? focusNode(self.__cpImgNode)
                           : toast("Marca tu nodo de imagen con #img.", "warn"));

  /* 2 · que se ejecuta */
  if (switches.size) {
    cpHeader(self, 2, "QUE SE EJECUTA",
      "Cada interruptor enciende varios nodos a la vez, esten donde esten.", C.blue);
    for (const key of [...switches.keys()].sort()) {
      const targets = switches.get(key);
      const cur = keep.has(key) ? keep.get(key) : targets.every((n) => n.mode === MODE_ON);
      cpToggle(self, app, key, targets.length,
        // getter en vivo: el estado real es el modo de los nodos, no una copia
        () => {
          const t = scan(self).switches.get(key) || [];
          return t.length ? t.every((n) => n.mode === MODE_ON) : false;
        },
        (v) => setMode(scan(self).switches.get(key) || [], v, offMode));
      setMode(targets, cur, offMode);
    }
    cpHint(self, "Apagar lo que no uses acelera mucho las pruebas. Para el render final, enciendelo todo.");
    cpButton(self, app, "Encender todo", () => {
      for (const nodes of scan(self).switches.values()) setMode(nodes, true, offMode);
      build(self);
    });
  }

  /* 3 · ajustes, agrupados por nodo de origen */
  if (uiNodes.length) {
    cpHeader(self, 3, "AJUSTES", "Estos son los unicos numeros que hay que tocar.", C.amber);
    let adj = null;
    for (const target of uiNodes) {
      const rows = (target.widgets || []).filter((tw) =>
        tw && !tw.__cpKind &&
        tw.type !== "converted-widget" && tw.type !== "hidden" &&
        tw.name !== "control_after_generate");
      if (!rows.length) continue;
      cpCaption(self, shortTitle(target));
      for (const tw of rows) {
        const kind = kindOf(tw);
        try {
          if (kind === "bool") {
            cpToggle(self, app, tw.name, 0, () => !!tw.value, (v) => {
              tw.value = v;
              tw.callback?.(v, app.canvas, target);
              app.graph.setDirtyCanvas(true, true);
            });
          } else {
            cpValue(self, app, tw.name,
              () => tw.value,
              (v) => {
                tw.value = v;
                tw.callback?.(v, app.canvas, target);
                app.graph.setDirtyCanvas(true, true);
              },
              {
                kind,
                values: tw.options?.values || [],
                step: tw.options?.step ? tw.options.step / 10 : 0.01,
                min: tw.options?.min, max: tw.options?.max,
                precision: tw.options?.precision,
              });
          }
          if (tw.name === "ajuste_denoise") adj = tw;
        } catch (err) {
          console.warn("[CP_ControlPanel] widget no reflejado:", tw?.name, err);
        }
      }
    }
    if (adj) denoiseGauge(self, () => Number(adj.value) || 0);
  }

  /* 4 · diagnostico */
  cpHeader(self, 4, "ALGO NO SALE BIEN?", "Elige el sintoma y te digo que tocar.", C.red);
  cpValue(self, app, "sintoma",
    () => DIAG[self.__cpDiag ?? 0][0],
    (v) => {
      const i = DIAG.findIndex((d) => d[0] === v);
      self.__cpDiag = i < 0 ? 0 : i;
      self.setSize(self.computeSize());
      app.graph.setDirtyCanvas(true, true);
    },
    { kind: "combo", values: DIAG.map((d) => d[0]) });
  diagCard(self, () => DIAG[self.__cpDiag ?? 0]);

  if (!switches.size && !uiNodes.length && !imgNode) {
    cpHint(self, "Ningun nodo marcado todavia. Doble clic en el titulo de un nodo y anade #sw:nombre, #ui o #img.");
  }

  const s = self.computeSize();
  self.setSize([Math.max(400, s[0]), s[1]]);
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

    const onBg = nodeType.prototype.onDrawBackground;
    nodeType.prototype.onDrawBackground = function (ctx) {
      const r = onBg?.apply(this, arguments);
      if (this.flags?.collapsed) return r;
      try {
        ctx.save();
        const g = ctx.createLinearGradient(0, 0, this.size[0], 0);
        g.addColorStop(0, "rgba(94,207,143,0.22)");
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
