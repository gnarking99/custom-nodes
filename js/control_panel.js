import { app } from "../../scripts/app.js";

/*
 * CP Control Panel
 * ----------------
 * Un panel unico que controla nodos que estan REPARTIDOS por todo el grafo.
 *
 * El problema con Fast Groups Bypasser de rgthree es que opera sobre GRUPOS:
 * para poder apagar tres nodos juntos hay que moverlos fisicamente al mismo
 * grupo, y el grafo acaba con cables cruzando la pantalla.
 *
 * Aqui la pertenencia se marca en el TITULO del nodo, asi que cada nodo se
 * queda donde tiene sentido que este:
 *
 *   #sw:mascara   -> aparece un interruptor "mascara" en el panel
 *   #ui           -> los widgets de ese nodo se reflejan en el panel
 *
 * Un nodo puede llevar los dos. El titulo se edita con doble clic.
 */

const TAG_SW = /#sw:([A-Za-z0-9_\-]+)/g;
const TAG_UI = /#ui\b/;
const NODE_ID = "CP_ControlPanel";

const MODE_ON = 0;
const MODE_BYPASS = 4;
const MODE_MUTE = 2;

function allNodes() {
  const g = app.graph;
  if (!g) return [];
  return g._nodes || g.nodes || [];
}

function cleanTitle(n) {
  const t = String(n.title || n.type || "nodo");
  return t.replace(TAG_SW, "").replace(TAG_UI, "").replace(/\s+/g, " ").trim() || n.type;
}

function scan(self) {
  const switches = new Map();  // nombre -> [nodos]
  const uiNodes = [];
  for (const n of allNodes()) {
    if (n === self || n.type === NODE_ID) continue;
    const title = String(n.title || "");
    TAG_SW.lastIndex = 0;
    let m;
    while ((m = TAG_SW.exec(title)) !== null) {
      const key = m[1];
      if (!switches.has(key)) switches.set(key, []);
      switches.get(key).push(n);
    }
    if (TAG_UI.test(title)) uiNodes.push(n);
  }
  return { switches, uiNodes };
}

function setGroupMode(nodes, on, offMode) {
  for (const n of nodes) n.mode = on ? MODE_ON : offMode;
  app.graph.setDirtyCanvas(true, true);
}

function build(self) {
  const keepValues = new Map();
  for (const w of self.widgets || []) {
    if (w.__cpKind === "switch") keepValues.set(w.__cpKey, w.value);
  }

  // conserva solo los widgets propios del nodo (los declarados en Python)
  self.widgets = (self.widgets || []).filter((w) => !w.__cpKind);

  const { switches, uiNodes } = scan(self);
  const offMode = self.properties?.["cp.off_mode"] === "mute" ? MODE_MUTE : MODE_BYPASS;

  if (switches.size === 0 && uiNodes.length === 0) {
    const w = self.addWidget("text", "sin etiquetas", "pon #sw:nombre o #ui en el titulo de un nodo", () => {});
    w.__cpKind = "hint";
    w.serialize = false;
  }

  const sorted = [...switches.keys()].sort();
  for (const key of sorted) {
    const targets = switches.get(key);
    const current = keepValues.has(key)
      ? keepValues.get(key)
      : targets.every((n) => n.mode === MODE_ON);

    const w = self.addWidget(
      "toggle",
      `${key}  (${targets.length})`,
      current,
      (v) => setGroupMode(scan(self).switches.get(key) || [], v, offMode),
      { on: "ACTIVO", off: "apagado" }
    );
    w.__cpKind = "switch";
    w.__cpKey = key;
    w.serialize = false;
    // sincroniza el grafo con el estado mostrado
    setGroupMode(targets, current, offMode);
  }

  for (const target of uiNodes) {
    const label = cleanTitle(target);
    for (const tw of target.widgets || []) {
      if (!tw || tw.__cpKind) continue;
      if (tw.type === "converted-widget" || tw.type === "hidden") continue;
      if (tw.name === "control_after_generate") continue;
      try {
        const proxy = self.addWidget(
          tw.type,
          `${label} · ${tw.name}`,
          tw.value,
          (v) => {
            tw.value = v;
            if (typeof tw.callback === "function") tw.callback(v, app.canvas, target);
            app.graph.setDirtyCanvas(true, true);
          },
          tw.options || {}
        );
        proxy.__cpKind = "proxy";
        proxy.serialize = false;
      } catch (err) {
        console.warn("[CP_ControlPanel] no se pudo reflejar el widget", tw?.name, err);
      }
    }
  }

  self.setSize(self.computeSize());
  app.graph.setDirtyCanvas(true, true);
}

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

      this.addWidget("button", "Refrescar panel", null, () => build(this));
      this.addWidget("button", "Todo ACTIVO", null, () => {
        const { switches } = scan(this);
        for (const nodes of switches.values()) setGroupMode(nodes, true, MODE_BYPASS);
        build(this);
      });
      this.addWidget("button", "Todo apagado", null, () => {
        const { switches } = scan(this);
        const off = this.properties["cp.off_mode"] === "mute" ? MODE_MUTE : MODE_BYPASS;
        for (const nodes of switches.values()) setGroupMode(nodes, false, off);
        build(this);
      });
      for (const w of this.widgets || []) w.serialize = false;

      setTimeout(() => build(this), 100);
      return r;
    };

    // reconstruye al cargar un workflow guardado
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      setTimeout(() => build(this), 200);
      return r;
    };
  },

  async setup() {
    // refresco perezoso: si cambia el numero de nodos, repinta los paneles
    let lastCount = -1;
    setInterval(() => {
      const nodes = allNodes();
      if (nodes.length === lastCount) return;
      lastCount = nodes.length;
      for (const n of nodes) {
        if (n.type === NODE_ID) build(n);
      }
    }, 1500);
  },
});
