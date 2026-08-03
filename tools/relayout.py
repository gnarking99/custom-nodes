# -*- coding: utf-8 -*-
"""Reorganiza un flujo de ComfyUI en columnas por profundidad topologica.

El flujo de datos en ComfyUI va de izquierda a derecha, asi que la columna
natural de un nodo es su distancia maxima desde una fuente. Colocando cada nodo
en su columna, los cables dejan de cruzarse hacia atras.
"""
import json, io, os
from collections import defaultdict

BASE = r"D:\Users\crist\Documents\GitHub\workflows"
SRC = os.path.join(BASE, "enhance lora propio v6.json")
DST = os.path.join(BASE, "enhance lora propio v7.json")

COL_W = 480          # ancho de columna
GAP_Y = 40           # separacion vertical entre nodos
GROUP_PAD = 60
NOTE_COL_GAP = 60

wf = json.load(open(SRC, encoding="utf-8"))
N = {n["id"]: n for n in wf["nodes"]}
links = wf["links"]

NOTES = [n for n in wf["nodes"] if n["type"] == "MarkdownNote"]
PANEL = [n for n in wf["nodes"] if n["type"] == "CP_ControlPanel"]
GRAPH = [n for n in wf["nodes"] if n["type"] not in ("MarkdownNote", "CP_ControlPanel")]
GIDS = {n["id"] for n in GRAPH}

# ---------------------------------------------------------------- topologia
preds = defaultdict(set)
succs = defaultdict(set)
for l in links:
    _, on, _, tn, _, _ = l[:6]
    if on in GIDS and tn in GIDS and on != tn:
        preds[tn].add(on)
        succs[on].add(tn)

depth = {}


def compute(nid, stack):
    if nid in depth:
        return depth[nid]
    if nid in stack:          # ciclo defensivo
        return 0
    stack.add(nid)
    p = preds.get(nid) or set()
    depth[nid] = 0 if not p else 1 + max(compute(x, stack) for x in p)
    stack.discard(nid)
    return depth[nid]


for n in GRAPH:
    compute(n["id"], set())

# los cargadores sin entradas se pegan a la columna de su primer consumidor
for n in GRAPH:
    if not preds.get(n["id"]) and succs.get(n["id"]):
        mn = min(depth[s] for s in succs[n["id"]])
        depth[n["id"]] = max(0, mn - 1)

cols = defaultdict(list)
for n in GRAPH:
    cols[depth[n["id"]]].append(n)

# ---------------------------------------------------------------- colocacion
ORIGIN_X, ORIGIN_Y = 0, 0
bounds = {}
# ancho real por columna: el nodo mas ancho manda, no una constante
col_x, cursor = {}, ORIGIN_X
for c in sorted(cols):
    col_x[c] = cursor
    widest = max(n.get("size", [300, 100])[0] for n in cols[c])
    cursor += max(COL_W, widest + 120)

for c in sorted(cols):
    # ordena por la altura media de sus predecesores: reduce cruces
    def key(n):
        p = preds.get(n["id"])
        if not p:
            return n["pos"][1]
        ys = [bounds[x][1] if x in bounds else N[x]["pos"][1] for x in p]
        return sum(ys) / len(ys)

    column = sorted(cols[c], key=key)
    x = col_x[c]
    y = ORIGIN_Y
    for n in column:
        h = n.get("size", [300, 100])[1]
        if n.get("flags", {}).get("collapsed"):
            h = 30
        n["pos"] = [x, y]
        bounds[n["id"]] = (x, y, n.get("size", [300, 100])[0], h)
        y += h + GAP_Y

# ---------------------------------------------------------------- grupos
STAGE = [
    ("0 · CARGA Y MODELOS", 0, 1, "#3f789e"),
    ("1 · REFERENCIAS Y CANON", 2, 4, "#b58b2a"),
    ("2 · PROMPT AUTOMATICO", 5, 7, "#a1309b"),
    ("3 · PASE 1 · EDICION", 8, 11, "#2a6"),
    ("4 · PUENTE Y PASE 2", 12, 15, "#A88"),
    ("5 · MASCARA Y SALIDA", 16, 99, "#8A8"),
]

maxdepth = max(depth.values()) if depth else 0
groups = []
for title, lo, hi, colr in STAGE:
    members = [n for n in GRAPH if lo <= depth[n["id"]] <= hi]
    if not members:
        continue
    xs = [n["pos"][0] for n in members]
    ys = [n["pos"][1] for n in members]
    x2 = max(n["pos"][0] + n.get("size", [300, 100])[0] for n in members)
    y2 = max(n["pos"][1] + (30 if n.get("flags", {}).get("collapsed")
                            else n.get("size", [300, 100])[1]) for n in members)
    groups.append({
        "id": len(groups) + 1, "title": title,
        "bounding": [min(xs) - GROUP_PAD, min(ys) - GROUP_PAD - 30,
                     x2 - min(xs) + GROUP_PAD * 2, y2 - min(ys) + GROUP_PAD * 2],
        "color": colr, "font_size": 26, "flags": {},
    })

# ---------------------------------------------------------------- panel y notas
total_h = max((b[1] + b[3]) for b in bounds.values()) if bounds else 800
panel_x = ORIGIN_X - 520 - NOTE_COL_GAP
for i, p in enumerate(PANEL):
    p["pos"] = [panel_x, ORIGIN_Y + i * 900]
    p["size"] = [420, 860]

note_x = cursor + NOTE_COL_GAP
ny = ORIGIN_Y
for n in NOTES:
    n["pos"] = [note_x, ny]
    if n.get("size", [0, 0])[0] < 420:
        n["size"] = [520, max(260, n.get("size", [0, 300])[1])]
    ny += n["size"][1] + GAP_Y

groups.append({
    "id": len(groups) + 1, "title": "PANEL DEL ARTISTA  —  empieza aqui",
    "bounding": [panel_x - GROUP_PAD, ORIGIN_Y - GROUP_PAD - 30,
                 420 + GROUP_PAD * 2, 900 + GROUP_PAD],
    "color": "#2a6", "font_size": 30, "flags": {},
})
groups.append({
    "id": len(groups) + 1, "title": "DOCUMENTACION",
    "bounding": [note_x - GROUP_PAD, ORIGIN_Y - GROUP_PAD - 30,
                 560 + GROUP_PAD * 2, max(400, ny - ORIGIN_Y) + GROUP_PAD],
    "color": "#555", "font_size": 26, "flags": {},
})

wf["groups"] = groups
wf["extra"] = wf.get("extra", {})
wf["extra"]["ds"] = {"scale": 0.32, "offset": [abs(panel_x) + 200, 200]}

json.dump(wf, io.open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ---------------------------------------------------------------- verificacion
errs, ids = [], set()
for l in links:
    lid, on, os_, tn, ts, ty = l[:6]
    if lid in ids:
        errs.append("dup")
    ids.add(lid)
    o, t = N[on], N[tn]
    if os_ >= len(o["outputs"]) or ts >= len(t["inputs"]):
        errs.append("L%d slot" % lid); continue
    if lid not in (o["outputs"][os_].get("links") or []):
        errs.append("L%d out" % lid)
    if t["inputs"][ts].get("link") != lid:
        errs.append("L%d in" % lid)

# solapes
overlap = 0
items = list(bounds.items())
for i in range(len(items)):
    _, (x1, y1, w1, h1) = items[i]
    for j in range(i + 1, len(items)):
        _, (x2, y2, w2, h2) = items[j]
        if x1 < x2 + w2 and x2 < x1 + w1 and y1 < y2 + h2 and y2 < y1 + h1:
            overlap += 1

print("columnas:", maxdepth + 1, "| nodos colocados:", len(GRAPH))
print("grupos:", len(groups), "| notas:", len(NOTES))
print("solapes de nodos:", overlap)
print("ERRORES de enlace:", len(errs), errs[:5])
print("->", os.path.basename(DST))
