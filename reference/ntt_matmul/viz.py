# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Hierarchical multi-level NTT decomposition viewer (Dash + dash-cytoscape).

    python3 viz.py                 # uses main.py's PLAN/TILE_NTT/TILE_MM
    python3 main.py                # builds + launches this viewer

Instead of unfolding the whole graph (unreadable), this shows the NTT at progressive
decomposition levels and lets you drill into ONE block at a time by clicking it:

  L1  raw NTT          (no tile, no int8): full-size matmul_modq / mul_modq
  L2  NTT-tile         tile_matmul(TILE_NTT) + all_to_all corner turns   (if it changes the block)
  L3  matmul/hardware  tile_matmul(TILE_MM): block -> 32x32 tiles, K accumulates
  L4  int8             lower_matmul_int8: decompose / i8 matmul / add / recombine

Click an expandable block -> it becomes a box containing its next real decomposition
(levels that don't change the block are skipped automatically). Click again to collapse.
Provenance is carried by each pass via the `prov` op attribute (the source op's name).

Scope: the programs produced by planner/passes here; built in-process (not parsed from MLIR).
"""
import sys

import dash
from dash import dcc, html, Input, Output, State
import dash_cytoscape as cyto

import ir
import passes
import planner
from ir import outputs

cyto.load_extra_layouts()


# --------------------------------------------------------------------------- #
# Build the level programs and the provenance forest
# --------------------------------------------------------------------------- #
def build_levels(plan, q, tile_ntt, tile_mm, radix=128, nlimbs=4):
    # L1: untiled raw NTT (clean -- one matmul per stage, reshape corner turns). L2: the proper
    # tiled pipeline (forward_program(tile=TILE_NTT) keeps data tile-granular across stages, so the
    # corner turn becomes an explicit all_to_all). L2->L3->L4 are pass transforms (prov chains);
    # L1->L2 is a regeneration, linked numerically in _link_levels (a tiled op is a sub-block of
    # its untiled parent's output).
    L1 = planner.forward_program(plan, q, transpose="fold")                       # untiled raw
    L2b = planner.forward_program(plan, q, transpose="fold", tile=tile_ntt)
    L2 = passes.tilize_corner_turns(passes.tile_matmul(L2b, tile_ntt), tile_ntt)  # tiled, all_to_all @ TILE_NTT
    if tile_mm < tile_ntt:
        # re-tilize the corner turn at TILE_MM: the MM tiles communicate directly (the coarse
        # all_to_all + the matmul concat-up collapse into one TILE_MM-granular all_to_all).
        L3 = passes.tilize_corner_turns(passes.tile_matmul(L2, tile_mm, prefix="%h"), tile_mm)
    else:
        L3 = L2
    L4 = passes.lower_matmul_int8(L3, radix=radix, nlimbs=nlimbs)
    return [L1, L2, L3, L4]


def _is_block(A, B):
    """True if array B equals A, or B equals a tile-block of the larger array A."""
    import numpy as _np
    A = _np.asarray(A); B = _np.asarray(B)
    if A.shape == B.shape:
        return _np.array_equal(A, B)
    if A.ndim != 2 or B.ndim != 2:
        return False
    ar, ac = A.shape; br, bc = B.shape
    if br > ar or bc > ac or ar % br or ac % bc:
        return False
    for i in range(0, ar, br):
        for j in range(0, ac, bc):
            if _np.array_equal(A[i:i + br, j:j + bc], B):
                return True
    return False


def _link_levels(L1, L2, q):
    """Map each L2 op (tiled) to its L1 op (untiled) parent, by matching output values: a tiled
    op's output is a sub-block of the untiled op it derives from. Returns {l2_node_id: l1_node_id}."""
    import numpy as _np
    n = int(_np.prod(L1.inputs[0].shape))
    x = _np.random.default_rng(1).integers(0, q, size=n, dtype=_np.int64)
    _, e1 = ir.run(L1, x, return_env=True)
    _, e2 = ir.run(L2, x, return_env=True)
    l1 = [(_node_id(o), o.opcode.value, e1.get(_node_id(o))) for o in L1.ops]
    l1oc = {l1id: oc for (l1id, oc, _) in l1}

    def fam(a, b):                                   # same opcode family?
        return a == b or ("matmul" in a and "matmul" in b) or \
               ("all_to_all" in b and a in ("reshape", "split", "concat", "transpose", "all_to_all"))

    xlink = {}
    for o in L2.ops:
        nid, oc, B = _node_id(o), o.opcode.value, e2.get(_node_id(o))
        if B is None:
            continue
        cands = [l1id for (l1id, _, A) in l1 if A is not None and _is_block(A, B)]
        if not cands:
            continue
        same = [c for c in cands if fam(l1oc[c], oc)]
        xlink[nid] = (same or cands)[0]
    return xlink


def _node_id(op):
    return outputs(op)[0].name


def _category(opcode):
    if "all_to_all" in opcode:
        return "all_to_all"
    for key in ("matmul", "recombine", "decompose", "add_modq", "mul", "const",
                "split", "concat", "reshape", "slice", "transpose"):
        if key in opcode:
            return "add" if key == "add_modq" else key
    return "other"


class Model:
    """Per-level lookup tables + the provenance forest (children by `prov`)."""
    def __init__(self, levels, xlink=None):
        self.levels = levels
        self.xlink = xlink or {}                     # L2 node_id -> L1 node_id (numeric link)
        self.LAST = len(levels)                     # 1-based deepest level
        self.ops = {}          # level -> {node_id: op}
        self.val2node = {}     # level -> {result_name: node_id}
        self.order = {}        # level -> {node_id: position}
        self.childmap = {}     # level -> {prov_name: [node_ids at level+1]}
        self.inputs = [v.name for v in levels[0].inputs]
        for li, prog in enumerate(levels, start=1):
            self.ops[li], self.val2node[li], self.order[li] = {}, {}, {}
            for pos, op in enumerate(prog.ops):
                nid = _node_id(op)
                self.ops[li][nid] = op
                self.order[li][nid] = pos
                for v in outputs(op):
                    self.val2node[li][v.name] = nid
        for li in range(1, self.LAST):
            cm = {}
            for op in levels[li].ops:               # ops of level li+1 (0-based li)
                cm.setdefault(self.prov(li + 1, op), []).append(_node_id(op))
            self.childmap[li] = cm
        # forest roots: all L1 ops, plus "orphan" ops introduced at a deeper level whose
        # provenance parent doesn't exist one level up (e.g. the cross-stage all_to_all,
        # created by tilize_corner_turns) -- shown as top-level stage nodes.
        self.roots = []
        for li in range(1, self.LAST + 1):
            for op in levels[li - 1].ops:
                nm = _node_id(op)
                if li == 1 or self.prov(li, op) not in self.ops.get(li - 1, {}):
                    self.roots.append((li, nm))

    def prov(self, level, op):
        """Effective parent name at level-1. For L2 (level 2) the parent is the numeric link to
        the untiled L1. Otherwise the `prov` attr (a stale prov not naming a node one level up is
        treated as self -- the op passed through unchanged)."""
        nm = _node_id(op)
        if level == 2:
            return self.xlink.get(nm, nm)
        p = op.attrs.get("prov", nm)
        if level <= 1 or p not in self.ops.get(level - 1, {}):
            return nm
        return p

    def children(self, level, name):
        """Next real decomposition of (level,name): skip levels that don't change it."""
        if level >= self.LAST:
            return []
        ch = self.childmap.get(level, {}).get(name, [])
        if ch == [name] and name in self.ops.get(level + 1, {}):     # pure pass-through
            return self.children(level + 1, name)
        return [(level + 1, c) for c in ch]

    def op(self, level, name):
        return self.ops[level].get(name)


# --------------------------------------------------------------------------- #
# device kernel that implements each IR op (see kernels/)
# --------------------------------------------------------------------------- #
_KERNEL = {
    "decompose": "compute/decompose_limbs.cpp",
    "add_modq": "compute/modadd_reduce.cpp",
    "recombine_modq": "compute/recombine_bigK.cpp",
    "mul_modq": "compute/modmul_compute.cpp",
    "all_to_all_tiles": "dataflow/all_to_all_corner_mc.cpp",
    "all_to_all": "dataflow/all_to_all_corner_mc.cpp",
    "reshape": "(layout — reader/writer addressing)",
    "slice": "(layout — reader/writer addressing)",
    "concat": "(layout — reader/writer addressing)",
    "split": "(layout — reader/writer addressing)",
    "transpose": "(layout — tile transpose)",
    "const": "(baked DRAM constant)",
    "input": "(DRAM input)",
}


def _kernel_of(op):
    oc = op.opcode.value
    if oc == "matmul_modq":
        i8 = bool(op.operands) and op.operands[0].dtype == "i8"
        return "compute/matmul_compute_mtk.cpp" if i8 else "— (modular matmul; lowers to int8)"
    return _KERNEL.get(oc, "—")


def _fmt_val(v):
    return f"{v.name}: {'x'.join(map(str, v.shape)) or 'scalar'} {v.dtype}"


# --------------------------------------------------------------------------- #
# Elements for a given expand state
# --------------------------------------------------------------------------- #
def build_elements(model, expanded):
    els, rendered, leaves = [], set(), []

    def label(level, name):
        op = model.op(level, name)
        oc = op.opcode.value.replace("ntt.", "")
        sh = outputs(op)[0].shape
        return f"{oc}\n{'x'.join(map(str, sh))}"

    def render(level, name, parent_gid):
        gid = f"{level}:{name}"
        ch = model.children(level, name)
        expandable = bool(ch)
        is_exp = gid in expanded and expandable
        op = model.op(level, name)
        cat = _category(op.opcode.value)
        cls = cat + (" expandable" if expandable else "") + (" expanded" if is_exp else "")
        data = {"id": gid, "label": label(level, name), "level": level,
                "expandable": expandable, "detail": {
                    "id": name, "level": f"L{level}", "opcode": op.opcode.value,
                    "kernel": _kernel_of(op),
                    "inputs": [_fmt_val(o) for o in op.operands] or ["—"],
                    "outputs": [_fmt_val(v) for v in outputs(op)],
                    "attrs": ", ".join(f"{k}={v}" for k, v in op.attrs.items() if k != "prov") or "—",
                    "children": len(ch)}}
        if parent_gid:
            data["parent"] = parent_gid
        els.append({"data": data, "classes": cls})
        rendered.add(gid)
        if is_exp:
            for cl, cn in ch:
                render(cl, cn, gid)
        else:
            leaves.append((level, name, gid))

    # input pseudo-nodes (always at L1)
    for name in model.inputs:
        gid = f"in:{name}"
        ish = next((v for v in model.levels[0].inputs if v.name == name), None)
        idet = _fmt_val(ish) if ish is not None else name
        els.append({"data": {"id": gid, "label": f"input\n{name}", "level": 1,
                             "expandable": False, "detail": {"id": name, "level": "input",
                             "opcode": "input", "kernel": _KERNEL["input"], "inputs": ["—"],
                             "outputs": [idet], "attrs": "—", "children": 0}}, "classes": "input"})
        rendered.add(gid)

    for level, name in model.roots:                 # L1 ops + orphan stage ops (all_to_all)
        render(level, name, None)

    def vis_rep(level, valname):
        if valname in model.inputs:
            return f"in:{valname}"
        nid = model.val2node[level].get(valname)
        if nid is None:
            return None
        cl, cn = level, nid
        while f"{cl}:{cn}" not in rendered:
            op = model.op(cl, cn)
            if op is None or cl <= 1:
                return None
            cn = model.prov(cl, op)
            cl -= 1
        return f"{cl}:{cn}"

    seen = set()
    for level, name, gid in leaves:
        for operand in model.op(level, name).operands:
            rep = vis_rep(level, operand.name)
            if rep and rep != gid and (rep, gid) not in seen:
                seen.add((rep, gid))
                els.append({"data": {"source": rep, "target": gid}})
    return els


# --------------------------------------------------------------------------- #
# Dash app
# --------------------------------------------------------------------------- #
COLORS = {"input": "#90caf9", "const": "#cfd8dc", "matmul": "#ffb74d", "mul": "#aed581",
          "split": "#ce93d8", "concat": "#ce93d8", "all_to_all": "#ef5350", "reshape": "#80cbc4",
          "slice": "#80cbc4", "transpose": "#f48fb1", "other": "#e0e0e0",
          "decompose": "#9fa8da", "recombine": "#f06292", "add": "#dce775"}
LEVEL_TINT = {1: "#ffffff", 2: "#f6fbff", 3: "#f3f7ff", 4: "#f0f4ff"}

STYLESHEET = [
    {"selector": "node", "style": {
        "shape": "round-rectangle", "width": 70, "height": 56,
        "label": "data(label)", "text-wrap": "wrap", "text-valign": "center",
        "text-halign": "center", "font-size": 9, "border-width": 1, "border-color": "#555"}},
    *[{"selector": f'.{c}', "style": {"background-color": col}} for c, col in COLORS.items()],
    # compound (expanded) nodes: light box, label on top
    {"selector": ":parent", "style": {"background-opacity": 0.10, "background-color": "#1565c0",
                                      "text-valign": "top", "font-size": 11, "font-weight": "bold",
                                      "border-width": 2, "border-color": "#1565c0", "padding": 12}},
    {"selector": ".expandable", "style": {"border-width": 3, "border-style": "double",
                                          "border-color": "#1565c0"}},
    {"selector": "edge", "style": {
        "width": 1.5, "line-color": "#9e9e9e", "target-arrow-color": "#9e9e9e",
        "target-arrow-shape": "triangle", "curve-style": "bezier"}},
    {"selector": ":selected", "style": {"border-width": 4, "border-color": "#d32f2f"}},
]


def serve(plan, q, tile_ntt, tile_mm, radix=128, nlimbs=4, port=8050):
    levels = build_levels(plan, q, tile_ntt, tile_mm, radix, nlimbs)
    model = Model(levels, _link_levels(levels[0], levels[1], q))
    app = dash.Dash(__name__)
    BTN = {"marginRight": 6, "marginTop": 6, "padding": "6px 10px", "fontFamily": "monospace"}
    app.layout = html.Div(style={"display": "flex", "height": "100vh", "fontFamily": "monospace"}, children=[
        html.Div(style={"width": "260px", "padding": "10px", "borderRight": "1px solid #ccc"}, children=[
            html.H3("NTT decomposition"),
            html.Div(f"plan = {plan}", style={"fontSize": 12}),
            html.Div(f"TILE_NTT = {tile_ntt}   TILE_MM = {tile_mm}", style={"fontSize": 12}),
            html.Hr(),
            html.Div("Click a block to inspect it; use the Expand / Collapse buttons in the panel "
                     "on the right to drill into (or fold) that block.", style={"fontSize": 12, "color": "#555"}),
            html.Ul([html.Li("L1 raw  L2 NTT-tile  L3 hw-tile  L4 int8")], style={"fontSize": 11}),
            html.Button("Collapse all", id="reset", n_clicks=0, style=BTN),
            html.Hr(),
            html.Div([html.B("Mouse:"), html.Br(),
                      "left-drag = box-select   ", html.Br(),
                      "left-drag a node = move (selected move together)", html.Br(),
                      "right-drag = pan   •   scroll = zoom"], style={"fontSize": 11, "color": "#555"}),
            html.Button("Re-layout", id="relayout", n_clicks=0, style=BTN),
            html.Div(id="rmb-dummy", style={"display": "none"}),
        ]),
        cyto.Cytoscape(id="dag", elements=build_elements(model, set()), stylesheet=STYLESHEET,
                       layout={"name": "dagre", "rankDir": "TB", "nodeSep": 18, "rankSep": 40},
                       boxSelectionEnabled=True, userPanningEnabled=False, userZoomingEnabled=True,
                       autoungrabify=False, style={"flex": 1, "height": "100%"}),
        html.Div(style={"width": "320px", "padding": "12px", "borderLeft": "1px solid #ccc",
                        "overflowY": "auto", "fontSize": 12}, children=[
            html.Div([
                html.Button("▶ Expand", id="btn-expand", n_clicks=0, style=BTN),
                html.Button("▼ Collapse", id="btn-collapse", n_clicks=0, style=BTN),
            ]),
            html.Div(id="panel", children=html.Em("click a node for details"), style={"marginTop": 8}),
        ]),
        dcc.Store(id="selected"), dcc.Store(id="expanded", data=[]),
    ])
    app._model = model

    @app.callback(Output("selected", "data"), Output("panel", "children"),
                  Input("dag", "tapNodeData"), State("expanded", "data"))
    def _select(tap, expanded):
        if not tap or "detail" not in tap:
            return None, html.Em("click a node for details")
        d = tap["detail"]
        is_exp = tap["id"] in set(expanded or [])
        state = ("expanded" if is_exp else "collapsed") if tap.get("expandable") else "leaf (atomic)"

        def io_block(items):
            return html.Td(html.Div([html.Div(s, style={"whiteSpace": "nowrap"}) for s in items]))

        rows = [
            html.Tr([html.Td(html.B("op"), style={"verticalAlign": "top", "paddingRight": 8}),
                     html.Td(d["opcode"])]),
            html.Tr([html.Td(html.B("kernel"), style={"verticalAlign": "top", "paddingRight": 8}),
                     html.Td(html.Code(d["kernel"]))]),
            html.Tr([html.Td(html.B("level"), style={"verticalAlign": "top", "paddingRight": 8}),
                     html.Td(d["level"])]),
            html.Tr([html.Td(html.B("inputs"), style={"verticalAlign": "top", "paddingRight": 8}),
                     io_block(d["inputs"])]),
            html.Tr([html.Td(html.B("outputs"), style={"verticalAlign": "top", "paddingRight": 8}),
                     io_block(d["outputs"])]),
            html.Tr([html.Td(html.B("attrs"), style={"verticalAlign": "top", "paddingRight": 8}),
                     html.Td(d["attrs"])]),
            html.Tr([html.Td(html.B("expands to"), style={"verticalAlign": "top", "paddingRight": 8}),
                     html.Td(f"{d['children']} ops" if d["children"] else "— (leaf)")]),
            html.Tr([html.Td(html.B("state"), style={"verticalAlign": "top", "paddingRight": 8}),
                     html.Td(state)]),
        ]
        return tap, [html.H4(d["id"]), html.Table(rows, style={"borderSpacing": "0 3px"}),
                     html.Div("Use Expand / Collapse above." if tap.get("expandable")
                              else "This block has no further decomposition.",
                              style={"color": "#777", "marginTop": 8})]

    @app.callback(Output("expanded", "data"),
                  Input("btn-expand", "n_clicks"), Input("btn-collapse", "n_clicks"),
                  Input("reset", "n_clicks"),
                  State("selected", "data"), State("expanded", "data"))
    def _fold(_e, _c, _r, selected, expanded):
        trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else ""
        exp = set(expanded or [])
        if trig == "reset":
            return []
        if selected and selected.get("expandable"):
            gid = selected["id"]
            if trig == "btn-expand":
                exp.add(gid)
            elif trig == "btn-collapse":
                exp.discard(gid)
        return sorted(exp)

    @app.callback(Output("dag", "elements"), Input("expanded", "data"))
    def _draw(expanded):
        return build_elements(model, set(expanded or []))

    @app.callback(Output("dag", "layout"), Input("relayout", "n_clicks"))
    def _relayout(n):
        return {"name": "dagre", "rankDir": "TB", "nodeSep": 18, "rankSep": 40, "nonce": n}

    # Right-drag = pan (cytoscape has no native right-button pan). A clientside hook grabs the
    # cytoscape instance (container._cyreg.cy) and pans on right-button drag; left-drag stays
    # box-select (userPanningEnabled=False), scroll stays zoom.
    app.clientside_callback(
        """
        function(_id) {
            var tries = 0;
            var iv = setInterval(function () {
                tries++;
                var el = document.getElementById('dag');
                var cy = el && el._cyreg && el._cyreg.cy;
                if (!cy) { if (tries > 100) clearInterval(iv); return; }
                clearInterval(iv);
                if (cy.__rmbPan) return;  cy.__rmbPan = true;
                var c = cy.container(), panning = false, last = null;
                c.addEventListener('contextmenu', function (e) { e.preventDefault(); });
                c.addEventListener('mousedown', function (e) {
                    if (e.button === 2) { panning = true; last = {x: e.clientX, y: e.clientY}; e.preventDefault(); }
                });
                window.addEventListener('mousemove', function (e) {
                    if (panning && last) { cy.panBy({x: e.clientX - last.x, y: e.clientY - last.y});
                                           last = {x: e.clientX, y: e.clientY}; }
                });
                window.addEventListener('mouseup', function (e) { if (e.button === 2) { panning = false; last = null; } });
                // scroll = zoom, centered at the cursor (own handler; native wheel zoom is flaky here)
                cy.userZoomingEnabled(false);
                c.addEventListener('wheel', function (e) {
                    e.preventDefault();
                    var r = c.getBoundingClientRect();
                    var rp = {x: e.clientX - r.left, y: e.clientY - r.top};
                    var lvl = cy.zoom() * Math.exp(-e.deltaY * 0.0015);
                    cy.zoom({level: lvl, renderedPosition: rp});
                }, {passive: false});
            }, 100);
            return '';
        }
        """,
        Output("rmb-dummy", "children"), Input("dag", "id"))

    print(f"serving hierarchical NTT viewer at http://localhost:{port}  (Ctrl-C to stop)")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    serve((2**7, 2**8), 167772161, 2**10, 32, port=int(sys.argv[1]) if len(sys.argv) > 1 else 8050)
