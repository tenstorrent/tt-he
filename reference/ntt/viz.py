# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""MLIR -> computation-DAG web viewer (Dash + dash-cytoscape).

    python3 viz.py [file.mlir]      # then open http://localhost:8050

Parses the `ntt` MLIR our printer emits (print_mlir), draws each op as a square node
(opcode + output shape), and shows full op detail in a side panel on click. You can
also paste MLIR into the textarea and hit Render.

Scope: flat single-block SSA MLIR (what we emit, plus simple generic MLIR). Nested
regions/blocks are not handled.
"""
import re
import sys

import dash
from dash import dcc, html, Input, Output, State
import dash_cytoscape as cyto

cyto.load_extra_layouts()                     # enables the 'dagre' layered layout

# --------------------------------------------------------------------------- #
# MLIR parsing
# --------------------------------------------------------------------------- #
_TENSOR = re.compile(r"tensor<([0-9x]+)x[a-z]\d*>")     # tensor<32x8xi32> -> "32x8"
_ASSIGN = re.compile(r"^(?P<lhs>%[\w%,\s]*?)\s*=\s*(?P<rhs>.*)$")


def _shapes(typesig):
    return [m.group(1) for m in _TENSOR.finditer(typesig)]


def _category(opcode):
    if "all_to_all" in opcode:
        return "all_to_all"
    for key in ("matmul", "mul", "const", "split", "concat", "reshape",
                "slice", "transpose", "input"):
        if key in opcode:
            return key
    return "other"


def parse_mlir(text):
    """Return (nodes, edges). node = dict(id,label,opcode,category,results,operands,attrs,typesig)."""
    nodes, edges, producer = [], [], {}

    def add(node):
        nodes.append(node)
        for r in node["results"]:
            producer[r] = node["id"]

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("module", "}", "//")):
            continue

        # function signature -> input nodes (one per argument)
        if line.startswith("func.func"):
            args = re.search(r"\((.*?)\)", line)
            if args:
                for a in args.group(1).split(","):
                    a = a.strip()
                    if not a:
                        continue
                    name = a.split(":")[0].strip()
                    sh = _shapes(a)
                    add(dict(id=name, label=f"input\n{sh[0] if sh else ''}", opcode="input",
                             category="input", results=[name], operands=[], attrs="",
                             typesig=a.split(":", 1)[1].strip() if ":" in a else ""))
            continue

        if line.startswith("return"):
            m = re.search(r"return\s+(%\S+)", line)
            if m and m.group(1) in producer:
                for n in nodes:
                    if n["id"] == producer[m.group(1)]:
                        n["is_output"] = True
            continue

        m = _ASSIGN.match(line)
        if not m:
            continue
        results = [r.strip() for r in m.group("lhs").split(",") if r.strip()]
        rhs = m.group("rhs")
        opcode = rhs.split()[0].replace("ntt.", "").strip('"')
        attrs = ""
        am = re.search(r"\{(.*)\}", rhs)
        if am:
            attrs = am.group(1)
        typesig = rhs.split(":", 1)[1].strip() if ":" in rhs.split("{")[-1] else \
            (rhs.split(":", 1)[1].strip() if ":" in rhs else "")
        # operands = %tokens before the first '{' or ':' (opcode has no %, results are on the LHS)
        head = re.split(r"[{:]", rhs, 1)[0]
        operands = re.findall(r"%\w+", head)
        sh = _shapes(typesig)
        out_sh = sh[-1] if sh else ""
        node_id = results[0]
        label = f"{opcode}\n{out_sh}" if out_sh else opcode
        add(dict(id=node_id, label=label, opcode=opcode, category=_category(opcode),
                 results=results, operands=operands, attrs=attrs, typesig=typesig))
        for o in operands:
            if o in producer:
                edges.append((producer[o], node_id))
    return nodes, edges


def to_elements(nodes, edges):
    els = []
    for n in nodes:
        data = {k: n[k] for k in ("id", "label", "category")}
        data["detail"] = n
        cls = n["category"] + (" output" if n.get("is_output") else "")
        els.append({"data": data, "classes": cls})
    for s, t in edges:
        els.append({"data": {"source": s, "target": t}})
    return els


# --------------------------------------------------------------------------- #
# Dash app
# --------------------------------------------------------------------------- #
COLORS = {"input": "#90caf9", "const": "#cfd8dc", "matmul": "#ffb74d", "mul": "#aed581",
          "split": "#ce93d8", "concat": "#ce93d8", "all_to_all": "#ef5350", "reshape": "#80cbc4",
          "slice": "#80cbc4", "transpose": "#f48fb1", "other": "#e0e0e0"}

STYLESHEET = [
    {"selector": "node", "style": {
        "shape": "round-rectangle", "width": 64, "height": 64,
        "label": "data(label)", "text-wrap": "wrap", "text-valign": "center",
        "text-halign": "center", "font-size": 10, "border-width": 1, "border-color": "#555"}},
    *[{"selector": f'.{c}', "style": {"background-color": col}} for c, col in COLORS.items()],
    {"selector": ".output", "style": {"border-width": 4, "border-color": "#d32f2f"}},
    {"selector": "edge", "style": {
        "width": 2, "line-color": "#9e9e9e", "target-arrow-color": "#9e9e9e",
        "target-arrow-shape": "triangle", "curve-style": "bezier"}},
    {"selector": ":selected", "style": {"border-width": 4, "border-color": "#1565c0"}},
]


def _initial_text():
    if len(sys.argv) > 1:
        return open(sys.argv[1]).read()
    for f in ("example_ntt_tiled.mlir", "example_ntt_fold.mlir", "example_ntt_keep.mlir"):
        try:
            return open(f).read()
        except FileNotFoundError:
            continue
    return "// paste ntt MLIR here, then press Render\n"


app = dash.Dash(__name__)
_text0 = _initial_text()
_n0, _e0 = parse_mlir(_text0)

app.layout = html.Div(style={"display": "flex", "height": "100vh", "fontFamily": "monospace"}, children=[
    html.Div(style={"width": "300px", "padding": "8px", "borderRight": "1px solid #ccc",
                    "display": "flex", "flexDirection": "column"}, children=[
        html.H3("MLIR → DAG"),
        dcc.Textarea(id="src", value=_text0, style={"flex": 1, "fontFamily": "monospace", "fontSize": 11}),
        html.Button("Render", id="render", n_clicks=0, style={"marginTop": 6}),
    ]),
    cyto.Cytoscape(id="dag", elements=to_elements(_n0, _e0), stylesheet=STYLESHEET,
                   layout={"name": "dagre", "rankDir": "TB", "nodeSep": 25, "rankSep": 45},
                   style={"flex": 1, "height": "100%"}),
    html.Div(id="panel", style={"width": "320px", "padding": "12px", "borderLeft": "1px solid #ccc",
                                 "overflowY": "auto", "fontSize": 12},
             children=html.Em("click a node for details")),
])


@app.callback(Output("dag", "elements"), Input("render", "n_clicks"), State("src", "value"))
def _render(_, text):
    return to_elements(*parse_mlir(text or ""))


@app.callback(Output("panel", "children"), Input("dag", "tapNodeData"))
def _detail(data):
    if not data:
        return html.Em("click a node for details")
    d = data["detail"]
    def row(k, v):
        return html.Tr([html.Td(html.B(k), style={"verticalAlign": "top", "paddingRight": 8}),
                        html.Td(str(v))])
    rows = [row("opcode", d["opcode"]), row("category", d["category"]),
            row("results", ", ".join(d["results"])),
            row("operands", ", ".join(d["operands"]) or "—"),
            row("attrs", d["attrs"] or "—"), row("type", d["typesig"] or "—")]
    if d.get("is_output"):
        rows.append(row("note", "program output"))
    return [html.H4(d["id"]), html.Table(rows)]


if __name__ == "__main__":
    print("serving DAG at http://localhost:8050  (Ctrl-C to stop)")
    app.run(host="127.0.0.1", port=8050, debug=False)
