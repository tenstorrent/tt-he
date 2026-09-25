# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""viz_bridge.py — render the NEW mapping through the EXISTING viz.py tool.

`viz.py` is the visualisation tool: an interactive canvas with a labelled DAG, a side panel, search, a
level tab per IR stage, a Gantt with a sub-tab per core count, and a legend. It builds its own workload
from the PARAMETERS block at its top. This module feeds it a program and a mapping that were produced
elsewhere -- by the four optimisation passes -- so the passes get the real tool instead of a new,
worse one.

Nothing about viz.py changes: this assembles exactly the data dict `viz.build()` returns and reuses
`viz._HTML` verbatim. What is added is per-op ENGINE (the passes' central decision, which viz.py's
category colouring cannot know about) and the pass ladder, both carried in `meta` and shown in the
side panel and header.
"""
from __future__ import annotations
import json
import os

import _paths                                     # noqa: F401
import passes as P
import viz                                        # its build() is behind a __main__ guard
from ir import outputs
from mlir_viz import layout, program_graph
from sched_viz import catcol

ENGINE_COLOR = {"sfpu": "#7C68FA", "fpu": "#00C2A8", "dma": "#F2A93B", "free": "#8b949e"}


def _gantt(prog, mp, est, colour_by_engine=True):
    """viz.py's gantt shape: {"nodes": [...], "sched": {str(P): {...}}} from OUR estimate."""
    prod_of = {}
    for i, op in enumerate(prog.ops):
        for v in outputs(op):
            prod_of[v.name] = i
    nodes = []
    for i, op in enumerate(prog.ops):
        outs = outputs(op)
        cat, col = catcol(op.opcode)
        eng = mp.engine[i]
        nodes.append({"i": i, "o": op.opcode,
                      "c": (eng if colour_by_engine else cat),
                      "col": (ENGINE_COLOR.get(eng, col) if colour_by_engine else col),
                      "r": (outs[0].name if outs else ""),
                      "in": [prod_of[o.name] for o in op.operands if o.name in prod_of]})
    intier = [[("noc" if mp.tier.get((d, i)) == "noc" else
                "dram" if mp.tier.get((d, i)) == "dram" else "sram")
               for d in nodes[i]["in"]] for i in range(len(nodes))]
    n_edges = sum(len(x) for x in intier)
    fused = sum(1 for x in intier for t in x if t == "sram")
    sched = {str(est["P"]): {
        "core": est["core"],
        "start": [round(x, 2) for x in est["start"]],
        "end": [round(x, 2) for x in est["end"]],
        "intier": intier,
        "makespan": round(est["makespan"], 1),
        "used": est["cores_used"],
        "reuse": 100 * sum(1 for r in mp.resident if r) // max(1, len(mp.resident)),
        "fusedB": 100 * fused // max(1, n_edges),
    }}
    return {"nodes": nodes, "sched": sched}


def write(path, levels, prog, mp, est, rungs, meta_extra=None, max_nodes=60000):
    """levels = [(name, Program), ...] for the DAG tabs; `prog` is the mapped one (L2) for the Gantt."""
    dag, skipped = {}, []
    for name, pr in levels:
        if len(pr.ops) > max_nodes:
            skipped.append(f"{name}({len(pr.ops)})")
            continue
        dag[name] = {"nodes": layout(program_graph(pr))}
    gantt = _gantt(prog, mp, est)

    lad = [{"name": r.get("pass_name", "?"),
            "ms": round(r.get("makespan", 0) / 1e3, 4),
            "groups": r.get("groups", 0),
            "note": r.get("note", "")} for r in rungs if not r.get("skipped")]
    hist = {}
    for i in range(len(prog.ops)):
        k = f"{P.op_class(prog.ops[i])}/{mp.engine[i]}"
        hist[k] = hist.get(k, 0) + 1

    meta = dict(ntt="", tile=32, fusionA=[], reuse=True, correct="n/a",
                l1=None, l2=len(prog.ops), l3=None, l3a=None,
                ladder=lad, engines=hist, skipped=skipped,
                dispatch_groups=est["groups"], cores=est["cores_used"], P=est["P"])
    meta.update(meta_extra or {})
    data = {"dag": dag, "gantt": gantt, "meta": meta}
    html = viz._HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    html = _augment(html)
    with open(path, "w") as f:
        f.write(html)
    return os.path.getsize(path)


def _augment(html):
    """Add what the passes need on top of viz.py's page: the source program, the pass ladder, and an
    engine legend. Injected rather than forked so viz.py stays the single owner of the view."""
    css = """
 #prog{position:fixed;left:10px;top:52px;max-width:330px;background:#161b22ee;border:1px solid #30363d;
   border-radius:8px;padding:8px 10px;font-size:11px;z-index:9;white-space:pre-wrap;color:#b9c2d0;
   max-height:150px;overflow:auto}
 #prog .hd{color:#7d8998;text-transform:uppercase;font-size:10px;margin-bottom:4px}
 #prog .k{color:#58a6ff;font-weight:700} #prog .c{color:#5d6672}
 #lad{position:fixed;right:332px;top:52px;background:#161b22ee;border:1px solid #30363d;border-radius:8px;
   padding:8px 10px;font-size:11px;z-index:9;color:#d7dde5}
 #lad table{border-collapse:collapse} #lad td{padding:1px 7px;text-align:right}
 #lad td.l{text-align:left;color:#9aa7b4} #lad .d{color:#3fb950}
 #lad .hd{color:#7d8998;text-transform:uppercase;font-size:10px;margin-bottom:4px}
"""
    js = r"""
(function(){
  const M = DATA.meta;
  if(M.program){
    const d=document.createElement("div"); d.id="prog";
    d.innerHTML='<div class="hd">program</div>'+M.program.split("\n").map(l=>
      l.trim().startsWith("#")?'<span class="c">'+l+'</span>'
      :l.replace(/\b(mul|add)\b/g,'<span class="k">$1</span>')).join("\n");
    document.body.appendChild(d);
  }
  if(M.ladder && M.ladder.length){
    const d=document.createElement("div"); d.id="lad";
    let h='<div class="hd">optimisation passes</div><table>';
    const first=M.ladder[0].ms, last=M.ladder[M.ladder.length-1].ms;
    for(const r of M.ladder)
      h+=`<tr><td class="l">${r.name}</td><td>${r.ms.toFixed(3)} ms</td>`
        +`<td class="l" style="color:#7d8998">${r.groups||''}</td></tr>`;
    h+=`<tr><td class="l" style="color:#eaf0f7">end to end</td><td class="d">`
      +`${(first/last).toFixed(2)}&times;</td><td></td></tr></table>`;
    d.innerHTML=h; document.body.appendChild(d);
  }
})();
"""
    html = html.replace("#ptabs{display:none;gap:4px}", css + " #ptabs{display:none;gap:4px}")
    return html.replace("</script></body></html>", js + "\n</script></body></html>")
