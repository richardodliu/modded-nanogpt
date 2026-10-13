"""Compare extracted Track-3 optimizers against source definitions, without executing trainers."""

import ast, importlib, json, math, sys
from pathlib import Path
import torch
from torch import Tensor, nn
import torch.nn.functional as F

B = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(B))
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-root", type=Path, required=True)
S = parser.parse_args().source_root


class SingleRank:
    def get_world_size(self):
        return 1

    def get_rank(self):
        return 0

    def all_gather(self, out, value):
        out[0].copy_(value)


count = 0
for row in json.loads(
    (B / "research/provenance/SOURCE.track3-aliases.json").read_text()
):
    tree = ast.parse((S / row["source"]).read_text())
    defs = {
        n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    selected = set(row["classes"])
    while True:
        nxt = selected | {
            n.id
            for name in selected
            for n in ast.walk(defs[name])
            if isinstance(n, ast.Name) and n.id in defs
        }
        if nxt == selected:
            break
        selected = nxt
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in selected
    ]
    for node in nodes:
        node.decorator_list = [
            d
            for d in node.decorator_list
            if not (isinstance(d, ast.Attribute) and d.attr == "compile")
        ]
    ns = dict(torch=torch, Tensor=Tensor, nn=nn, F=F, math=math, dist=SingleRank())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), row["source"], "exec"), ns)
    module = importlib.import_module(row["module"])
    for name in row["classes"]:
        torch.manual_seed(77)
        left = [nn.Parameter(torch.randn(*shape)) for shape in ((8, 12), (12, 8))]
        right = [nn.Parameter(p.detach().clone()) for p in left]
        if name == "NesterovOuterOptimizer":
            a = ns[name](left, sync_interval=2)
            b = getattr(module, name)(right, sync_interval=2)
            for _ in range(4):
                for x, y in zip(left, right):
                    delta = torch.randn_like(x) * 0.01
                    x.data.add_(delta)
                    y.data.add_(delta)
                a.maybe_outer_step()
                b.maybe_outer_step()
        else:
            a = ns[name](left)
            b = getattr(module, name)(right)
            for _ in range(4):
                for x, y in zip(left, right):
                    x.grad = torch.randn_like(x)
                    y.grad = x.grad.clone()
                a.step()
                b.step()
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        count += 1
print(
    f'PASS: {count} optimizer/source pairs over {len(json.loads((B/"research/provenance/SOURCE.track3-aliases.json").read_text()))} source files; exact CPU tensor equality after four steps.'
)
