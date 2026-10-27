"""Generate model-free PyTorch default CPU SDPA fixtures (short/64-query blocks)."""

from argparse import ArgumentParser
from pathlib import Path
import json
import torch

p = ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
if args.output.exists():
    raise ValueError("output must be new")
if not torch.__version__.startswith("2.7.1"):
    raise ValueError("expected pinned PyTorch 2.7.1")
torch.manual_seed(37)
cases = []
for length in [17, 193]:
    shape = (1, 2, length, 8)
    q, k, v = [torch.randn(shape) for _ in range(3)]
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    cases.append(
        dict(
            shape=shape,
            q=q.flatten().tolist(),
            k=k.flatten().tolist(),
            v=v.flatten().tolist(),
            out=out.flatten().tolist(),
        )
    )
args.output.write_text(
    json.dumps(
        dict(
            source="PyTorch 2.7.1 CPU F32 default scaled_dot_product_attention; seed 37",
            cases=cases,
        )
    )
)
