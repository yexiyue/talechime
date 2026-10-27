"""Generate model-free independent PyTorch CPU normalization fixtures."""

from argparse import ArgumentParser
from pathlib import Path
import json
import torch

p = ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
if args.output.exists():
    raise ValueError("output must be a new file")
if not torch.__version__.startswith("2.7.1"):
    raise ValueError("expected pinned PyTorch 2.7.1")
torch.manual_seed(29)
cases = []
for width in [17, 128, 512, 768]:
    x = torch.randn(1, width)
    normalized, mean, rstd = torch.native_layer_norm(x, (width,), None, None, 1e-5)
    weight = torch.linspace(0.2, 1.7, width)
    bias = torch.linspace(-0.3, 0.5, width)
    affine = torch.native_layer_norm(x, (width,), weight, bias, 1e-5)[0]
    instance, im, ir = torch.native_batch_norm(
        x.reshape(1, 1, width),
        torch.tensor([0.7]),
        torch.tensor([-0.3]),
        None,
        None,
        True,
        0.0,
        1e-5,
    )
    cases.append(
        dict(
            input=x.flatten().tolist(),
            mean=mean.item(),
            rstd=rstd.item(),
            normalized=normalized.flatten().tolist(),
            weight=weight.tolist(),
            bias=bias.tolist(),
            affine=affine.flatten().tolist(),
            instance=instance.flatten().tolist(),
            instance_mean=im.item(),
            instance_rstd=ir.item(),
        )
    )
args.output.write_text(
    json.dumps(
        dict(source="PyTorch 2.7.1 CPU F32, native_layer_norm; seed 29", cases=cases)
    )
)
