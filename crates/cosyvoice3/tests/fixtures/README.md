# CPU RNG oracle

`torch-cpu-rng.json` contains independent `torch.randn(n)` outputs from
PyTorch 2.7.1, macOS arm64 CPU, seed 0, float32. Sizes exercise scalar,
16-element vector, and overlapping tail-block paths. Regenerate explicitly:

```python
import json, torch
from pathlib import Path
sizes = (1, 7, 15, 16, 17, 31, 32, 33)
data = {str(n): torch.randn(n, generator=torch.Generator().manual_seed(0)).tolist()
        for n in sizes}
Path("crates/cosyvoice3/tests/fixtures/torch-cpu-rng.json").write_text(
    json.dumps(data, indent=2) + "\n")
```

These are synthetic numeric constants, with no model weights or user material.
The Rust tests do not need Python, network access, or downloaded models.

`torch-sine-source.json` contains eight F0 frame values and the official
CosyVoice `SineGen2` output at the first sample of each frame (480 samples
per frame, 24 kHz, eight harmonics, zero initial phase and zero noise).
It was generated with the pinned official checkout and PyTorch 2.7.1 CPU:

```python
import json, torch
from pathlib import Path
from cosyvoice.hifigan.generator import SineGen2
f0 = torch.tensor([123.4567, 1000.123, 5000.2345, 789.0123,
                   2345.6787, 110.1234, 3500.5, 700.004])
gen = SineGen2(24000, 480, harmonic_num=8, causal=False).eval()
gen.causal = True
# Construct only these tiny buffers; no 300-second noise allocation.
gen.rand_ini = torch.zeros(1, 9)
gen.sine_waves = torch.zeros(1, len(f0) * 480, 9)
waves = gen(f0.repeat_interleave(480).reshape(1, -1, 1))[0][:, ::480].squeeze(0)
Path("crates/cosyvoice3/tests/fixtures/torch-sine-source.json").write_text(
    json.dumps({"f0": f0.tolist(), "waves": waves.tolist()}, indent=2) + "\n")
```

The oracle catches scalar-division-to-reciprocal rewrites that change
harmonic phase; it does not duplicate the Rust cumsum implementation.
