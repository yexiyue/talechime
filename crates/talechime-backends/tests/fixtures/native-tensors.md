# Synthetic tensor archive

`native-tensors.pt` is a 16-value synthetic CPU tensor test fixture generated with
PyTorch 2.7.1 (`torch.arange(16, dtype=torch.float32)`), plus two F64 values.
It contains no model weights or user data. The nested `model` mapping contains
an offset view (`base[4:8]`), a transposed 4x4 view, and `nested.double=[0.5,2]`.
SHA-256: `01b921749b21e8f10141d3f0d066b118e13bd47cb2ad9c8c681e6802fd290e7e`.
Tests read this fixed input without Python or network access.
