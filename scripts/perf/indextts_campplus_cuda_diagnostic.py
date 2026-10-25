"""Isolated official CAMPPlus F32 inference, not a TTS performance baseline."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    envelope = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output = Path(sys.argv[2])
    config = envelope["request"]
    sys.path.insert(0, config["checkout"])
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        import torchaudio
        from indextts.infer_v2_5 import IndexTTS2
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    clock = time.perf_counter()
    sequence = 0

    def event(kind, payload):
        nonlocal sequence
        print(json.dumps(dict(schema="talechime/performance/1", attempt_id=envelope["attempt_id"],
                              sequence=sequence, offset_seconds=time.perf_counter() - clock,
                              kind=kind, payload=payload)), flush=True)
        sequence += 1

    event("prepared", dict(kind="official_campplus_component_not_TTS", precision="F32", device="cuda:0"))
    state = torch.load(config["campplus_checkpoint"], map_location="cpu")
    with contextlib.redirect_stdout(sys.stderr):
        model = CAMPPlus(feat_dim=80, embedding_size=192)
        model.load_state_dict(state)
        model = model.to("cuda:0").eval()
    assert all(p.dtype == torch.float32 and p.device.type == "cuda" for p in model.parameters())
    actual = model.state_dict()
    assert set(actual) == set(state)
    for key, tensor in actual.items():
        expected = state[key].contiguous()
        value = tensor.detach().cpu().contiguous()
        assert value.shape == expected.shape and value.dtype == expected.dtype
        assert torch.equal(value.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)), key
    with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
        # This official method does not access self; retain its librosa defaults
        # and the original 15-second cap rather than rebuilding a shorter input.
        audio, rate = IndexTTS2._load_and_cut_audio(None, config["reference_audio"], 15)
        audio16 = torchaudio.transforms.Resample(rate, 16000)(audio)
        features = torchaudio.compliance.kaldi.fbank(
            audio16.to("cuda:0"), num_mel_bins=80, dither=0, sample_frequency=16000
        )
        features = features - features.mean(dim=0, keepdim=True)
        features = features.unsqueeze(0)
        np.save(output / "features.npy", features.cpu().numpy(), allow_pickle=False)
        timings = []
        embeddings = []
        for index in range(3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            embedding = model(features)
            torch.cuda.synchronize()
            timings.append(dict(index=index, warmup=index == 0, component_seconds=time.perf_counter() - started))
            assert embedding.shape == (1, 192) and torch.isfinite(embedding).all().item()
            embeddings.append(embedding.detach().cpu().contiguous())
        assert all(torch.equal(e.view(torch.uint8), embeddings[0].view(torch.uint8)) for e in embeddings)
        np.save(output / "embedding.npy", embeddings[0].numpy(), allow_pickle=False)
    result = dict(kind="isolated_official_campplus_not_TTS_baseline", torch=torch.__version__,
                  torchaudio=torchaudio.__version__, parameter_dtype="float32", all_state_tensors_exact=True,
                  state_tensors=len(state), input_shape=list(features.shape), reference_loader_rate=rate,
                  reference_resampled_samples=audio16.shape[-1], embedding_shape=[1, 192],
                  repeats_bitwise_equal=True, timings=timings,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cuda_allocated_bytes=torch.cuda.memory_allocated(), cuda_reserved_bytes=torch.cuda.memory_reserved(),
                  interpretation="component success does not prove full F32 model fits or establish old error cause")
    (output / "diagnostic.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    artifacts = []
    for name, kind in [("features.npy", "diagnostic_tensor"), ("embedding.npy", "diagnostic_tensor"),
                       ("diagnostic.json", "diagnostic_state")]:
        p = output / name
        artifacts.append(dict(path=name, kind=kind, sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                              bytes=p.stat().st_size, derived_from=None))
    event("observation", dict(diagnostic_result=dict(summary=result, artifacts=artifacts)))
    event("terminal", dict(outcome="success"))


if __name__ == "__main__":
    main()
