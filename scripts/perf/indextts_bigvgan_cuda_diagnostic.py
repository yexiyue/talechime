"""Isolated pinned official BigVGAN component, not an IndexTTS baseline."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output = Path(sys.argv[2])
    config = request["request"]
    sys.path.insert(0, config["checkout"])
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import soundfile as sf
        import torch
        from indextts.s2mel.modules.bigvgan import bigvgan

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    clock, sequence = time.perf_counter(), 0

    def event(kind, payload):
        nonlocal sequence
        print(json.dumps(dict(schema="talechime/performance/1", attempt_id=request["attempt_id"],
                              sequence=sequence, offset_seconds=time.perf_counter()-clock,
                              kind=kind, payload=payload)), flush=True)
        sequence += 1

    started = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        model = bigvgan.BigVGAN.from_pretrained(config["model"], use_cuda_kernel=False)
        model = model.to("cuda:0")
        model.remove_weight_norm()
        model.eval()
    torch.cuda.synchronize()
    runtime = dict(kind="isolated_bigvgan_synthetic_mel_not_tts", torch=torch.__version__,
                   dtype=str(next(model.parameters()).dtype),
                   load_seconds=time.perf_counter()-started, custom_cuda_kernel=False,
                   cuda_allocated=torch.cuda.memory_allocated(),
                   cuda_reserved=torch.cuda.memory_reserved())
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    event("prepared", runtime)
    mel = torch.linspace(-8.0, 0.0, 80*128, device="cuda:0").reshape(1, 80, 128)
    for index in range(request["warmup_rounds"] + request["hot_rounds"]):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            wave = model(mel)
            pcm = np.asarray(wave.detach().float().cpu().numpy(), dtype="<f4").reshape(-1).copy()
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        assert pcm.size > 0 and np.isfinite(pcm).all()
        name = f"round-{index}-synthetic-mel-f32.wav"
        path = output / name
        sf.write(path, pcm, 22050, subtype="FLOAT")
        artifact = dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        bytes=path.stat().st_size, kind="audio_f32", derived_from=None)
        event("round", dict(index=index, warmup=index < request["warmup_rounds"], outcome="success",
                            synthesis_seconds=elapsed, first_pcm_seconds=elapsed,
                            audio_seconds=pcm.size/22050, delivery_boundary="segment_delivery_first_pcm",
                            output_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                            observations=[], artifacts=[artifact]))
    event("terminal", dict(outcome="success"))


if __name__ == "__main__":
    main()
