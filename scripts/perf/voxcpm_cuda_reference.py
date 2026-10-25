"""Pinned official streaming inference; Rust xtask owns repetition and processes."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output = Path(sys.argv[2]).resolve()
    config = request["request"]
    if config.get("loading_runtime"):
        sys.path.insert(0, str(Path(config["loading_runtime"])))
    sys.path.insert(0, str(Path(config["checkout"]) / "src"))
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import soundfile as sf
        import torch
        import transformers
        from voxcpm import VoxCPM

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
        if config.get("loading_control") == "meta_stream":
            from voxcpm_lowpeak_loading import load
            model = load(config["model"], output)
        else:
            assert not config.get("loading_control"), "unknown loading control"
            owner = VoxCPM(config["model"], enable_denoiser=False,
                           zipenhancer_model_path=None, optimize=False, device="cuda")
            model = owner.tts_model
    model.eval()
    torch.cuda.synchronize()
    frame_limit = config.get("frame_limit", 200)
    assert isinstance(frame_limit, int) and 1 <= frame_limit <= 2000
    runtime = dict(torch=torch.__version__, transformers=transformers.__version__,
                   load_seconds=time.perf_counter()-started, optimize=False,
                   source=config["checkout"], model_dtype=str(next(model.base_lm.parameters()).dtype),
                   codec_dtype=str(next(model.audio_vae.parameters()).dtype),
                   reference_boundary="prepared once before warmup; cached hot requests" if config.get("use_reference", True) else "none; zero-shot narrator",
                   api="official _generate_with_prompt_cache streaming; no model math changes",
                   loading_control=config.get("loading_control"),
                   frame_limit=frame_limit, text_length_limit_override=100000.0)
    started = time.perf_counter()
    prompt = None
    if config.get("use_reference", True):
        voice = json.loads(Path(config["voice_metadata"]).read_text(encoding="utf-8"))
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            prompt = model.build_prompt_cache(prompt_text=voice["transcript"],
                                             prompt_wav_path=config["reference_wav"])
    torch.cuda.synchronize()
    runtime["reference_prepare_seconds"] = time.perf_counter()-started if prompt is not None else 0.0
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    event("prepared", runtime)
    last_stop = []

    def capture_stop(module, args, result):
        last_stop[:] = [result.detach()]

    model.stop_head.register_forward_hook(capture_stop)
    text = Path(config["text_file"]).read_text(encoding="utf-8").strip()
    rate = model.sample_rate
    assert rate == 48000
    for index in range(request["warmup_rounds"] + request["hot_rounds"]):
        chunks, first, samples = [], None, 0
        last_stop.clear()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            generator = model._generate_with_prompt_cache(
                text, prompt, max_len=frame_limit, inference_timesteps=10, cfg_value=2.0,
                retry_badcase=False, retry_badcase_ratio_threshold=100000.0,
                streaming=True, seed=config["seed"])
            try:
                for wav, _, _ in generator:
                    chunk = np.asarray(wav.detach().float().cpu().numpy(), dtype="<f4").reshape(-1).copy()
                    if chunk.size:
                        if first is None:
                            first = time.perf_counter()-started
                        chunks.append(chunk)
                        samples += chunk.size
            finally:
                generator.close()
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        assert last_stop and last_stop[0].argmax(dim=-1)[0].item() == 1, "no confirmed model stop"
        pcm = np.concatenate(chunks)
        assert pcm.size == samples and samples > 0 and np.isfinite(pcm).all() and np.any(pcm)
        name = f"round-{index}-f32.wav"
        path = output / name
        sf.write(path, pcm, rate, subtype="FLOAT")
        assert sf.info(path).frames == samples
        (output / f"round-{index}-stream.json").write_text(json.dumps(dict(
            chunks=len(chunks), samples=samples, wav_frames=sf.info(path).frames,
            stop_head=1, frame_limit=frame_limit)), encoding="utf-8")
        artifact = dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        bytes=path.stat().st_size, kind="audio_f32", derived_from=None)
        event("round", dict(index=index, warmup=index < request["warmup_rounds"], outcome="success",
                            synthesis_seconds=elapsed, first_pcm_seconds=first,
                            audio_seconds=samples/rate, delivery_boundary="native_streaming_first_pcm",
                            output_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                            observations=[], artifacts=[artifact]))
    event("terminal", dict(outcome="success"))


if __name__ == "__main__":
    main()
