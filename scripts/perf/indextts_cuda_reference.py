"""Pinned official IndexTTS 2.5 adapter; Rust xtask owns process/round evidence."""
import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output = Path(sys.argv[2]).resolve()
    config = request["request"]
    os.chdir(config["checkout"])
    sys.path.insert(0, config["checkout"])
    # Upstream prints progress on stdout. Keep the event stream valid JSON Lines.
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import soundfile as sf
        import torch
        import transformers
        import safetensors
        from indextts.infer_v2_5 import IndexTTS2
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    clock = time.perf_counter()
    sequence = 0

    def event(kind, payload):
        nonlocal sequence
        print(json.dumps(dict(schema="talechime/performance/1", attempt_id=request["attempt_id"], sequence=sequence, offset_seconds=time.perf_counter()-clock, kind=kind, payload=payload)), flush=True)
        sequence += 1

    checkpoint_backend = config.get("checkpoint_backend", "mmap")
    assert checkpoint_backend in ("mmap", "pread")
    opened_checkpoints = []
    mapped_gpt_checkpoints = []
    original_torch_load = torch.load
    meta_gpt_audit = None

    def construct_meta_gpt(*args, **kwargs):
        with torch.device("meta"):
            return original_gpt_class(*args, **kwargs)

    def load_meta_gpt(model, filename):
        nonlocal meta_gpt_audit
        oracle = json.loads(Path(config["gpt_loading_state_audit"]).read_text(encoding="utf-8"))
        state = original_torch_load(filename, map_location="cpu", mmap=True, weights_only=True)
        state = state["model"] if "model" in state else state
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        assert not missing and not unexpected
        audited = 0
        for key, tensor in list(model.named_buffers()):
            if not tensor.is_meta:
                continue
            expected = oracle["summary"]["nonpersistent_buffers"][key]
            if key.endswith(".attn.bias") and tensor.dtype == torch.bool:
                value = torch.tril(torch.ones(tensor.shape[-2:], dtype=torch.bool)).view(tensor.shape)
            elif key.endswith(".attn.masked_bias") and tensor.ndim == 0:
                value = torch.tensor(-1e4, dtype=tensor.dtype)
            else:
                raise AssertionError(f"unhandled meta buffer: {key}")
            assert list(value.shape) == expected["shape"] and str(value.dtype) == expected["dtype"]
            assert hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest() == expected["sha256"], key
            prefix, leaf = key.rsplit(".", 1)
            model.get_submodule(prefix)._buffers[leaf] = value
            audited += 1
        assert not any(t.is_meta for t in list(model.parameters()) + list(model.buffers()))
        meta_gpt_audit = dict(state_tensors=len(state), restored_nonpersistent_buffers=audited)
        return {}

    def load_with_mapped_gpt(filename, *args, **kwargs):
        expected = Path(config["model"]) / "gpt.pth"
        if isinstance(filename, (str, os.PathLike)) and Path(filename).resolve() == expected.resolve():
            kwargs["mmap"] = True
            mapped_gpt_checkpoints.append(str(filename))
        return original_torch_load(filename, *args, **kwargs)

    def open_checkpoint(filename, *args, **kwargs):
        opened_checkpoints.append(str(filename))
        kwargs["backend"] = "pread"
        return safetensors.safe_open(filename, *args, **kwargs)

    started = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr), contextlib.ExitStack() as loading:
        if config.get("gpt_meta_assign", False):
            import indextts.infer_v2_5 as official_module
            original_gpt_class = official_module.UnifiedVoice
            loading.enter_context(patch.object(official_module, "UnifiedVoice", construct_meta_gpt))
            loading.enter_context(patch.object(official_module, "load_checkpoint", load_meta_gpt))
        if checkpoint_backend == "pread":
            loading.enter_context(patch("transformers.modeling_utils.safe_open", open_checkpoint))
        if config.get("gpt_checkpoint_mmap", False):
            loading.enter_context(patch("torch.load", load_with_mapped_gpt))
        model = IndexTTS2(cfg_path=config["config"], model_dir=config["model"],
                          device="cuda:0", use_bf16=config.get("use_bf16", False), use_cuda_kernel=False,
                          use_deepspeed=False, use_accel=False,
                          use_torch_compile=False, use_qwen_emo=False)
    torch.cuda.synchronize()
    runtime = dict(source=inspect.getfile(IndexTTS2), torch=torch.__version__, transformers=transformers.__version__, load_seconds=time.perf_counter()-started,
                   safetensors=safetensors.__version__, checkpoint_backend=checkpoint_backend,
                   cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                   matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                   float32_matmul_precision=torch.get_float32_matmul_precision(),
                   opened_checkpoints=opened_checkpoints,
                   mapped_gpt_checkpoints=mapped_gpt_checkpoints,
                   meta_gpt_loading_audit=meta_gpt_audit,
                   gpt_dtype=str(next(model.gpt.parameters()).dtype), codec_dtype=str(next(model.semantic_codec.parameters()).dtype),
                   cfm_dtype=str(next(model.s2mel.parameters()).dtype), decoder_dtype=str(next(model.bigvgan.parameters()).dtype),
                   reference_prepare_seconds=None, reference_prepare_reason="official lazy reference cache; first synthesis includes preparation",
                   output_contract="official infer returns int16; export float32 normalized by 32767", cwd=os.getcwd(),
                   cuda_allocated_after_load=torch.cuda.memory_allocated(), cuda_reserved_after_load=torch.cuda.memory_reserved())
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    event("prepared", runtime)
    sampled_stops, raw_lengths, stage_times = [], [], {}

    def observe_stage(name, operation):
        def measured(*args, **kwargs):
            if not config.get("profile_stages", False):
                return operation(*args, **kwargs)
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = operation(*args, **kwargs)
            torch.cuda.synchronize()
            stage_times.setdefault(name, []).append(time.perf_counter()-started)
            return result
        return measured

    original_speech = observe_stage("gpt_inference_speech", model.gpt.inference_speech)
    model.s2mel.models["cfm"].inference = observe_stage("cfm", model.s2mel.models["cfm"].inference)
    model.bigvgan.forward = observe_stage("bigvgan", model.bigvgan.forward)

    def confirmed_eos(stops):
        return bool(stops) and all(bool(stop.eq(model.stop_mel_token).all().item()) for stop in stops)

    def observe_speech(*args, **kwargs):
        result = original_speech(*args, **kwargs)
        codes = result[0]
        sampled_stops.append(codes[:, -1].detach().clone())
        raw_lengths.append(codes.shape[-1])
        if config.get("diagnostic_eos_before_decode", False):
            (output / "predecode-code-audit.json").write_text(json.dumps(dict(
                raw_code_lengths=raw_lengths, last_tokens=[stop.cpu().tolist() for stop in sampled_stops],
                stop_token=model.stop_mel_token)), encoding="utf-8")
            assert confirmed_eos(sampled_stops), "no confirmed GPT EOS before decode"
        return result

    model.gpt.inference_speech = observe_speech
    text = Path(config["text_file"]).read_text(encoding="utf-8").strip()
    for index in range(request["warmup_rounds"] + request["hot_rounds"]):
        sampled_stops.clear()
        raw_lengths.clear()
        stage_times.clear()
        random.seed(config["seed"])
        np.random.seed(config["seed"] % (2**32))
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            sample_rate, samples = model.infer(
                spk_audio_prompt=config["reference_audio"], text=text,
                output_path=None, lang="ZH", verbose=False,
                duration_factor=1.0, text_normalization=True,
                max_text_tokens_per_segment=120, interval_silence=200,
                do_sample=True, top_p=0.8, top_k=30, temperature=0.8,
                length_penalty=0.0, num_beams=3, repetition_penalty=10.0,
                max_mel_tokens=config.get("max_mel_tokens", 1500))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        assert confirmed_eos(sampled_stops), "no confirmed GPT EOS"
        (output / f"round-{index}-detail.json").write_text(json.dumps(dict(
            raw_code_lengths=raw_lengths, last_tokens=[stop.cpu().tolist() for stop in sampled_stops],
            stop_token=model.stop_mel_token, synchronized_stages=stage_times)), encoding="utf-8")
        assert samples.dtype == np.int16
        pcm = samples.reshape(-1).astype(np.float32) / 32767.0
        assert pcm.size > 0 and np.isfinite(pcm).all()
        name = f"round-{index}-f32.wav"
        sf.write(output / name, pcm, sample_rate, subtype="FLOAT")
        path = output / name
        artifact = dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size, kind="audio_f32", derived_from=None)
        event("round", dict(index=index, warmup=index < request["warmup_rounds"], outcome="success", synthesis_seconds=elapsed,
                            audio_seconds=pcm.size/sample_rate, first_pcm_seconds=elapsed, delivery_boundary="segment_delivery_first_pcm",
                            output_sha256=hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest(), observations=[], artifacts=[artifact]))
    event("terminal", dict(outcome="success"))


if __name__ == "__main__":
    main()
