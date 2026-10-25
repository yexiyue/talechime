"""Pinned official Nano inference; xtask owns processes, inputs and repetition."""
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
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import sentencepiece as spm
        import soundfile as sf
        import torch
        import transformers
        from transformers import AutoModel, AutoModelForCausalLM

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    clock, sequence = time.perf_counter(), 0

    def event(kind, payload):
        nonlocal sequence
        print(json.dumps(dict(schema="talechime/performance/1", attempt_id=request["attempt_id"],
                              sequence=sequence, offset_seconds=time.perf_counter()-clock,
                              kind=kind, payload=payload)), flush=True)
        sequence += 1

    assets = Path(config["assets"])
    max_frames = config.get("max_frames", 375)
    assert isinstance(max_frames, int) and 1 <= max_frames <= 375
    started = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        model = AutoModelForCausalLM.from_pretrained(
            str(assets / "tts"), trust_remote_code=True, local_files_only=True,
            attn_implementation="sdpa").to(device="cuda", dtype=torch.float32).eval()
        model._set_attention_implementation("sdpa")
        codec = AutoModel.from_pretrained(
            str(assets / "codec"), trust_remote_code=True, local_files_only=True,
            attn_implementation="eager").to(device="cuda", dtype=torch.float32).eval()
        tokenizer = spm.SentencePieceProcessor(model_file=str(assets / "tts/tokenizer.model"))
    torch.cuda.synchronize()
    runtime = dict(torch=torch.__version__, transformers=transformers.__version__,
                   load_seconds=time.perf_counter()-started,
                   model_module=type(model).__module__, codec_module=type(codec).__module__,
                   model_dtype=str(next(model.parameters()).dtype),
                   codec_dtype=str(next(codec.parameters()).dtype),
                   attention="TTS sdpa propagated; codec eager",
                   output_mode="official full generation and full codec decode",
                   max_frames=max_frames,
                   eos_observation="retain actual sampled assistant token; no extra device sync per step")
    started = time.perf_counter()
    manifest = json.loads(Path(config["manifest"]).read_text(encoding="utf-8"))
    voice = next(v for v in manifest["builtin_voices"] if v["voice"] == config["voice"])
    templates = manifest["prompt_templates"]
    rows = []

    def append(tokens):
        rows.extend([[token] + [1024] * 16 for token in tokens])

    append([4])
    append(templates["user_prompt_prefix_token_ids"])
    append([6])
    rows.extend([[8] + codes for codes in voice["prompt_audio_codes"]])
    append([7])
    append(templates["user_prompt_after_reference_token_ids"])
    append(tokenizer.encode(Path(config["text_file"]).read_text(encoding="utf-8").strip()))
    append(templates["assistant_prompt_prefix_token_ids"])
    append([6])
    inputs = torch.tensor([rows], device="cuda", dtype=torch.long)
    torch.cuda.synchronize()
    runtime["reference_prepare_seconds"] = time.perf_counter()-started
    (output / "input.json").write_text(json.dumps(rows), encoding="utf-8")
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    event("prepared", runtime)
    last_token = []
    original_sample = model._sample_next_assistant_text_token

    def observe_token(*args, **kwargs):
        result = original_sample(*args, **kwargs)
        last_token[:] = [result.detach()]
        return result

    model._sample_next_assistant_text_token = observe_token
    for index in range(request["warmup_rounds"] + request["hot_rounds"]):
        last_token.clear()
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            generated = model.generate(
                inputs, max_new_frames=max_frames, do_sample=True, text_temperature=1.0,
                text_top_p=1.0, text_top_k=2, audio_temperature=0.8,
                audio_top_p=0.95, audio_top_k=25, audio_repetition_penalty=1.2)
            codes = model._trim_generated_audio_token_ids(generated.audio_token_ids[0], effective_nq=16)
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter()-started
            waveform, rate = model._decode_audio_token_ids_to_waveform(
                audio_tokenizer=codec, audio_token_ids=codes, target_sample_rate=48000,
                effective_nq=16, resolved_device=torch.device("cuda"))
            # Official waveform is channel-major; exported PCM is interleaved like the Backend.
            pcm = np.asarray(waveform.detach().float().cpu().numpy().T, dtype="<f4").copy()
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        assert last_token and last_token[0].numel() == 1
        assert last_token[0].item() == model.config.audio_end_token_id, "no confirmed model EOS"
        assert 0 < codes.shape[0] < max_frames
        assert rate == 48000 and pcm.ndim == 2 and pcm.shape[1] == 2
        assert pcm.shape[0] > 0 and np.isfinite(pcm).all() and np.any(pcm)
        name = f"round-{index}-f32.wav"
        path = output / name
        sf.write(path, pcm, rate, subtype="FLOAT")
        info = sf.info(path)
        assert info.frames == pcm.shape[0] and info.channels == 2
        detail = dict(generation_seconds=generation_seconds,
                      decode_and_transfer_seconds=elapsed-generation_seconds,
                      eos_token=int(last_token[0].item()), code_frames=int(codes.shape[0]),
                      channels=2, sample_rate=rate, samples_per_channel=pcm.shape[0])
        (output / f"round-{index}-detail.json").write_text(json.dumps(detail), encoding="utf-8")
        (output / f"round-{index}-codes.json").write_text(json.dumps(codes.cpu().tolist()), encoding="utf-8")
        artifact = dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        bytes=path.stat().st_size, kind="audio_f32", derived_from=None)
        event("round", dict(index=index, warmup=index < request["warmup_rounds"], outcome="success",
                            synthesis_seconds=elapsed, first_pcm_seconds=elapsed,
                            audio_seconds=pcm.shape[0]/rate, delivery_boundary="segment_delivery_first_pcm",
                            output_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                            observations=[], artifacts=[artifact]))
    event("terminal", dict(outcome="success"))


if __name__ == "__main__":
    main()
