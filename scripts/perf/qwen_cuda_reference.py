"""Pinned official Qwen batch APIs; xtask owns processes and repetition."""
import contextlib
import hashlib
import json
from functools import partial
from pathlib import Path
import sys
import time
from unittest.mock import patch


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output = Path(sys.argv[2]).resolve()
    config = request["request"]
    sys.path.insert(0, config["checkout"])
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import soundfile as sf
        import torch
        import transformers
        import safetensors
        from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer, VoiceClonePromptItem

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    clock = time.perf_counter()
    sequence = 0

    def event(kind, payload):
        nonlocal sequence
        print(json.dumps(dict(schema="talechime/performance/1", attempt_id=request["attempt_id"],
                              sequence=sequence, offset_seconds=time.perf_counter()-clock,
                              kind=kind, payload=payload)), flush=True)
        sequence += 1

    # Preserve the checkpoint's F32 codec, matching the native production path.
    # Override loading kwargs only; the official tokenizer implementation is unchanged.
    original_loader = Qwen3TTSTokenizer.from_pretrained

    def load_codec(*args, **kwargs):
        kwargs["dtype"] = torch.float32
        return original_loader(*args, **kwargs)

    started = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr), contextlib.ExitStack() as loading, patch.object(
        Qwen3TTSTokenizer, "from_pretrained", side_effect=load_codec
    ):
        if config.get("checkpoint_backend") == "pread":
            loading.enter_context(patch("transformers.modeling_utils.safe_open",
                                        partial(safetensors.safe_open, backend="pread")))
        model = Qwen3TTSModel.from_pretrained(config["model"], dtype=torch.bfloat16,
                                             device_map="cuda:0", attn_implementation="sdpa",
                                             local_files_only=True)
    model.model.eval()
    codec = model.model.speech_tokenizer.model
    assert next(model.model.talker.parameters()).dtype == torch.bfloat16
    assert next(codec.parameters()).dtype == torch.float32
    torch.cuda.synchronize()
    runtime = dict(torch=torch.__version__, transformers=transformers.__version__,
                   safetensors=safetensors.__version__, checkpoint_backend=config.get("checkpoint_backend", "mmap"),
                   model_dtype="bfloat16", codec_dtype="float32", attention="sdpa",
                   load_seconds=time.perf_counter()-started, codec_loading_override=True,
                   delivery="official batch API; first PCM equals full synthesis time",
                   flash_attention_installed=False, source=config["checkout"])
    conditioning = dict(speaker="Uncle_Fu", instruct="")
    synthesize = model.generate_custom_voice
    if config.get("mode") == "base":
        voice = json.loads(Path(config["voice_metadata"]).read_text(encoding="utf-8"))
        reference_started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            if config.get("native_prompt"):
                native = json.loads(Path(config["native_prompt"]).read_text(encoding="utf-8"))
                prompt = [VoiceClonePromptItem(
                    ref_code=torch.tensor(native["codes"], dtype=torch.long, device="cuda:0"),
                    ref_spk_embedding=torch.tensor(native["embedding"], dtype=torch.float32, device="cuda:0"),
                    x_vector_only_mode=False, icl_mode=True, ref_text=voice["transcript"])]
            else:
                prompt = model.create_voice_clone_prompt(
                    ref_audio=config["reference_wav"], ref_text=voice["transcript"],
                    x_vector_only_mode=False)
        torch.cuda.synchronize()
        runtime["reference_preparation_seconds"] = time.perf_counter()-reference_started
        runtime["speaker_encoder_dtype"] = str(next(model.model.speaker_encoder.parameters()).dtype)
        runtime["reference_code_frames"] = prompt[0].ref_code.shape[0]
        runtime["speaker_embedding_elements"] = prompt[0].ref_spk_embedding.numel()
        runtime["conditioning"] = ("shared native prompt; official reference encoding skipped"
                                   if config.get("native_prompt") else "ICL, prompt prepared once before hot rounds")
        if config.get("export_reference_prompt", False):
            (output / "reference-prompt.json").write_text(json.dumps(dict(
                embedding=prompt[0].ref_spk_embedding.detach().float().cpu().reshape(-1).tolist(),
                codes=prompt[0].ref_code.detach().cpu().tolist()
            )), encoding="utf-8")
        conditioning = dict(voice_clone_prompt=prompt)
        synthesize = model.generate_voice_clone
    elif config.get("mode") == "design":
        conditioning = dict(instruct=config["description"])
        synthesize = model.generate_voice_design
        runtime["conditioning"] = "VoiceDesign description; independent of narration speaker IDs"
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    event("prepared", runtime)
    text = Path(config["text_file"]).read_text(encoding="utf-8").strip()
    if config.get("prefill_capture", False):
        prefill = {}

        def capture_inputs(module, args, kwargs):
            embeds = kwargs.get("inputs_embeds")
            if prefill or embeds is None or embeds.shape[1] <= 1:
                return
            prefill["input_shape"] = list(embeds.shape)
            prefill["inputs"] = embeds.detach().float().cpu().reshape(-1).tolist()
            for name in ("position_ids", "cache_position"):
                value = kwargs.get(name)
                prefill[name] = None if value is None else value.detach().cpu().tolist()

        def capture_hidden(module, args, result):
            if "input_shape" in prefill and "last_hidden" not in prefill:
                hidden = result.last_hidden_state.detach().float().cpu()
                prefill["prefix_hidden"] = hidden[:, :9].reshape(-1).tolist()
                prefill["last_hidden"] = hidden[:, -1].reshape(-1).tolist()

        def capture_logits(module, args, result):
            if "last_hidden" in prefill and "logits" not in prefill:
                prefill["logits"] = result.logits[:, -1].detach().float().cpu().reshape(-1).tolist()
                (output / "prefill.json").write_text(json.dumps(prefill), encoding="utf-8")

        model.model.talker.model.register_forward_pre_hook(capture_inputs, with_kwargs=True)
        model.model.talker.model.register_forward_hook(capture_hidden)
        model.model.talker.register_forward_hook(capture_logits)
    original_generate = model.model.generate
    generated_codes = []
    diagnostics = {}
    original_talker_generate = model.model.talker.generate

    def checked_talker_generate(*args, **kwargs):
        result = original_talker_generate(*args, **kwargs)
        ids = result.sequences
        eos = model.model.config.talker_config.codec_eos_token_id
        ended = bool(torch.all(ids[:, -1] == eos).item())
        if not ended:
            (output / "failed-generation-termination.json").write_text(
                json.dumps(dict(tokens=ids.shape[-1], last_tokens=ids[:, -1].tolist(),
                                eos_token_id=eos, ended_with_eos=False)), encoding="utf-8"
            )
        assert ended, "official generation reached its budget without EOS"
        return result

    model.model.talker.generate = checked_talker_generate
    if config.get("diagnostics", False):
        checked_generate = model.model.talker.generate
        original_decode = model.model.speech_tokenizer.decode

        def diagnose_talker(*args, **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = checked_generate(*args, **kwargs)
            torch.cuda.synchronize()
            diagnostics["talker_generate_seconds"] = time.perf_counter()-started
            ids = result.sequences.detach().cpu().tolist()
            diagnostics["semantic_sequences"] = ids
            diagnostics["eos_token_id"] = model.model.config.talker_config.codec_eos_token_id
            return result

        def diagnose_decode(*args, **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = original_decode(*args, **kwargs)
            torch.cuda.synchronize()
            diagnostics["codec_decode_seconds"] = time.perf_counter()-started
            return result

        model.model.talker.generate = diagnose_talker
        model.model.speech_tokenizer.decode = diagnose_decode

    def generate(*args, **kwargs):
        result = original_generate(*args, **kwargs)
        generated_codes[:] = result[0]
        assert len(generated_codes) == 1 and 0 < generated_codes[0].shape[0] < 375, "frame budget reached"
        if config.get("diagnostics", False):
            diagnostics["code_frames"] = generated_codes[0].shape[0]
        return result

    for index in range(request["warmup_rounds"] + request["hot_rounds"]):
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode(), patch.object(
            model.model, "generate", side_effect=generate
        ):
            wavs, sample_rate = synthesize(
                text=text, language="Chinese", **conditioning,
                non_streaming_mode=False, do_sample=not config.get("semantic_greedy", False), top_k=50, top_p=0.9,
                temperature=0.9, repetition_penalty=config.get("repetition_penalty", 1.05), subtalker_dosample=False,
                max_new_tokens=375)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        assert len(wavs) == 1 and sample_rate == 24000
        pcm = np.asarray(wavs[0], dtype="<f4").reshape(-1)
        assert pcm.size and np.isfinite(pcm).all() and np.any(pcm)
        name = f"round-{index}-f32.wav"
        sf.write(output / name, pcm, sample_rate, subtype="FLOAT")
        path = output / name
        if config.get("diagnostics", False):
            (output / f"round-{index}-diagnostics.json").write_text(
                json.dumps(diagnostics, indent=2), encoding="utf-8"
            )
        artifact = dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        bytes=path.stat().st_size, kind="audio_f32", derived_from=None)
        event("round", dict(index=index, warmup=index < request["warmup_rounds"], outcome="success",
                            synthesis_seconds=elapsed, audio_seconds=pcm.size/sample_rate,
                            first_pcm_seconds=elapsed, delivery_boundary="segment_delivery_first_pcm",
                            output_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(), observations=[],
                            artifacts=[artifact]))
        generated_codes.clear()
    event("terminal", dict(outcome="success"))


if __name__ == "__main__":
    main()
