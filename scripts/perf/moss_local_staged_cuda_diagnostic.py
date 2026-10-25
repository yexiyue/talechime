"""Official GPU math with sequential residency; NOT a steady-state TTS baseline.

One request per process. Elapsed time includes both loads and the residency swap.
This diagnostic deliberately does not claim native streaming or a hot-round RTF.
"""
import contextlib
import gc
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    envelope = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output = Path(sys.argv[2])
    config = envelope["request"]
    assert envelope["warmup_rounds"] == 0 and envelope["hot_rounds"] == 1
    source = Path(config["checkout"])
    sys.path.insert(0, str(source))
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import safetensors
        import soundfile as sf
        import torch
        import transformers
        from functools import partial
        from unittest.mock import patch
        from transformers import AutoTokenizer
        from moss_tts_local.configuration_moss_tts import MossTTSDelayConfig
        from moss_tts_local.modeling_moss_tts import MossTTSDelayModel
        from moss_tts_local.processing_moss_tts import MossTTSDelayProcessor

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

    runtime = dict(kind="staged_cuda_loading_and_audio_diagnostic_not_hot_baseline",
                   torch=torch.__version__, transformers=transformers.__version__,
                   safetensors=safetensors.__version__, main_dtype="bfloat16", codec_dtype="float32",
                   autocast=False, load_deviation="direct CUDA device_map + pread header IO control",
                   residency_deviation="unload main before codec load; both inference stages CUDA",
                   delivery="one whole decoded segment; no streaming acceptance",
                   recommended_torch="2.9.1+cu128", torch_deviation=torch.__version__ != "2.9.1+cu128")
    event("prepared", runtime)
    started = time.perf_counter()
    stages = []

    def stage(name, start):
        torch.cuda.synchronize()
        stages.append(dict(name=name, seconds=time.perf_counter() - start,
                           cuda_allocated_bytes=torch.cuda.memory_allocated(),
                           cuda_reserved_bytes=torch.cuda.memory_reserved()))
        (output / "stages.json").write_text(json.dumps(stages, indent=2), encoding="utf-8")

    tick = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr), patch(
        "transformers.modeling_utils.safe_open", partial(safetensors.safe_open, backend="pread")
    ):
        model_config = MossTTSDelayConfig.from_pretrained(config["model"], local_files_only=True)
        model = MossTTSDelayModel.from_pretrained(config["model"], config=model_config,
                                                 dtype=torch.bfloat16, device_map="cuda:0",
                                                 attn_implementation="sdpa", local_files_only=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
        template = Path(config["chat_template"]).read_text(encoding="utf-8")
        assert tokenizer.chat_template in (None, template), "conflicting tokenizer template"
        tokenizer.chat_template = template
        # No reference audio is encoded for this request. The official optional
        # codec stays None until the main model has finished and been unloaded.
        processor = MossTTSDelayProcessor(tokenizer=tokenizer, audio_tokenizer=None,
                                          model_config=model.config)
    assert all(p.dtype == torch.bfloat16 and p.device.type == "cuda" for p in model.parameters())
    runtime["main_parameter_bytes"] = sum(p.numel() * p.element_size() for p in model.parameters())
    runtime["main_parameter_tensors"] = sum(1 for _ in model.parameters())
    runtime["chat_template_sha256"] = hashlib.sha256(template.encode("utf-8")).hexdigest()
    stage("main_load", tick)
    tick = time.perf_counter()
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
        text = Path(config["text_file"]).read_text(encoding="utf-8").strip()
        batch = processor([[processor.build_user_message(text=text)]], mode="generation")
        generated = model.generate(
            input_ids=batch["input_ids"].to("cuda:0"),
            attention_mask=batch["attention_mask"].to("cuda:0"), max_new_tokens=375,
            n_vq_for_inference=32, text_temperature=1.5, text_top_k=50, text_top_p=1.0,
            text_repetition_penalty=1.0, audio_temperature=1.0, audio_top_k=50,
            audio_top_p=0.95, audio_repetition_penalty=1.1,
        )
    stage("official_generate_codes", tick)
    assert len(generated) == 1
    start_length, generated_ids = generated[0]
    assert int(generated_ids[-1, 0]) == model.config.audio_end_token_id, "missing EOS or frame limit"
    assert generated_ids.ndim == 2 and generated_ids.shape[1] == 33
    raw_ids = generated_ids.detach().cpu()
    saved_generation = [(int(start_length), raw_ids)]
    np.save(output / "codes.npy", raw_ids.numpy(), allow_pickle=False)
    runtime.update(start_length=int(start_length), generated_rows=len(raw_ids),
                   audio_end_token_id=model.config.audio_end_token_id,
                   final_control_token=int(raw_ids[-1, 0]))
    tick = time.perf_counter()
    del model, generated, generated_ids
    gc.collect()
    torch.cuda.empty_cache()
    stage("main_unload", tick)
    assert torch.cuda.memory_allocated() < 64 * 1024 * 1024, "main CUDA storage still retained"
    tick = time.perf_counter()
    name = "moss_staged_reference_codec"
    source = Path(config["codec_checkout"])
    spec = importlib.util.spec_from_file_location(name, source / "__init__.py",
                                                 submodule_search_locations=[str(source)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    module = importlib.import_module(name + ".modeling_moss_audio_tokenizer")
    with contextlib.redirect_stdout(sys.stderr), patch(
        "transformers.modeling_utils.safe_open", partial(safetensors.safe_open, backend="pread")
    ):
        codec = module.MossAudioTokenizerModel.from_pretrained(
            config["codec"], dtype=torch.float32, device_map="cuda:0", local_files_only=True
        ).eval()
    assert all(p.dtype == torch.float32 and p.device.type == "cuda" for p in codec.parameters())
    stage("codec_load_default_f32", tick)
    tick = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
        processor = MossTTSDelayProcessor(tokenizer=tokenizer, audio_tokenizer=codec,
                                          model_config=processor.model_config)
        loaded_batch = processor([[processor.build_user_message(text=text)]], mode="generation")
        for key in ["input_ids", "attention_mask"]:
            assert torch.equal(batch[key], loaded_batch[key]), "codec residency changed prompt"
        runtime["prompt_equal_before_after_codec_load"] = True
        messages = processor.decode(saved_generation)
        assert len(messages) == 1 and messages[0] is not None
        assert len(messages[0].audio_codes_list) == 1
        pcm = messages[0].audio_codes_list[0].detach().float().cpu().numpy().reshape(-1)
    stage("official_codec_decode", tick)
    elapsed = time.perf_counter() - started
    assert len(pcm) > 0 and np.isfinite(pcm).all()
    sf.write(output / "staged-f32.wav", pcm, 24000, subtype="FLOAT")
    runtime.update(pcm_samples=len(pcm),
                   eos_evidence="actual final generated control token equals configured audio_end_token_id",
                   main_source=sys.modules[MossTTSDelayModel.__module__].__file__, codec_source=module.__file__)
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    artifacts = []
    for name, kind in [("staged-f32.wav", "audio_f32"), ("codes.npy", "diagnostic_codes")]:
        path = output / name
        artifacts.append(dict(path=name, kind=kind, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              bytes=path.stat().st_size, derived_from=None))
    event("round", dict(index=0, warmup=False, outcome="success", synthesis_seconds=elapsed,
                        audio_seconds=len(pcm) / 24000, first_pcm_seconds=elapsed,
                        delivery_boundary="segment_delivery_first_pcm",
                        output_sha256=hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest(),
                        observations=[], artifacts=artifacts))
    event("terminal", dict(outcome="success"))


if __name__ == "__main__":
    main()
