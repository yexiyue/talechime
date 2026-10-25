"""Official CUDA adapter; Rust xtask owns repetition, containment and evidence audit."""
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import transformers
from omnivoice.models.omnivoice import OmniVoice


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output = Path(sys.argv[2])
    config = request["request"]
    clock = time.perf_counter()
    sequence = 0

    def event(kind, payload):
        nonlocal sequence
        print(json.dumps(dict(schema="talechime/performance/1", attempt_id=request["attempt_id"], sequence=sequence, offset_seconds=time.perf_counter()-clock, kind=kind, payload=payload)), flush=True)
        sequence += 1

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    model = OmniVoice.from_pretrained(config["model"], device_map="cuda", dtype=torch.float16, local_files_only=True)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    conditioning = {}
    prompt_seconds = None
    prompt_frames = None
    if config.get("reference_audio"):
        started = time.perf_counter()
        prompt = model.create_voice_clone_prompt(
            ref_audio=config["reference_audio"], ref_text=config["reference_text"], preprocess_prompt=True
        )
        torch.cuda.synchronize()
        prompt_seconds = time.perf_counter() - started
        prompt_frames = list(prompt.ref_audio_tokens.shape)
        conditioning["voice_clone_prompt"] = prompt
    runtime = dict(source=inspect.getfile(OmniVoice), torch=torch.__version__, transformers=transformers.__version__, attention=model.llm.config._attn_implementation,
                   llm_dtype=str(next(model.llm.parameters()).dtype), decoder_dtype=str(next(model.audio_tokenizer.parameters()).dtype), load_seconds=load_seconds,
                   reference_prepare_seconds=prompt_seconds, reference_token_shape=prompt_frames)
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    event("prepared", runtime)
    text = Path(config["text_file"]).read_text(encoding="utf-8").strip()
    texts = config.get("texts", [text])
    seeds = config.get("seeds", [config["seed"]] * len(texts))
    assert texts and len(texts) == len(seeds)
    for index in range(request["warmup_rounds"] + request["hot_rounds"]):
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.synchronize()
        start = time.perf_counter()
        chunks = []
        first_pcm = None
        with torch.inference_mode():
            for part, (segment, seed) in enumerate(zip(texts, seeds)):
                if len(texts) > 1:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                audios = model.generate(text=segment, language="zh", num_step=32, guidance_scale=2.0, t_shift=0.1, layer_penalty_factor=5.0, position_temperature=5.0, class_temperature=0.0, **conditioning)
                chunks.append(audios[0])
                if part == 0:
                    torch.cuda.synchronize()
                    first_pcm = time.perf_counter() - start
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        pcm = np.asarray(np.concatenate(chunks) if len(chunks) > 1 else chunks[0], dtype=np.float32)
        assert pcm.ndim == 1 and pcm.size > 0 and np.isfinite(pcm).all()
        name = f"round-{index}-f32.wav"
        sf.write(output / name, pcm, model.sampling_rate, subtype="FLOAT")
        artifact = dict(path=name, sha256=hashlib.sha256((output/name).read_bytes()).hexdigest(), bytes=(output/name).stat().st_size, kind="audio_f32", derived_from=None)
        event("round", dict(index=index, warmup=index < request["warmup_rounds"], outcome="success", synthesis_seconds=elapsed, audio_seconds=pcm.size/model.sampling_rate,
                            first_pcm_seconds=first_pcm, delivery_boundary="segment_delivery_first_pcm", output_sha256=hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest(), observations=[], artifacts=[artifact]))
    event("terminal", dict(outcome="success"))


if __name__ == "__main__":
    main()
