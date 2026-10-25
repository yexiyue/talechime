"""Official VoiceDesign with per-request loading/export/teardown, driven by xtask."""
import contextlib
import gc
import hashlib
import json
from functools import partial
from pathlib import Path
import sys
import time
import weakref
from unittest.mock import patch


def main():
    envelope = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output = Path(sys.argv[2])
    config = envelope["request"]
    sys.path.insert(0, config["checkout"])
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import soundfile as sf
        import torch
        import safetensors
        from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    clock = time.perf_counter()
    sequence = 0

    def event(kind, payload):
        nonlocal sequence
        print(json.dumps(dict(schema="talechime/performance/1", attempt_id=envelope["attempt_id"],
                              sequence=sequence, offset_seconds=time.perf_counter()-clock,
                              kind=kind, payload=payload)), flush=True)
        sequence += 1

    original_loader = Qwen3TTSTokenizer.from_pretrained

    def load_codec(*args, **kwargs):
        kwargs["dtype"] = torch.float32
        return original_loader(*args, **kwargs)

    text = Path(config["text_file"]).read_text(encoding="utf-8").strip()

    def run_round(path):
        with contextlib.redirect_stdout(sys.stderr), patch.object(
            Qwen3TTSTokenizer, "from_pretrained", side_effect=load_codec
        ), patch("transformers.modeling_utils.safe_open", partial(safetensors.safe_open, backend="pread")):
            model = Qwen3TTSModel.from_pretrained(config["model"], dtype=torch.bfloat16,
                                                device_map="cuda:0", attn_implementation="sdpa",
                                                local_files_only=True)
        model.model.eval()
        assert next(model.model.parameters()).dtype == torch.bfloat16
        assert next(model.model.speech_tokenizer.model.parameters()).dtype == torch.float32
        original_generate = model.model.talker.generate
        ended = []

        def checked_generate(*args, **kwargs):
            result = original_generate(*args, **kwargs)
            ended.append(bool(torch.all(result.sequences[:, -1] == model.model.config.talker_config.codec_eos_token_id).item()))
            return result

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode(), patch.object(
            model.model.talker, "generate", side_effect=checked_generate
        ):
            wavs, rate = model.generate_voice_design(
                text=text, instruct=config["description"], language="Chinese",
                non_streaming_mode=False, do_sample=True, top_k=50, top_p=0.9, temperature=0.9,
                repetition_penalty=1.05, subtalker_dosample=False, max_new_tokens=375)
        torch.cuda.synchronize()
        assert ended and all(ended) and len(wavs) == 1 and rate == 24000
        samples = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        assert np.isfinite(samples).all() and rate <= samples.size <= rate*15
        assert np.abs(samples).max() > 0.001
        # Match the production reference export's clamp/truncate convention.
        integers = (np.clip(samples, -1, 1)*32767).astype(np.int16)
        sf.write(path, integers, rate, subtype="PCM_16")
        return samples.size, rate, weakref.ref(model)

    event("prepared", dict(scope="official VoiceDesign per-request load/generate/export/teardown",
                           torch=torch.__version__, seed=42, dtype="BF16 main/F32 codec",
                           checkpoint_backend="pread", exported_format="PCM16",
                           teardown="Python scope exit, gc.collect and cuda.empty_cache included"))
    for index in range(envelope["warmup_rounds"] + envelope["hot_rounds"]):
        path = output / f"round-{index}.wav"
        assert not path.exists()
        torch.cuda.synchronize()
        started = time.perf_counter()
        samples, rate, model_ref = run_round(path)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        event("observation", dict(round=index, model_released=model_ref() is None,
                                  cuda_allocated_after_teardown=torch.cuda.memory_allocated(),
                                  cuda_reserved_after_teardown=torch.cuda.memory_reserved()))
        assert model_ref() is None, "official model remains live after teardown"
        assert torch.cuda.memory_allocated() < 64*1024*1024, "CUDA tensors remain after teardown"
        exported, exported_rate = sf.read(path, dtype="float32")
        assert exported_rate == rate and exported.size == samples
        artifact = dict(path=path.name, kind="audio_pcm16", bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), derived_from=None)
        event("round", dict(index=index, warmup=index < envelope["warmup_rounds"], outcome="success",
                            synthesis_seconds=elapsed, audio_seconds=samples/rate, first_pcm_seconds=elapsed,
                            delivery_boundary="segment_delivery_first_pcm",
                            output_sha256=hashlib.sha256(exported.astype("<f4").tobytes()).hexdigest(),
                            observations=[], artifacts=[artifact]))
    event("terminal", dict(outcome="success"))


if __name__ == "__main__":
    main()
