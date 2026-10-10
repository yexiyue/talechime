"""Development-only Nano references from explicitly supplied pinned official sources.

Args: OFFICIAL_SOURCE_DIR NATIVE_MODEL_DIR REFERENCE_WAV OUTPUT_DIR
The source directory contains official_tts/ and the two codec Python files.
No model downloads or production Python dependency are introduced.
"""
import json
import sys
from pathlib import Path

import numpy as np
import sentencepiece as sp
import soundfile as sf
import torch
from transformers import GPT2Config


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def main():
    source, model_dir, reference_wav, output = map(Path, sys.argv[1:])
    sys.path.insert(0, str(source.resolve()))
    from official_tts.gpt2_decoder import MossTTSNanoGPT2Model
    from official_tts.configuration_moss_tts_nano import MossTTSNanoConfig
    from official_tts.modeling_moss_tts_nano import MossTTSNanoForCausalLM
    from configuration_moss_audio_tokenizer import MossAudioTokenizerConfig
    from modeling_moss_audio_tokenizer import MossAudioTokenizerModel

    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    codec_config = MossAudioTokenizerConfig.from_pretrained(model_dir / "codec", local_files_only=True)
    codec = MossAudioTokenizerModel.from_pretrained(
        model_dir / "codec", config=codec_config, local_files_only=True,
    ).eval()
    audio, rate = sf.read(reference_wav, dtype="float32")
    assert rate == 48000 and audio.ndim == 2 and audio.shape[1] == 2
    with torch.inference_mode():
        encoded = codec.encode(torch.from_numpy(audio.T.copy()).unsqueeze(0), num_quantizers=16)
    codes = encoded.audio_codes[:, 0, :int(encoded.audio_codes_lengths[0])].T.tolist()
    write(output / "encoder-reference.json", {"samples": audio.flatten().tolist(), "codes": codes})

    config = json.loads((model_dir / "tts/config.json").read_text())
    weights = torch.load(model_dir / "tts/pytorch_model.bin", map_location="cpu", weights_only=True)
    global_model = MossTTSNanoGPT2Model(GPT2Config(**config["gpt2_config"]), attn_implementation="sdpa").eval()
    global_model.load_state_dict({k.removeprefix("transformer."): v.float() for k, v in weights.items() if k.startswith("transformer.")})
    local_config = dict(config["gpt2_config"], n_layer=config["local_transformer_layers"])
    local = MossTTSNanoGPT2Model(GPT2Config(**local_config), attn_implementation="sdpa").eval()
    local.wte = torch.nn.Identity()
    local.load_state_dict({k.removeprefix("local_transformer."): v.float() for k, v in weights.items() if k.startswith("local_transformer.")})
    rows = [[10+i]+[1024]*16 for i in range(32)] + [[9]+[(i*17+j*31) % 1024 for j in range(16)] for i in range(64)]
    ids = torch.tensor(rows).unsqueeze(0)
    embed = lambda ids, name: torch.nn.functional.embedding(ids, weights[name].float())
    x = embed(ids[:, :, 0], "transformer.wte.weight")
    for i in range(16):
        a = ids[:, :, i+1]
        x += embed(a.clamp(max=1023), f"audio_embeddings.{i}.weight") * (a != 1024).unsqueeze(-1)
    with torch.inference_mode():
        h = global_model(inputs_embeds=x, use_cache=False, return_dict=True).last_hidden_state[:, -1:, :]
        inp = torch.cat([h, embed(torch.tensor([[9]]), "transformer.wte.weight")], dim=1)
        l = local(inputs_embeds=inp, use_cache=False, return_dict=True).last_hidden_state
    write(output / "tts-reference.json", {"rows": rows, "global": h.flatten().tolist(), "local": l.flatten().tolist()})
    del global_model, local

    c = MossTTSNanoConfig.from_pretrained(model_dir / "tts", local_files_only=True)
    c.attn_implementation = "sdpa"
    c.local_transformer_attn_implementation = "sdpa"
    model = MossTTSNanoForCausalLM(c).eval()
    model.load_state_dict(weights)
    model.float()
    tokenizer = sp.SentencePieceProcessor(model_file=str(model_dir / "tts/tokenizer.model"))
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "crates/talechime-backends/src/moss/assets/browser_poc_manifest.json").read_text())
    codes = torch.tensor(next(v for v in manifest["builtin_voices"] if v["voice"] == "Weiguo")["prompt_audio_codes"])
    prompt_cases = []
    for previous in [None, "上午好。"]:
        target = "你好。"
        dummy_codes = torch.full((2, 16), 42)
        prompt, _ = model.build_inference_input_ids(
            target, tokenizer, mode="voice_clone" if previous is None else "continuation",
            prompt_text=previous, prompt_audio_codes=dummy_codes,
        )
        prompt_cases.append({
            "tokens": tokenizer.encode(target if previous is None else previous + target),
            "codes": dummy_codes.tolist(), "continuation": previous is not None,
            "none_tokens": tokenizer.encode("None"), "rows": prompt[0].tolist(),
        })
    write(output / "prompt-reference.json", {"revision": "44502f80dbf9743528fa921cc544d662c685ebec", "cases": prompt_cases})

    # Control RNG draws to isolate inference math. Official Torch's RNG differs
    # from Candle's LCG; this is not a claim of identical default seeded output.
    state = 42
    def draw(probs, num_samples, **kwargs):
        nonlocal state
        assert num_samples == 1 and probs.shape[0] == 1
        state = (state * 1664525 + 1013904223) & 0xffffffff
        return (probs.double().cumsum(-1) < (state >> 8) / 16777216).sum(-1, keepdim=True)

    texts = [
        "傍晚，林间的小路渐渐安静下来。陈舟收起地图，沿着河岸向前走。",
        "风从桥下吹过，远处的灯一盏接一盏亮起。",
        "他放慢脚步，听见水声，也听见村口有人轻轻呼唤他的名字。",
    ]
    cases, previous = [], None
    original_multinomial = torch.multinomial
    try:
        torch.multinomial = draw
        with torch.inference_mode():
            for i, text in enumerate(texts):
                state = 42
                rows, mask = model.build_inference_input_ids(
                    text, tokenizer, mode="voice_clone" if previous is None else "continuation",
                    prompt_text=previous, prompt_audio_codes=codes,
                )
                text_logits, audio_logits = [], [[] for _ in range(16)]
                hooks = [model.text_lm_head.register_forward_hook(
                    lambda module, args, value: text_logits.append(value[0, [9, 7]].tolist())
                )]
                for channel, head in enumerate(model.audio_lm_heads):
                    hooks.append(head.register_forward_hook(
                        lambda module, args, value, channel=channel: audio_logits[channel].append(value[0, ::64].tolist())
                    ))
                frames = [e["audio_token_ids"][0].tolist() for e in model._iter_generation_events(
                    rows, attention_mask=mask, max_new_frames=375, do_sample=True,
                    text_temperature=1., text_top_p=1., text_top_k=50,
                    audio_temperature=.8, audio_top_p=.95, audio_top_k=25, audio_repetition_penalty=1.2,
                ) if e["type"] == "frame"]
                for hook in hooks:
                    hook.remove()
                assert 0 < len(frames) < 375
                cases.append({"rows": rows[0].tolist(), "frames": frames, "text_logits": text_logits, "audio_logits": audio_logits})
                codec._reset_batch_decode_streaming_state()
                if previous is not None:
                    codec.batch_decode([codes.T], streaming=True, max_batch_size=1, reset_stream=True)
                chunks = []
                for start in range(0, len(frames), 3):
                    decoded = codec.batch_decode(
                        [torch.tensor(frames[start:start+3]).T], streaming=True, max_batch_size=1,
                        reset_stream=previous is None and start == 0,
                    )
                    chunks.append(decoded.audio[0].T)
                audio = torch.cat(chunks).numpy()
                sf.write(output / f"official-{i+1}.wav", audio, 48000, subtype="FLOAT")
                print(f"case {i}: {len(frames)} frames, RMS {20*np.log10(np.sqrt(np.mean(audio**2))):.3f} dBFS", flush=True)
                codec._reset_batch_decode_streaming_state()
                encoded = codec.encode(torch.from_numpy(audio.T.copy()).unsqueeze(0), num_quantizers=16)
                codes = encoded.audio_codes[:, 0, :int(encoded.audio_codes_lengths[0])].T
                previous = text
    finally:
        torch.multinomial = original_multinomial
        codec._reset_batch_decode_streaming_state()
    write(output / "generation-reference.json", cases)


if __name__ == "__main__":
    main()
