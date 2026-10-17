"""Development-only numerical oracle for the Candle IndexTTS port.

Runtime Rust inference never invokes this script. It imports the pinned
official index-tts checkout (pass --source), loads the ORIGINAL torch
checkpoints, runs each pipeline stage on fixed inputs and dumps float
tensors as JSON for the Rust `#[ignore]` comparison tests. Outputs contain
tensors, not redistributed weights.

Setup (development machine):
    uv sync            # inside the index-tts checkout
    python tools/tts/indextts_reference.py \
        --source /path/to/index-tts \
        --checkpoints /path/to/checkpoints \
        --output target/indextts-oracle

Stages dumped (see crates/indextts validation tests):
    tokenizer   encode() ids for a fixed corpus (exact match expected)
    w2v         SeamlessM4T features + hidden_states[17] + normalized
    campplus    kaldi fbank + style embedding
    mel         reference mel spectrogram
    codec       quantize() codes + decode() features on fixed inputs
    regulator   length_regulator prompt/target conditions
    cfm         euler trajectory mel with a fixed torch seed
    bigvgan     vocoder output on the CFM mel
    gpt         prepare_gpt_inputs tensors + merge_emovec + first-step
                logits under a fixed conditioning (no sampling)
"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch


def tensor_entry(t):
    t = t.detach().to(torch.float32).cpu()
    return {"shape": list(t.shape), "data": t.flatten().tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True,
                   help="pinned index-tts checkout")
    p.add_argument("--checkpoints", type=Path, required=True,
                   help="official checkpoints directory (with hf_cache/)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-wav", type=Path, default=None,
                   help="speaker reference clip (defaults to a generated sine)")
    args = p.parse_args()
    sys.path.insert(0, str(args.source))

    # Import the upstream package without triggering its CLI/download logic.
    import indextts  # noqa: F401  (ensures package import works)

    from indextts.infer_v2_5 import IndexTTS2, apply_pronunciation_annotations
    from indextts.utils.tokenizer import get_tokenizer, lang_to_token

    args.output.mkdir(parents=True, exist_ok=True)
    out = {}

    torch.set_num_threads(2)
    torch.manual_seed(42)

    cfg_path = args.checkpoints / "config.yaml"
    tts = IndexTTS2(cfg_path=str(cfg_path), model_dir=str(args.checkpoints),
                    use_bf16=False, device="cpu", use_cuda_kernel=False)

    # --- tokenizer -------------------------------------------------------
    tokenizer = get_tokenizer(multilingual=True, model_dir=str(args.checkpoints))
    corpus = [
        "<|zh|> 欢迎大家来体验indextts2，并给予我们意见与反馈，谢谢大家。",
        "<|zh|> 共465篇，约315万字。晕XUAN4是一种GAN3觉。",
        "<|en|> this is a simple english sentence for the candle port.",
        "<|zh|> <|SPECIAL_TOKEN_2|>XING2<|SPECIAL_TOKEN_2|>是一件<|SPECIAL_TOKEN_1|>G OW1<|SPECIAL_TOKEN_1|>的事。",
    ]
    out["tokenizer"] = {"texts": corpus,
                        "ids": [tokenizer.encode(t, allowed_special="all") for t in corpus],
                        "lang_zh": int(lang_to_token("ZH")),
                        "lang_en": int(lang_to_token("EN")),
                        "lang_unknown": int(lang_to_token("xx-klingon"))}
    out["frontend"] = {
        "annotations": [apply_pronunciation_annotations(c.split(" ", 1)[-1]) for c in corpus],
        "note": "normalize() uses wetext/TN and is NOT byte-comparable with the Rust subset",
    }

    # --- reference audio -------------------------------------------------
    if args.reference_wav and Path(args.reference_wav).is_file():
        import torchaudio
        audio, sr = torchaudio.load(str(args.reference_wav))
        audio = audio[:1, : sr * 10]
    else:
        sr = 22050
        t = torch.arange(sr * 2).float() / sr
        audio = (0.3 * torch.sin(2 * 3.14159265 * 220 * t) *
                 (0.6 + 0.4 * torch.sin(2 * 3.14159265 * 3 * t))).unsqueeze(0)
    audio_22k = audio if sr == 22050 else torchaudio.functional.resample(audio, sr, 22050)
    audio_16k = torchaudio.functional.resample(audio, sr, 16000)
    out["reference"] = {"sr_in": int(sr),
                        "pcm_22k": tensor_entry(audio_22k),
                        "pcm_16k": tensor_entry(audio_16k)}

    # --- w2v features ----------------------------------------------------
    inputs = tts.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
    out["w2v"] = {
        "input_features": tensor_entry(inputs["input_features"]),
        "attention_mask": tensor_entry(inputs["attention_mask"].float()),
    }
    with torch.no_grad():
        emb = tts.semantic_model(input_features=inputs["input_features"],
                                 attention_mask=inputs["attention_mask"],
                                 output_hidden_states=True)
        h17 = emb.hidden_states[17]
    out["w2v"]["hidden_17"] = tensor_entry(h17)
    out["w2v"]["normalized"] = tensor_entry((h17 - tts.semantic_mean) / tts.semantic_std)

    # --- campplus + mel --------------------------------------------------
    feat = torchaudio.compliance.kaldi.fbank(audio_16k, num_mel_bins=80, dither=0,
                                             sample_frequency=16000)
    out["campplus"] = {"fbank": tensor_entry(feat),
                       "fbank_mean_sub": tensor_entry(feat - feat.mean(dim=0, keepdim=True))}
    with torch.no_grad():
        style = tts.campplus_model((feat - feat.mean(dim=0, keepdim=True)).unsqueeze(0))
    out["campplus"]["style"] = tensor_entry(style)

    ref_mel = tts.mel_fn(audio_22k.float())
    out["mel"] = {"ref_mel": tensor_entry(ref_mel)}

    # --- codec -----------------------------------------------------------
    spk_cond_emb = tts.get_emb(inputs["input_features"], inputs["attention_mask"])
    with torch.no_grad():
        codes, quantized = tts.semantic_codec.quantize(spk_cond_emb)
        decoded = tts.semantic_codec.decode(codes)
    out["codec"] = {"input": tensor_entry(spk_cond_emb), "codes": tensor_entry(codes.float()),
                    "quantized": tensor_entry(quantized), "decoded": tensor_entry(decoded)}

    # --- length regulator ------------------------------------------------
    with torch.no_grad():
        prompt_condition = tts.s2mel.models['length_regulator'](
            spk_cond_emb, ylens=torch.LongTensor([ref_mel.size(2)]), n_quantizers=3, f0=None)[0]
    out["regulator"] = {"ylens": [int(ref_mel.size(2))],
                        "prompt_condition": tensor_entry(prompt_condition)}

    # --- gpt conditioning + first-step logits ----------------------------
    text = corpus[0]
    seg = text.split(" ", 1)[1]
    toks = tokenizer.encode("<|zh|> " + seg, allowed_special="all")
    text_tokens = torch.IntTensor(toks).unsqueeze(0)
    text_tokens = torch.nn.functional.pad(text_tokens, (0, 1), value=1)
    lang = torch.LongTensor([lang_to_token("ZH")])
    with torch.no_grad():
        emovec = tts.gpt.merge_emovec(
            spk_cond_emb, spk_cond_emb,
            torch.tensor([spk_cond_emb.shape[-1]]),
            torch.tensor([spk_cond_emb.shape[-1]]), alpha=1.0)
        spk_latent = tts.gpt.spk_emb_proj(style).unsqueeze(0)  # (1,1,1280)
        conds_latent = torch.cat((spk_latent + emovec.unsqueeze(1),
                                  torch.zeros(1, 2, spk_latent.size(-1))), 1)
        fake_ids, embeds, attn_mask = tts.gpt.prepare_gpt_inputs(conds_latent, text_tokens, lang)
        tts.gpt.inference_model.store_mel_emb(embeds)
        logits = tts.gpt.inference_model(
            input_ids=fake_ids, attention_mask=attn_mask, use_cache=True).logits
    out["gpt"] = {"text_tokens": tensor_entry(text_tokens.float()),
                  "lang": int(lang.item()),
                  "emovec": tensor_entry(emovec),
                  "fake_ids": tensor_entry(fake_ids.float()),
                  "embeds": tensor_entry(embeds),
                  "attn_mask": tensor_entry(attn_mask.float()),
                  "first_step_logits": tensor_entry(logits[0, -1])}

    # --- cfm + bigvgan ----------------------------------------------------
    with torch.no_grad():
        S_infer = tts.semantic_codec.decode(codes)
        target_lengths = torch.LongTensor([int(S_infer.shape[1] * 1.72)])
        cond = tts.s2mel.models['length_regulator'](
            S_infer, ylens=target_lengths, n_quantizers=3, f0=None)[0]
        cat_condition = torch.cat([prompt_condition, cond], dim=1)
        torch.manual_seed(42)
        vc_target = tts.s2mel.models['cfm'].inference(
            cat_condition, torch.LongTensor([cat_condition.size(1)]), ref_mel, style, None,
            25, inference_cfg_rate=0.7)
        vc_target = vc_target[:, :, ref_mel.size(-1):]
        wav = tts.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
    out["cfm"] = {"target_lengths": [int(target_lengths[0])],
                  "cond": tensor_entry(cond),
                  "vc_target": tensor_entry(vc_target)}
    out["bigvgan"] = {"wav": tensor_entry(wav)}

    path = args.output / "oracle.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print("stage keys:", ", ".join(out.keys()))


if __name__ == "__main__":
    main()
