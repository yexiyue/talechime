#!/usr/bin/env python3
"""Export model-free official component fixtures (no model downloads).

PYTHONPATH=<CosyVoice>:<Matcha-TTS> python cosyvoice3_components.py --out runs/cv3-components
Use with: cargo run --release -p cosyvoice3 --example compare -- <out>
"""
import argparse
import json
from pathlib import Path

import torch
import torchaudio
import whisper
from safetensors.torch import save_file
from transformers import Qwen2Config, Qwen2Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    from cosyvoice.flow.DiT.dit import DiT
    from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor
    from matcha.utils.audio import mel_spectrogram

    # Broadband input plus a tone exercises low-energy bins and boundaries.
    audio = torch.randn(16003) * 0.03 + torch.sin(torch.arange(16003) * 0.17) * 0.2
    audio24 = torchaudio.functional.resample(audio, 16000, 24000)
    fixtures = {
        'audio': audio, 'resampled': audio24,
        'whisper': whisper.log_mel_spectrogram(audio, n_mels=128).unsqueeze(0),
        'fbank': torchaudio.compliance.kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000),
        'matcha': mel_spectrogram(audio24.unsqueeze(0), 1920, 80, 24000, 480, 1920, 0, None, center=False),
    }
    for n in (1, 7, 15, 16, 17, 31, 32, 33, 1200000):
        fixtures[f'randn_{n}'] = torch.randn(n, generator=torch.Generator().manual_seed(0))
    # Same GQA ratio as the released model: 14 query heads / 2 KV heads.
    cfg = Qwen2Config(vocab_size=64, hidden_size=56, intermediate_size=112, num_hidden_layers=2,
                      num_attention_heads=14, num_key_value_heads=2, max_position_embeddings=128,
                      rope_theta=1000000, rms_norm_eps=1e-6, attention_dropout=0)
    llm = Qwen2Model(cfg).eval()
    speech = torch.nn.Embedding(6761, 56)
    decoder = torch.nn.Linear(56, 6761, bias=False)
    weights = {'llm.' + k: v for k, v in llm.state_dict().items()}
    weights.update({'speech_embedding.weight': speech.weight, 'llm_decoder.weight': decoder.weight})
    save_file({k:v.detach().contiguous() for k,v in weights.items()}, args.out/'small-llm.safetensors')
    text = torch.tensor([[4, 9, 12]])
    prompt_text = torch.tensor([[5, 7]])
    prompt_speech = torch.tensor([[1, 2, 3]])
    embeds = torch.cat([speech(torch.tensor([[6561]])), llm.embed_tokens(torch.cat([prompt_text, text],1)),
                        speech(torch.tensor([[6563]])), speech(prompt_speech)], 1)
    result = llm(inputs_embeds=embeds, use_cache=True)
    fixtures.update(text_token=text, prompt_text_token=prompt_text, prompt_speech_token=prompt_speech,
                    first_logits=decoder(result.last_hidden_state[:, -1]))
    next_result = llm(inputs_embeds=speech(torch.tensor([[11]])), past_key_values=result.past_key_values, use_cache=True)
    fixtures['second_logits'] = decoder(next_result.last_hidden_state[:, -1])
    dit = DiT(dim=128, depth=2, heads=2, dim_head=64, ff_mult=2, mel_dim=80, spk_dim=80,
              out_channels=80, static_chunk_size=50, num_decoding_left_chunks=-1).eval()
    save_file({k:v.contiguous() for k,v in dit.state_dict().items()}, args.out/'small-dit.safetensors')
    x, mu, cond = [torch.randn(2,80,112) for _ in range(3)]
    mask, t, spks = torch.ones(2,1,112), torch.tensor([0.1,0.8]), torch.randn(2,80)
    fixtures.update(x=x, mu=mu, cond=cond, mask=mask, t=t, spks=spks,
                    dit=dit(x,mask,mu,t,spks,cond), dit_stream=dit(x,mask,mu,t,spks,cond,streaming=True))
    f0 = CausalConvRNNF0Predictor(in_channels=80, cond_channels=16).double().eval()
    # Evaluate fused weights, avoiding parametrization differences in Rust.
    from torch.nn.utils.parametrize import remove_parametrizations
    for module in f0.modules():
        if hasattr(module, 'parametrizations'):
            remove_parametrizations(module, 'weight', leave_parametrized=True)
    save_file({k:v.float().contiguous() for k,v in f0.state_dict().items()}, args.out/'small-f0.safetensors')
    f0.float().double()
    fixtures['f0_mel'] = torch.randn(1,80,12)
    fixtures['f0'] = f0(fixtures['f0_mel'].double()).float()
    save_file({k:v.detach().contiguous() for k,v in fixtures.items()}, args.out/'components.safetensors')
    (args.out/'versions.json').write_text(json.dumps({'torch':torch.__version__, 'torchaudio':torchaudio.__version__}, indent=2))
    print(f'Wrote official component fixtures to {args.out}')


if __name__ == '__main__':
    with torch.inference_mode():
        main()
