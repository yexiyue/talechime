#!/usr/bin/env python3
"""Fair performance benchmark for the official CosyVoice3 PyTorch code.

Boundary-matched with `crates/cosyvoice3/examples/bench.rs` so both sides
time the same spans: process start, model load, reference feature
extraction, LLM prefill/decode, flow, HiFT, per-chunk streaming arrivals.
No forward hooks, no fixture export and no file I/O inside timed regions.

Requires the pinned official checkout (074ca6dc) on PYTHONPATH together
with its matcha dependency, e.g.
  PYTHONPATH=runs/reference/CosyVoice:runs/reference/matcha \
  python tools/tts/cosyvoice3_bench.py --model_dir runs/cv3-model \
      --case_dir runs/cv3-case-short --mode warm --threads 8

`--mode mps_probe` is an explicitly unsupported configuration check: it
moves the CPU-loaded modules onto the MPS device and reports the failure
(or result); it never feeds the comparison tables.
"""

T0 = __import__('time').perf_counter()

import argparse  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

IMPORT_SECONDS = None  # set after heavy imports


def _import_stack():
    global IMPORT_SECONDS, load_file
    started = time.perf_counter()
    import torch  # noqa: F401
    from safetensors.torch import load_file as _load_file
    from cosyvoice.cli.cosyvoice import CosyVoice3
    load_file = _load_file
    IMPORT_SECONDS = time.perf_counter() - started
    return torch, CosyVoice3


def load_case(case_dir: Path):
    meta = json.loads((case_dir / 'metadata.json').read_text())
    front = load_file(str(case_dir / 'frontend.safetensors'))
    llm = load_file(str(case_dir / 'llm.safetensors'))
    prompt = {
        'flow_prompt_speech_token': front['prompt_speech_token'].long(),
        'prompt_speech_feat': front['prompt_speech_feat'].float(),
        'flow_embedding': front['prompt_embedding'].float(),
        'llm_prompt_speech_token': front['prompt_speech_token'].long(),
    }
    return meta, prompt, llm


def staged_round(torch, model, prompt, llm_tokens_fixture, seed, device_str):
    """Prefill/decode/flow/hift timing via the public modules only."""
    torch.manual_seed(seed)
    llm = model.llm
    text_token = llm_tokens_fixture['text_token'].long().to(device_str)
    prompt_text_token = llm_tokens_fixture['prompt_text_token'].long().to(device_str)
    llm_prompt = prompt['llm_prompt_speech_token'].to(device_str)

    t = time.perf_counter()
    generator = llm.inference(
        text=text_token,
        text_len=torch.tensor([text_token.shape[1]], dtype=torch.int32),
        prompt_text=prompt_text_token,
        prompt_text_len=torch.tensor([prompt_text_token.shape[1]], dtype=torch.int32),
        prompt_speech_token=llm_prompt,
        prompt_speech_token_len=torch.tensor([llm_prompt.shape[1]], dtype=torch.int32),
        embedding=torch.zeros(0, 192),
    )
    step_times = []
    tokens = []
    for out in generator:
        step_times.append(time.perf_counter() - t)
        tokens.append(int(out))
        t = time.perf_counter()
    prefill = step_times[0] if step_times else float('nan')
    decode = sum(step_times[1:])

    torch.manual_seed(seed)
    token = torch.tensor([tokens], dtype=torch.int32)
    t = time.perf_counter()
    mel, _ = model.flow.inference(
        token=token,
        token_len=torch.tensor([token.shape[1]], dtype=torch.int32),
        prompt_token=prompt['flow_prompt_speech_token'].to(device_str),
        prompt_token_len=torch.tensor([prompt['flow_prompt_speech_token'].shape[1]], dtype=torch.int32),
        prompt_feat=prompt['prompt_speech_feat'].to(device_str),
        prompt_feat_len=torch.tensor([prompt['prompt_speech_feat'].shape[1]], dtype=torch.int32),
        embedding=prompt['flow_embedding'].to(device_str),
        streaming=False,
        finalize=True,
    )
    flow_s = time.perf_counter() - t

    t = time.perf_counter()
    pcm, _ = model.hift.inference(speech_feat=mel, finalize=True)
    hift_s = time.perf_counter() - t
    pcm = pcm.float().flatten()
    return {
        'llm_prefill_seconds': prefill,
        'llm_decode_seconds': decode,
        'llm_tokens': len(tokens),
        'flow_seconds': flow_s,
        'hift_seconds': hift_s,
        'stage_pcm_samples': pcm.numel(),
        'stage_pcm_valid': bool(pcm.numel() and torch.isfinite(pcm).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--case_dir', required=True)
    parser.add_argument('--mode', required=True,
                        choices=['cold', 'warm', 'stream', 'components', 'mps_probe'])
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default=None)
    parser.add_argument('--label', default='')
    parser.add_argument('--wav_out', default=None)
    parser.add_argument('--reextract', action='store_true')
    parser.add_argument('--no_warmup', action='store_true',
                        help='skip the untimed warmup request (stream mode: keeps the official\nshared-model hop state at its fresh value of 25 for the first round)')
    args = parser.parse_args()

    torch, CosyVoice3 = _import_stack()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    else:
        args.threads = torch.get_num_threads()

    report = {
        'impl': 'python-official',
        'torch': torch.__version__,
        'onnxruntime': __import__('onnxruntime').__version__,
        'numpy': __import__('numpy').__version__,
        'commit': '074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc',
        'label': args.label,
        'device': 'cpu',
        'mode': args.mode,
        'rounds': args.rounds,
        'threads': args.threads,
        'torch_num_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'seed': args.seed,
        'reextract': args.reextract,
        'started_unix_ms': int(time.time() * 1000),
    }

    case_dir = Path(args.case_dir)
    meta, prompt, llm_fixture = load_case(case_dir)
    text = meta['text']
    prompt_text = meta['prompt_text']
    prompt_wav = meta.get('prompt_wav')

    t = time.perf_counter()
    cv3 = CosyVoice3(args.model_dir)
    load_seconds = time.perf_counter() - t
    report['pipeline_load_seconds'] = load_seconds
    report['import_seconds'] = IMPORT_SECONDS
    report['ready_since_process_start'] = time.perf_counter() - T0
    model = cv3.model
    frontend = cv3.frontend
    device_str = str(model.device)

    import torchaudio  # noqa: E402  (official dependency)

    def extract_features():
        t0 = time.perf_counter()
        model_input = frontend.frontend_zero_shot(text, prompt_text, prompt_wav, 24000, '')
        return time.perf_counter() - t0, model_input

    def fixture_input():
        mi = {
            'text': llm_fixture['text_token'].long(),
            'prompt_text': llm_fixture['prompt_text_token'].long(),
            'llm_prompt_speech_token': prompt['llm_prompt_speech_token'],
            'flow_prompt_speech_token': prompt['flow_prompt_speech_token'],
            'prompt_speech_feat': prompt['prompt_speech_feat'],
            'flow_embedding': prompt['flow_embedding'],
        }
        return mi

    def save_wav(pcm, path):
        wave = pcm.float().cpu().reshape(1, -1)
        torchaudio.save(path, wave, 24000)

    def mark(event):
        print(f"BENCH_{event}", file=sys.stderr, flush=True)

    def unix_ms():
        return int(time.time() * 1000)

    mark(f"READY {unix_ms()} {args.mode}")

    if args.mode == 'cold':
        extract_s, model_input = extract_features()
        torch.manual_seed(args.seed)
        t = time.perf_counter()
        for out in model.tts(**model_input, stream=False, speed=1.0):
            pcm = out['tts_speech']
        synth = time.perf_counter() - t
        report['cold'] = {
            'ready_since_process_start': report['ready_since_process_start'],
            'first_extract_seconds': extract_s,
            'first_synthesize_seconds': synth,
            'first_request_total_seconds': extract_s + synth,
            'pcm_samples': pcm.numel(),
            'pcm_valid': bool(pcm.numel() and torch.isfinite(pcm).all()),
            'audio_seconds': pcm.numel() / 24000.0,
        }
    elif args.mode == 'warm':
        rounds = []
        # Warmup both paths (untimed).
        staged_round(torch, model, prompt, llm_fixture, args.seed, device_str)
        model_input = fixture_input()
        torch.manual_seed(args.seed)
        for out in model.tts(**model_input, stream=False, speed=1.0):
            pass
        for i in range(args.rounds):
            mark(f"ROUND_START {i} {unix_ms()}")
            entry = {'round': i}
            try:
                model_input = fixture_input()
                if args.reextract:
                    entry['extract_seconds'], model_input = extract_features()
                stages = staged_round(torch, model, prompt, llm_fixture, args.seed, device_str)
                entry.update(stages)
                torch.manual_seed(args.seed)
                t = time.perf_counter()
                for out in model.tts(**model_input, stream=False, speed=1.0):
                    pcm = out['tts_speech']
                entry['synthesize_seconds'] = time.perf_counter() - t
                audio = pcm.numel() / 24000.0
                entry['pcm_samples'] = pcm.numel()
                entry['audio_seconds'] = audio
                entry['rtf'] = entry['synthesize_seconds'] / audio
                entry['pcm_valid'] = bool(pcm.numel() and torch.isfinite(pcm).all())
                if i == 0 and args.wav_out:
                    save_wav(pcm, args.wav_out)
            except Exception as exc:  # noqa: BLE001 — record, keep benchmarking
                entry['error'] = repr(exc)
            rounds.append(entry)
            mark(f"ROUND_END {i} {unix_ms()}")
        report['warm_rounds'] = rounds
    elif args.mode == 'stream':
        rounds = []
        if not args.no_warmup:
            model_input = fixture_input()
            torch.manual_seed(args.seed)
            for out in model.tts(**model_input, stream=True, speed=1.0):
                pass
        for i in range(args.rounds):
            mark(f"ROUND_START {i} {unix_ms()}")
            entry = {'round': i}
            try:
                model_input = fixture_input()
                if args.reextract:
                    entry['extract_seconds'], model_input = extract_features()
                stages = staged_round(torch, model, prompt, llm_fixture, args.seed, device_str)
                entry.update(stages)
                torch.manual_seed(args.seed)
                request_start = time.perf_counter()
                chunks = []
                got_final = False
                all_pcm = []
                for out in model.tts(**model_input, stream=True, speed=1.0):
                    pcm = out['tts_speech'].float().flatten()
                    chunks.append({
                        'index': len(chunks),
                        'samples': pcm.numel(),
                        'audio_seconds': pcm.numel() / 24000.0,
                        'arrival_since_request': time.perf_counter() - request_start,
                        'final': None,  # official generator has no final flag
                    })
                    all_pcm.append(pcm)
                total = time.perf_counter() - request_start
                whole = torch.cat(all_pcm) if all_pcm else torch.zeros(0)
                for chunk in chunks:
                    chunk['gen_seconds'] = None
                if not chunks:
                    raise RuntimeError('stream produced no chunks')
                entry.update({
                    'first_pcm_since_request': chunks[0]['arrival_since_request'],
                    'first_pcm_since_session': chunks[0]['arrival_since_request'],
                    'chunks': chunks,
                    'chunk_count': len(chunks),
                    'total_seconds': total,
                    'pcm_samples': whole.numel(),
                    'audio_seconds': whole.numel() / 24000.0,
                    'rtf': total / (whole.numel() / 24000.0),
                    'completed': None,  # official generator exposes no terminal state
                    'hit_token_limit': None,
                    'pcm_valid': bool(whole.numel() and torch.isfinite(whole).all()),
                })
                if i == 0 and args.wav_out:
                    save_wav(whole, args.wav_out)
            except Exception as exc:  # noqa: BLE001
                entry['error'] = repr(exc)
            rounds.append(entry)
            mark(f"ROUND_END {i} {unix_ms()}")
        report['stream_rounds'] = rounds
    elif args.mode == 'components':
        flow_fx = load_file(str(case_dir / 'flow.safetensors'))
        hift_fx = load_file(str(case_dir / 'hift.safetensors'))
        token = flow_fx['token'].to(torch.int32)
        prompt_token = flow_fx['prompt_token']
        prompt_feat = flow_fx['prompt_feat'].float()
        embedding = flow_fx['embedding'].float()
        mel = hift_fx['mel'].float()

        def flow_once():
            t0 = time.perf_counter()
            out, _ = model.flow.inference(
                token=token,
                token_len=torch.tensor([token.shape[1]], dtype=torch.int32),
                prompt_token=prompt_token,
                prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32),
                prompt_feat=prompt_feat,
                prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32),
                embedding=embedding,
                streaming=False,
                finalize=True,
            )
            return time.perf_counter() - t0, out

        def hift_once():
            t0 = time.perf_counter()
            out, _ = model.hift.inference(speech_feat=mel, finalize=True)
            return time.perf_counter() - t0, out

        flow_once()  # warmup
        hift_once()
        flow_rounds = []
        for _ in range(args.rounds):
            seconds, out = flow_once()
            flow_rounds.append({'seconds': seconds, 'mel_frames': out.shape[2]})
        hift_rounds = []
        for _ in range(args.rounds):
            seconds, out = hift_once()
            pcm = out.float().flatten()
            if not bool(torch.isfinite(pcm).all()):
                raise RuntimeError('component hift produced non-finite pcm')
            hift_rounds.append({'seconds': seconds, 'pcm_samples': pcm.numel()})
        report['components'] = {
            'tokens': int(token.shape[1]),
            'prompt_tokens': int(prompt_token.shape[1]),
            'mel_frames_fixture': int(mel.shape[2]),
            'flow': flow_rounds,
            'hift': hift_rounds,
        }
    elif args.mode == 'mps_probe':
        # Official code hardcodes cuda-if-available-else-cpu; moving to MPS
        # is NOT a supported official configuration. This probe documents
        # what happens, for the record only.
        result = {'official_device': str(model.device)}
        try:
            model.llm.to('mps')
            model.flow.to('mps')
            model.hift.to('mps')
            result['modules_moved'] = True
            stages = staged_round(torch, model, prompt, llm_fixture, args.seed, 'mps')
            result['staged_round'] = stages
        except Exception as exc:  # noqa: BLE001
            result['modules_moved'] = False
            result['error'] = repr(exc)
        report['mps_probe'] = result

    report['ru_maxrss_kb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['wall_since_process_start'] = time.perf_counter() - T0

    text_json = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text_json)
    print(text_json)


if __name__ == '__main__':
    main()
