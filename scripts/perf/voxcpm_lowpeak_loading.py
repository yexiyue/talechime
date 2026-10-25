"""Experimental loading-only control; pinned official model classes remain unchanged."""
import contextlib, json, sys, gc
from pathlib import Path

def load(model_path, output):
    import torch
    from safetensors import safe_open
    from transformers import LlamaTokenizerFast
    from voxcpm.model.voxcpm2 import VoxCPMConfig, VoxCPM2Model, AudioVAEV2
    path = Path(model_path)
    config = VoxCPMConfig.model_validate_json((path / 'config.json').read_text(encoding='utf-8'))
    assert config.dtype == 'bfloat16' and config.max_length == 8192
    tokenizer = LlamaTokenizerFast.from_pretrained(path, local_files_only=True)
    print('loading-control: open pread before materializing weights', file=sys.stderr, flush=True)
    with safe_open(path / 'model.safetensors', framework='pt', device='cpu', backend='pread') as reader:
        vae = AudioVAEV2(config=config.audio_vae_config) if config.audio_vae_config else AudioVAEV2()
        with torch.device('meta'):
            model = VoxCPM2Model(config, tokenizer, vae, device='cuda')
        expected = {k: v for k, v in model.state_dict().items() if not k.startswith('audio_vae.')}
        assert set(reader.keys()) == set(expected), 'main checkpoint key mismatch'
        parameter_count = 0
        for key in reader.keys():
            source = reader.get_tensor(key)
            assert source.dtype == torch.bfloat16 and source.shape == expected[key].shape, key
            prefix, leaf = key.rsplit('.', 1)
            module = model.get_submodule(prefix)
            assert leaf in module._parameters, key
            requires_grad = module._parameters[leaf].requires_grad
            target = source.to('cuda')
            assert torch.equal(target.cpu(), source), key
            module._parameters[leaf] = torch.nn.Parameter(target, requires_grad=requires_grad)
            parameter_count += 1
            del target, source
            if parameter_count % 100 == 0:
                print(f'loading-control: {parameter_count} main tensors verified', file=sys.stderr, flush=True)
        assert parameter_count == 577
    del expected
    rope_count = 0
    buffer_count = 0
    for name, module in list(model.named_modules()):
        if type(module).__name__ != 'MiniCPMLongRoPE':
            continue
        replacement = type(module)(module.config).to(dtype=torch.bfloat16)
        expected_buffers = {k: v.clone() for k, v in replacement.named_buffers()}
        replacement = replacement.to('cuda')
        for key, v in replacement.named_buffers():
            assert torch.equal(v.cpu(), expected_buffers[key]), name + '.' + key
            buffer_count += 1
        prefix, leaf = name.rsplit('.', 1)
        setattr(model.get_submodule(prefix), leaf, replacement)
        rope_count += 1
    assert rope_count == 3 and buffer_count == 9
    print('loading-control: load original F32 VAE', file=sys.stderr, flush=True)
    state = torch.load(path / 'audiovae.pth', map_location='cpu', weights_only=True)
    state = state.get('state_dict', state)
    vae.load_state_dict(state, strict=True)
    vae = vae.to('cuda')
    actual = vae.state_dict()
    for key, v in state.items():
        assert actual[key].dtype == v.dtype and torch.equal(actual[key].cpu(), v), key
    assert len(state) == 312
    del state, actual
    gc.collect()
    assert all((not x.is_meta for x in model.parameters()))
    assert all((not x.is_meta for x in model.buffers()))
    assert all((p.dtype == torch.bfloat16 for n, p in model.named_parameters() if not n.startswith('audio_vae.')))
    assert all((p.dtype == torch.float32 for p in model.audio_vae.parameters()))
    model.eval()
    torch.cuda.synchronize()
    receipt = dict(kind='experimental_loading_only_official_math', main_tensors_exact=parameter_count, vae_state_tensors_exact=312, nonpersistent_rope_buffers_exact=buffer_count, max_length=config.max_length, main_dtype='bfloat16', codec_dtype='float32', checkpoint_reader='safetensors0.8 pread; opened before weight materialization', constructor='meta main modules; real original VAE', no_missing_or_unexpected_keys=True, default_from_local_not_used=True)
    (output / 'loading-control.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return model
