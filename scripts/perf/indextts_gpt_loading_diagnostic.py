"""CPU checkpoint loading control; no TTS timing or production loader changes."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    envelope = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output = Path(sys.argv[2])
    config = envelope["request"]
    sys.path.insert(0, config["checkout"])
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from omegaconf import OmegaConf
        from indextts.gpt.model_v2 import UnifiedVoice
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    sequence = 0
    clock = time.perf_counter()

    def event(kind, payload):
        nonlocal sequence
        print(json.dumps(dict(schema="talechime/performance/1", attempt_id=envelope["attempt_id"],
                              sequence=sequence, offset_seconds=time.perf_counter()-clock,
                              kind=kind, payload=payload)), flush=True)
        sequence += 1

    def bits(tensor):
        return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)

    def signature(tensor):
        return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                    sha256=hashlib.sha256(bits(tensor).numpy().tobytes()).hexdigest())

    event("prepared", dict(kind="GPT_loading_only_not_TTS", device="cpu", checkpoint_backend="torch_mmap"))
    cfg = OmegaConf.load(config["config"])
    torch.manual_seed(42)
    meta = config.get("meta_assign", False)
    with contextlib.redirect_stdout(sys.stderr), torch.device("meta" if meta else "cpu"):
        model = UnifiedVoice(**cfg.gpt, use_accel=False, spk_cond_mode="campplus")
    # Record nonpersistent buffers too; load_state_dict must leave these intact.
    if meta:
        oracle = json.loads(Path(config["ordinary_state_audit"]).read_text(encoding="utf-8"))
        before_buffers = oracle["summary"]["nonpersistent_buffers"]
    else:
        before_buffers = {key: signature(value) for key, value in model.named_buffers()}
    before_aliases = [[a, b] for a, pa in model.named_parameters(remove_duplicate=False)
                      for b, pb in model.named_parameters(remove_duplicate=False)
                      if a < b and pa is pb]
    checkpoint = torch.load(config["checkpoint"], map_location="cpu", mmap=True, weights_only=True)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False, assign=meta)
    if meta:
        for key, value in list(model.named_buffers()):
            if not value.is_meta:
                continue
            expected = before_buffers[key]
            assert list(value.shape) == expected["shape"] and str(value.dtype) == expected["dtype"], key
            if key.endswith(".attn.bias") and value.dtype == torch.bool:
                replacement = torch.tril(torch.ones(value.shape[-2:], dtype=torch.bool)).view(value.shape)
            elif key.endswith(".attn.masked_bias") and value.ndim == 0:
                replacement = torch.tensor(-1e4, dtype=value.dtype)
            else:
                raise AssertionError(f"unhandled meta buffer: {key}")
            assert signature(replacement) == expected, key
            prefix, leaf = key.rsplit(".", 1)
            model.get_submodule(prefix)._buffers[leaf] = replacement
        assert not any(t.is_meta for t in list(model.parameters()) + list(model.buffers()))
    actual = model.state_dict()
    assert not missing, f"uninitialized checkpoint keys: {missing}"
    audited = []
    for key, value in actual.items():
        source = state[key]
        assert value.dtype == source.dtype and value.shape == source.shape, key
        assert torch.equal(bits(value), bits(source)), key
        audited.append(dict(key=key, **signature(value)))
    after_aliases = [[a, b] for a, pa in model.named_parameters(remove_duplicate=False)
                     for b, pb in model.named_parameters(remove_duplicate=False)
                     if a < b and pa is pb]
    assert before_aliases == after_aliases
    nonpersistent = {}
    for key, value in model.named_buffers():
        if key not in actual:
            nonpersistent[key] = signature(value)
            assert nonpersistent[key] == before_buffers[key], key
    result = dict(kind="meta_assign_GPT_state_audit_not_full_TTS" if meta else "mmap_copy_GPT_state_audit_not_full_TTS", torch=torch.__version__,
                  model_state_tensors=len(actual), checkpoint_state_tensors=len(state),
                  missing=list(missing), unexpected=list(unexpected), all_loaded_state_bitwise_exact=True,
                  aliases=after_aliases, nonpersistent_buffers=nonpersistent,
                  interpretation="meta/assign with audited nonpersistent buffers" if meta else "ordinary CPU construction and copy preserved; mmap changes checkpoint backing only")
    (output / "state-audit.json").write_text(json.dumps(dict(summary=result, tensors=audited), indent=2), encoding="utf-8")
    path = output / "state-audit.json"
    artifact = dict(path=path.name, kind="diagnostic_state", bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), derived_from=None)
    event("observation", dict(diagnostic_result=dict(summary=result, artifacts=[artifact])))
    event("terminal", dict(outcome="success"))


if __name__ == "__main__":
    main()
