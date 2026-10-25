"""Pinned official ONNX loop, single chunk and official streaming codec; diagnostic."""
import contextlib
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
output = Path(sys.argv[2])
config = request["request"]
seed = config.get("seed", 42)
assert isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2**64
sys.path.insert(0, config["checkout"])
with contextlib.redirect_stdout(sys.stderr):
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf
    import onnx_tts_runtime as official_runtime
    from onnx_tts_runtime import OnnxTtsRuntime
profile_sessions = []
if config.get("profile_sessions", False):
    original_session = ort.InferenceSession

    def profiled_session(path, *args, **kwargs):
        assert isinstance(path, (str, Path)), "profile requires a named model file"
        options = args[0] if args else kwargs.pop("sess_options", None)
        options = options if options is not None else ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(output / f"profile-{len(profile_sessions)}-{Path(path).stem}")
        session = original_session(path, options, *args[1:], **kwargs)
        profile_sessions.append((str(path), session))
        return session

    ort.InferenceSession = profiled_session
codec_budget = config.get("codec_frame_budget_control")
if codec_budget is not None:
    assert isinstance(codec_budget, int) and not isinstance(codec_budget, bool) and 1 <= codec_budget <= 8
    official_runtime._resolve_stream_decode_frame_budget = lambda *_: codec_budget
clock = time.perf_counter()
sequence = 0

def event(kind, payload):
    global sequence
    print(json.dumps(dict(schema="talechime/performance/1", attempt_id=request["attempt_id"],
                          sequence=sequence, offset_seconds=time.perf_counter()-clock,
                          kind=kind, payload=payload)), flush=True)
    sequence += 1

started = time.perf_counter()
with contextlib.redirect_stdout(sys.stderr):
    runtime = OnnxTtsRuntime(config["models"], execution_provider="cuda", thread_count=4,
                             sample_mode="fixed", max_new_frames=int(config.get("max_new_frames", 375)), output_dir=output)
details = dict(ort=ort.__version__, ort_build=ort.get_build_info(),
               profiling_enabled=bool(config.get("profile_sessions", False)),
               providers={name: session.get_providers() for name, session in runtime.sessions.items()},
               load_seconds=time.perf_counter()-started,
               frontend="official synthesize_single_chunk; no top-level normalizer",
               seed=seed, rng=config.get("rng_mode", "numpy_default_rng"), streaming=True,
               codec_frame_budget_control=codec_budget)
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
kernel.GetModuleHandleW.restype = ctypes.c_void_p
kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_ulong]
modules = {}
for name in ["onnxruntime.dll", "onnxruntime_providers_cuda.dll", "cublas64_13.dll",
             "cublasLt64_13.dll", "cudart64_13.dll", "cudnn64_9.dll"]:
    handle = kernel.GetModuleHandleW(name)
    buffer = ctypes.create_unicode_buffer(32768)
    if handle and kernel.GetModuleFileNameW(handle, buffer, len(buffer)):
        path = Path(buffer.value)
        modules[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
details["loaded_modules"] = modules
(output / "runtime.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
event("prepared", details)
prompt = runtime.resolve_prompt_audio_codes(voice="Weiguo", prompt_audio_path=None)
original_frame = runtime.run_local_fixed_sampled_frame
original_decode = runtime.codec_streaming_session.run_frames
original_builder = runtime.build_voice_clone_request_rows
state = {}

def builder(*args, **kwargs):
    rows = original_builder(*args, **kwargs)
    if config.get("prepend_im_start_control"):
        row = [int(runtime.manifest["tts_config"]["im_start_token_id"])] + [1024]*16
        rows["inputIds"].insert(0, row)
        rows["attentionMask"][0].insert(0, 1)
    state["prompt_rows"] = rows
    return rows

def frame(*args, **kwargs):
    result = original_frame(*args, **kwargs)
    state["last_continue"] = bool(result[0])
    return result

def decode(*args, **kwargs):
    state.setdefault("codec_chunks", []).append(len(args[0]))
    result = original_decode(*args, **kwargs)
    if result is not None and result[1] > 0 and "first" not in state:
        state["first"] = time.perf_counter()
    return result

class LcgControl:
    def __init__(self):
        self.state = seed & 0xffffffff

    def random(self):
        self.state = (self.state * 1664525 + 1013904223) & 0xffffffff
        return (self.state >> 8) / 16777216.0

runtime.run_local_fixed_sampled_frame = frame
runtime.codec_streaming_session.run_frames = decode
runtime.build_voice_clone_request_rows = builder
for index in range(request["warmup_rounds"] + request["hot_rounds"]):
    state.clear()
    runtime.rng = LcgControl() if config.get("rng_mode") == "lcg_control" else np.random.default_rng(seed)
    started = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        result = runtime.synthesize_single_chunk(text=config["text"], prompt_audio_codes=prompt,
                                                 streaming=True)
    elapsed = time.perf_counter()-started
    assert state.get("last_continue") is False, "no observed graph stop before frame budget"
    pcm = np.asarray(result["waveform"], dtype="<f4")
    assert pcm.ndim == 2 and pcm.shape[1] == 2 and pcm.size and np.isfinite(pcm).all()
    assert "first" in state
    (output / f"round-{index}-prompt-rows.json").write_text(json.dumps(state["prompt_rows"]), encoding="utf-8")
    (output / f"round-{index}-completion.json").write_text(json.dumps(dict(
        last_continue=state["last_continue"], code_frames=len(result["generated_frames"]),
        rng_mode=config.get("rng_mode", "numpy_default_rng"),
        codec_chunks=state.get("codec_chunks", []))), encoding="utf-8")
    path = output / f"round-{index}-f32.wav"
    sf.write(path, pcm, 48000, subtype="FLOAT")
    (output / f"round-{index}-codes.json").write_text(json.dumps(result["generated_frames"]), encoding="utf-8")
    event("round", dict(index=index, warmup=index < request["warmup_rounds"], outcome="success",
                        synthesis_seconds=elapsed, first_pcm_seconds=state["first"]-started,
                        audio_seconds=pcm.shape[0]/48000,
                        delivery_boundary="segment_delivery_first_pcm",
                        output_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(), observations=[],
                        artifacts=[dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                        bytes=path.stat().st_size, kind="audio_f32", derived_from=None)]))
if profile_sessions:
    profiles = []
    for model_path, session in profile_sessions:
        path = Path(session.end_profiling())
        profiles.append(dict(model=model_path, path=path.relative_to(output).as_posix(),
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             bytes=path.stat().st_size, kind="diagnostic_ort_profile", derived_from=None))
    (output / "profile-index.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    event("observation", dict(kind="ort_node_profiles_not_performance_baseline", profiles=profiles))
event("terminal", dict(outcome="success"))
