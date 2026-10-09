"""Direct ORT feasibility probe, using kaldi-native-fbank as a reference frontend.

Not a Rust implementation. Frontend/LFR/CMVN contracts were inspected at
k2-fsa/sherpa-onnx revision 9db1af1871ed78cee9357c23898e5a7d8702d21a.
No sherpa recognizer or C++ ASR decoding runtime is used by this adapter.
"""

import math
import re


def recognizer(path, backend, threads):
    import kaldi_native_fbank as knf
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path / "model.int8.onnx"), options, providers=["CPUExecutionProvider"]
    )
    metadata = session.get_modelmeta().custom_metadata_map
    tokens = {}
    for line in (path / "tokens.txt").read_text(encoding="utf-8").splitlines():
        token, identity = line.rsplit(" ", 1)
        tokens[int(identity)] = token
    sensevoice = backend == "sensevoice-ort"

    def recognize(audio):
        config = knf.FbankOptions()
        config.frame_opts.dither = 0
        config.frame_opts.snip_edges = True
        config.frame_opts.window_type = "hamming" if sensevoice else "povey"
        config.mel_opts.num_bins = 80
        config.mel_opts.high_freq = 0
        fbank = knf.OnlineFbank(config)
        scaled = audio * 32768
        fbank.accept_waveform(16000, scaled.tolist())
        fbank.input_finished()
        features = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
        if sensevoice:
            width, shift = (
                int(metadata["lfr_window_size"]),
                int(metadata["lfr_window_shift"]),
            )
            centers = np.arange(math.ceil(len(features) / shift))[:, None] * shift
            indices = np.clip(
                centers + np.arange(width) - (width - 1) // 2, 0, len(features) - 1
            )
            features = features[indices].reshape(len(indices), -1)
            features = features + np.fromstring(
                metadata["neg_mean"], sep=",", dtype=np.float32
            )
            features *= np.fromstring(metadata["inv_stddev"], sep=",", dtype=np.float32)
            inputs = {
                "x": features[None].astype(np.float32),
                "x_length": np.array([len(features)], dtype=np.int32),
                "language": np.array([int(metadata["lang_zh"])], dtype=np.int32),
                "text_norm": np.array([int(metadata["without_itn"])], dtype=np.int32),
            }
        else:
            features -= np.fromstring(metadata["cmvn_mean"], sep=",", dtype=np.float32)
            features *= np.fromstring(
                metadata["cmvn_inv_stddev"], sep=",", dtype=np.float32
            )
            names = [value.name for value in session.get_inputs()]
            length_type = (
                np.int64
                if session.get_inputs()[1].type == "tensor(int64)"
                else np.int32
            )
            inputs = {
                names[0]: features[None].astype(np.float32),
                names[1]: np.array([len(features)], dtype=length_type),
            }
        output = session.run(None, inputs)
        logits = output[0][0]
        if not sensevoice and len(output) > 1:
            logits = logits[: int(output[1].reshape(-1)[0])]
        predicted = logits.argmax(axis=-1)
        identities = predicted[np.r_[True, predicted[1:] != predicted[:-1]]]
        blank = int(metadata.get("blank_id", 0))
        pieces = [tokens[int(identity)] for identity in identities if identity != blank]
        text = "".join(pieces).replace("▁", " ")
        text = re.sub(r"<\|.*?\|>", "", text).strip()
        return text, {"raw_tokens": pieces, "onnxruntime_version": ort.__version__}

    return recognize
