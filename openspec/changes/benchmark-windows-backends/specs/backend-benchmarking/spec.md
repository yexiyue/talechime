# Spec Delta

## Purpose

为当前 Windows TTS 后端提供可复现、可追溯且可恢复的 CPU 与 CUDA 性能测量，并以原始数据、图表和离线试听报告交付结果，让性能观察、生成成功与人工听感确认能够分别核查。

## ADDED Requirements

### Requirement: Explicit model and device coverage
The suite SHALL enumerate all current public TTS variants of MOSS, Qwen, VoxCPM, OmniVoice, CosyVoice3 and IndexTTS on Windows CPU/CUDA, including experimental variants. Auxiliary voice generation SHALL be listed separately. Unsupported combinations MUST be marked not applicable without adding device support or silently falling back.

#### Scenario: GPU-only model requested on CPU
- **WHEN** the matrix includes MOSS Local/Realtime or VoxCPM original BF16 on CPU and the product does not expose that combination
- **THEN** the report retains the combination as not applicable with its reason, without measuring a library-only CPU path as product performance

#### Scenario: ONNX GPU provider is unavailable
- **WHEN** Nano ONNX CUDA fails provider execution or lacks compatible kernels
- **THEN** that row records the actual failure and evidence, and any CPU measurement is a separate row rather than a GPU success

### Requirement: Fixed attributable measurement cases
Every case SHALL record source/binary/model/manifest/reference/corpus identities, actual device and provider, feature set, precision, threads, seed, effective parameters and generation mode. Short and multi-segment workloads SHALL be included. Supported full, streaming and continuation modes MUST be distinguished without adding unsupported product modes.

#### Scenario: Different voice capabilities
- **WHEN** Base cloning, CustomVoice or VoiceDesign are benchmarked
- **THEN** each case records its actual reference, preset voice or description and sampling settings, and the report does not represent them as identical voice conditioning

### Requirement: Repeated serial performance measurements
Formal baseline cases SHALL use at least three independent processes, each with one excluded warmup and five measured hot rounds. Release binaries MUST be built before serial measurement. Each round and process mean SHALL remain available, with mean-of-process-means, ranges and successful/expected sample counts; incomplete cases MUST NOT be labelled complete baselines.

#### Scenario: One round fails
- **WHEN** a process returns an error, timeout, cancellation or truncated output during the required rounds
- **THEN** raw completed rounds and failure evidence remain available, the case is labelled incomplete, and successful-only statistics disclose their reduced coverage

### Requirement: Honest timing and memory boundaries
The suite SHALL distinguish verification, model loading, reference preparation, first request, hot synthesis, first PCM, output duration, chunks and request completion. RTF MUST use actual output duration and disclosed timing boundaries. Device synchronization SHALL bound asynchronous timings. Export I/O MUST be excluded from synthesis timing; unavailable measurements MUST remain null with reasons.

#### Scenario: Non-streaming output
- **WHEN** a model delivers only an entire semantic segment or request
- **THEN** its first PCM is labelled with that delivery boundary and is not presented as native streaming latency

#### Scenario: GPU memory sampling
- **WHEN** memory measurements include whole-device VRAM or process RSS
- **THEN** they retain distinct names and sampling scope, whole-device memory is not claimed as process-exclusive VRAM, and unobserved values are not replaced with zero

### Requirement: Recoverable execution and terminal evidence
The runner SHALL preserve versioned manifests, per-attempt logs and terminal outcomes in persistent run directories. Resume MUST verify compatible identities before reusing completed cases and MUST preserve failed attempts. Cases SHALL support bounded execution, cancellation and process cleanup without overlapping inference jobs or leaving a model owner active.

#### Scenario: Resume a partially completed suite
- **WHEN** a run is resumed after interruption
- **THEN** compatible completed cases are retained, incompatible inputs require a new case/run identity, and unfinished cases are rerun without overwriting earlier evidence

### Requirement: Portable offline listening report
The suite SHALL deliver persistent raw JSON/CSV, charts and an offline HTML report with relative local assets. Every successful model/device/workload SHALL have traceable complete and per-segment WAVs, text, generation parameters and audio digests. All measured-round audio MUST remain available, and representative report samples MUST be chosen deterministically without speed-based selection.

#### Scenario: Open an exported report offline
- **WHEN** the report directory is copied or unpacked on another computer without model files or network access
- **THEN** its tables, charts and audio players work from local assets, with no dependency on target or the originating absolute home path

### Requirement: Separate coverage, output checks and listening acceptance
Reports MUST retain success, blocked, failure, timeout and not-applicable rows, disclose coverage, and distinguish valid finite PCM/natural completion from content or listening acceptance. No cross-model speed ranking SHALL imply equivalent quality, content or output. Generated audio, user references and model files MUST NOT enter repository history.

#### Scenario: Valid generation awaiting user listening
- **WHEN** a case finishes with valid PCM and a successful generation terminal status
- **THEN** it can be recorded as measured while listening remains pending, without asserting actual playback completion, absence of omitted words or long-duration stability
