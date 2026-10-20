# Spec Delta

## Purpose

Provide macOS native execution and attributable resource observations for the shared Rust performance tools. Tool readiness and short production-path smoke evidence remain separate from full CPU/Metal performance baselines, observation overhead acceptance and listening results.

## ADDED Requirements

### Requirement: Contained macOS attempts
Mac attempts SHALL run serially within an owned process group with bounded output, timeout and verified cleanup. Cleanup MUST cover supported probe descendants after normal exit, failure or cancellation, preserve unrelated processes and refuse further inference when cleanup cannot be confirmed. Containment limitations MUST be explicit.

#### Scenario: Parent exits while a descendant holds pipes
- **WHEN** the probe parent exits but a child or grandchild still holds an output pipe
- **THEN** cleanup terminates owned descendants within its budget, readers finish and cleanup evidence is retained without waiting indefinitely

#### Scenario: Cleanup is uncertain
- **WHEN** remaining descendants cannot be ruled out or a descendant leaves supported containment
- **THEN** the attempt is not committed as successful and the runner stops scheduling further inference with an explicit cleanup failure

#### Scenario: Another process runs alongside the probe
- **WHEN** the runner cleans up its attempt
- **THEN** unrelated processes remain alive and are not signalled by name or broad PID matching

### Requirement: macOS CLI cancellation
The Mac CLI SHALL treat SIGINT and SIGTERM as cancellation requests, retain partial evidence, perform bounded owned-process cleanup and exit unsuccessfully. A cancelled batch MUST NOT start another attempt or publish a success commit for the interrupted attempt.

#### Scenario: Cancel the runner during inference
- **WHEN** SIGINT or SIGTERM reaches the runner while a probe is active
- **THEN** the interrupted attempt records cancellation and cleanup state, pending attempts are not launched and partial evidence remains available

### Requirement: Native process resource readings
Mac observations SHALL distinguish current process RSS, sampled maxima and available OS high-water marks, each with source, unit, scope, interval and status. Unsupported or failed readings MUST be null with reasons. A short-lived process MUST NOT be represented as using zero memory because sampling missed it.

#### Scenario: High-water metric is unavailable
- **WHEN** only periodic RSS readings can be collected
- **THEN** the report labels their maximum as sampled RSS and retains a separate unavailable OS-peak reading with its reason

### Requirement: Attributable Mac execution conditions
Mac runs SHALL freeze and verify machine, OS/build, architecture, CPU/device identity, effective build features, math implementation and supported thread controls. Requested CPU or Metal execution MUST NOT silently fall back. Unknown effective conditions MUST block comparable real-model execution rather than become verified labels.

#### Scenario: Accelerate is compiled but operator use is not measured
- **WHEN** build evidence identifies Accelerate support without per-operator device evidence
- **THEN** conditions describe the build/math implementation without claiming every operator used AMX or the requested thread count was observed

#### Scenario: Execution identity changes before resume
- **WHEN** OS, provider, effective math, thread controls or binary identity differs from the frozen run
- **THEN** resume rejects reuse and preserves earlier evidence

### Requirement: Honest Metal completion and memory scope
Metal request observations SHALL identify completion of delivered host PCM through the owning production path, distinct from host submission, kernel duration and playback. Memory readings MUST identify their actual process/device scope; RSS or shared unified memory MUST NOT be labelled exclusive VRAM. Missing native GPU metrics MUST remain unavailable without blocking an otherwise valid request.

#### Scenario: PCM reaches the consumer
- **WHEN** a Metal probe drains finite PCM and an explicit successful stream terminal event
- **THEN** request output completion is recorded without claiming device-wide idle, kernel timings or actual playback completion

### Requirement: Evidence-based Mac capability activation
Mac plans SHALL preserve all selected model/device rows and distinguish unsupported combinations, missing features/resources, unverified models and verified tool capabilities. Native tool verification MUST NOT globally mark CPU or Metal model rows as performance-validated; Instruments MAY remain deferred without blocking ordinary tool execution.

#### Scenario: Tool lifecycle passes but a model is absent
- **WHEN** Mac native execution is verified while a selected model has no prepared resources
- **THEN** the tool capability is available but the model row retains its explicit resource or unverified state

### Requirement: Separate tool readiness evidence
Mac tool acceptance SHALL include a Python-independent synthetic build/run/resume/compare/report/pack flow and short production-path CPU and Metal smoke using existing validated resources. Missing resources or device access MUST be recorded as blocked smoke, never success. Full matrix baselines, real-model overhead acceptance and listening acceptance MUST remain separate work.

#### Scenario: Synthetic workflow completes
- **WHEN** deterministic fixtures run on Mac without a Python runtime in the measurement path
- **THEN** event integrity, artifacts, compatible resume, comparison and portable report assets are verified without presenting synthetic timings as model performance

#### Scenario: Prepared model smoke completes
- **WHEN** a short request succeeds through a production CPU or Metal adapter
- **THEN** evidence records resource/binary identity, actual provider, finite nonempty PCM and terminal outcome, labelled wiring smoke rather than a stable baseline or listening acceptance

#### Scenario: No suitable Metal resource exists
- **WHEN** no already-prepared production-validated model can exercise Metal
- **THEN** the Metal smoke remains blocked with a concrete reason, no download or conversion is launched implicitly and readiness is reported as partial
