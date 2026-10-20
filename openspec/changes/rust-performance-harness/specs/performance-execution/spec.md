# Spec Delta

## Purpose

Provide a reusable Rust-driven performance workflow whose plans, isolated attempts and offline evidence remain attributable across machines. Windows execution is the initial implementation; macOS extension points and deferred capability states preserve future CPU and Metal measurement without claiming unverified support.

## ADDED Requirements

### Requirement: Python-independent measurement workflow
The developer performance entry point SHALL plan, build, execute, resume, compare and render reports without a Python runtime. Any upstream conversion dependency MUST be confined to explicit preparation and recorded with its provenance. Measurement MUST NOT acquire or convert resources.

#### Scenario: Run with already prepared resources
- **WHEN** a developer runs a performance plan without Python installed
- **THEN** measurement, aggregation and offline report generation work using prepared resources

#### Scenario: Conversion requires an upstream tool
- **WHEN** a selected resource requires a Python-only converter
- **THEN** preparation reports the explicit dependency and provenance, while measurement refuses the unprepared resource without launching the converter

### Requirement: Platform-scoped frozen plans
Plans SHALL identify operating system, architecture, machine, execution provider, effective features, precision, threads, workload, conditioning and repetition policy. Every selected combination MUST retain a capability state and reason; missing implementation, missing build features and unsupported product combinations MUST remain distinguishable without device fallback.

#### Scenario: Plan future macOS execution
- **WHEN** a plan includes macOS CPU or Metal for which native support has not been implemented or verified
- **THEN** the plan retains those rows as deferred or unverified with reasons, without reporting Windows evidence as Mac success

#### Scenario: CPU math implementation differs
- **WHEN** two CPU runs use different effective math implementations such as Accelerate and a pure Rust path
- **THEN** their environment records retain that distinction for reporting and comparison

### Requirement: Isolated bounded attempts
Model measurements SHALL execute serially in isolated processes built for explicit feature sets. Attempts MUST have timeouts, cancellation, bounded output handling and verified child cleanup before another measurement starts. Compilation MUST finish before the timed measurement batch.

#### Scenario: Timed-out child retains descendants
- **WHEN** an attempt exceeds its budget and child cleanup cannot be confirmed
- **THEN** the runner preserves timeout and cleanup evidence and refuses to start the next inference attempt

### Requirement: Attributable resumable evidence
Each run SHALL retain versioned plan, case and attempt identities, source and binary identities, resource receipts, raw rounds and terminal outcomes. Resume MUST reuse only compatible complete evidence and MUST preserve previous attempts. New versions or incompatible records MUST be rejected explicitly rather than interpreted silently.

#### Scenario: Binary changes before resume
- **WHEN** a developer resumes with a different measurement binary or missing recorded artifacts
- **THEN** the old attempt remains intact and is not reused as matching completed evidence

### Requirement: Portable offline reports
Reports SHALL contain raw JSON/CSV, locally rendered charts, coverage, missing-data reasons and traceable audio links without Python, network services or the original machine path. Generation, actual playback and listening acceptance MUST remain separate. Outputs MUST stay outside version control and exports MUST exclude models and unnecessary reference originals.

#### Scenario: Move a report to another machine
- **WHEN** a completed report directory is copied to a different path and opened offline
- **THEN** tables, charts and supplied audio work from local assets, while deferred and failed combinations remain visible

### Requirement: Explicit platform measurement scope
Resource measurements SHALL declare unit, process or device scope, sampling interval and source. Unsupported readings MUST be null with reasons. Shared memory and whole-device readings MUST NOT be presented as exclusive process GPU allocation, and sampled maxima MUST be distinguished from operating-system high-water marks.

#### Scenario: macOS unified memory collector is absent
- **WHEN** a future Mac attempt has no verified native GPU memory collector
- **THEN** GPU allocation is unavailable with a reason, and neither RSS nor total unified memory is relabelled as exclusive VRAM
