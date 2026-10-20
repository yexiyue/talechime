# Spec Delta

## Purpose

Define consistent performance observations across backend implementations while preserving request ownership and normal streaming behavior. Raw observations, aggregated metrics and diagnostic traces serve different purposes and must remain correlated without treating host submission time as GPU execution time.

## ADDED Requirements

### Requirement: Defined measurement boundaries
Every measurement SHALL declare its unit, start and end boundaries, observation location and inclusion relationship. Verification, loading, reference preparation, first request, hot request and completion MUST remain distinguishable. Overlapping stages MUST NOT be summed as independent elapsed time; unavailable boundaries MUST have reasons.

#### Scenario: Reference encoding is lazy
- **WHEN** reference encoding occurs within the first request and cannot be isolated accurately
- **THEN** the observation discloses that inclusion and does not count it again as an independent hot-request cost

### Requirement: Observable delivery and backpressure
The system SHALL distinguish owner-produced PCM, adapter delivery and consumer receipt where observable, identifying native chunk or segment delivery. Chunk gaps, bounded-send waiting and queue audio duration MUST declare their locations. Audio export I/O MUST remain outside synthesis timing, and generation MUST NOT imply actual playback completion.

#### Scenario: Consumer causes channel waiting
- **WHEN** the bounded PCM channel delays delivery after generation
- **THEN** measured channel waiting is distinguishable from inference work and is not presented as GPU compute time

### Requirement: Correlated bounded diagnostics
Diagnostic records SHALL correlate run, attempt, request and applicable segment/chunk across async tasks and owner threads without transferring model or audio ownership. Trace buffering MUST be bounded, and dropped records, incomplete flush or abnormal termination MUST be disclosed as incomplete diagnostics.

#### Scenario: An owner thread emits a stage observation
- **WHEN** request execution moves from a local task to its model owner thread
- **THEN** observations retain explicit request correlation without extending an entered async span across suspension

#### Scenario: Trace writer cannot keep up
- **WHEN** detailed diagnostic output exceeds its configured buffer
- **THEN** the attempt records the loss policy and dropped-record count or an explicit unknown count, and does not claim a complete trace

### Requirement: Separate aggregation and raw evidence
Aggregated counters, gauges and histograms SHALL have bounded label sets and explicit units. Unique run, request, text and voice identities MUST remain in raw evidence or trace context rather than unbounded metric labels. Aggregates MUST NOT replace per-round authoritative results, and each attempt MUST avoid inheriting earlier measurements.

#### Scenario: Two attempts run in one orchestration session
- **WHEN** aggregation is collected for a new isolated attempt
- **THEN** its snapshot excludes prior-attempt observations and retains links to its own raw evidence

### Requirement: Isolated observation modes
Baseline, detailed tracing and native profiling runs SHALL carry distinct observation identities. Official baseline statistics MUST exclude diagnostic runs. Observation overhead MUST be evaluated with matched output conditions, reporting measured overhead or an explicit unverified state rather than assuming instrumentation is free.

#### Scenario: Compare a traced run to a baseline
- **WHEN** a detailed trace changes the collection configuration
- **THEN** it remains diagnostic evidence and is not pooled into baseline timing statistics

### Requirement: Honest asynchronous device timing
Request timing SHALL include completion of the relevant device work at its declared boundary. Host spans MUST identify wall or submission timing; device execution time requires device-native evidence. Internal stage instrumentation MUST NOT force synchronization at every span or use an unrelated context as proof of completion.

#### Scenario: GPU launches return before execution completes
- **WHEN** a host stage finishes submitting asynchronous device operations
- **THEN** its duration is not labelled kernel execution time and request completion uses the owning execution path's verified boundary

### Requirement: Host-controlled telemetry installation
Reusable libraries SHALL leave diagnostic subscriber and metric recorder installation to the calling executable or host. Normal product execution MUST retain its existing protocol output and inference behavior, and performance diagnostics MUST NOT emit non-protocol data to worker stdout.

#### Scenario: Embed a backend in an instrumented host
- **WHEN** a host already owns telemetry configuration
- **THEN** backend observations use that configuration without replacing global handlers or starting an exporter service
