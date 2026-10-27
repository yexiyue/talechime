## ADDED Requirements

### Requirement: Official native acoustic inference
The system SHALL synthesize 24 kHz mono audio using the pinned official Kokoro v1.1-zh weights without Python at runtime, preserving the official token, style, duration and excitation semantics.

#### Scenario: Independent fixed-input comparison
- **WHEN** official and native implementations receive identical IDs, style, speed and excitation random tensors
- **THEN** integer durations and output shapes match and independent stage/PCM errors are recorded against explicit numeric gates

#### Scenario: Invalid or excessive input
- **WHEN** IDs, style, speed, resource format or output budget is invalid
- **THEN** synthesis fails explicitly without overwriting data or returning successful completion

### Requirement: Complete Mandarin frontend
The system SHALL provide Mandarin 1.1 pronunciation using pinned official normalization, dictionaries, segmentation, sandhi, erhua and token mapping, and SHALL preserve all source ranges when splitting long input.

#### Scenario: Long text
- **WHEN** the phoneme budget is exceeded
- **THEN** input is split into complete bounded segments with immutable original UTF-8 ranges and no silent truncation

#### Scenario: Unsupported symbols or language
- **WHEN** input cannot be pronounced by the supported frontend
- **THEN** the limitation is explicit and spoken content is not silently discarded

### Requirement: Optional backend lifecycle
The system SHALL expose Kokoro through explicit features and registry preparation using current resource paths, accurate built-in voice/parameter/device capabilities, dedicated inference-thread ownership and bounded cancellable audio delivery.

#### Scenario: Cancellation and reuse
- **WHEN** synthesis or model loading is cancelled
- **THEN** the owner joins/releases its work and a later valid request can complete without stale PCM or false completion

### Requirement: Independent validation
Ordinary tests SHALL NOT download models. Real model numeric, audio and performance evidence SHALL identify actual source/weight revisions, device, precision, randomness and timing boundaries; untested devices SHALL NOT be reported as accepted.

#### Scenario: Release verification
- **WHEN** a device path is declared verified
- **THEN** its evidence includes independent official comparison and real runtime checks appropriate to that path

### Requirement: Separate contextual pronunciation improvement
The system SHALL retain faithful official pronunciation as a baseline and SHALL expose contextual disambiguation improvements through an explicit optional path, with independent pronunciation improvement/regression evidence.

#### Scenario: Contextual correction
- **WHEN** the contextual path is explicitly selected for an ambiguous Mandarin character
- **THEN** it uses independently verified contextual predictions, preserves source-character mapping and applies the existing sandhi/token mapping after the correction

#### Scenario: Official comparison
- **WHEN** the faithful official path is selected
- **THEN** contextual corrections do not alter its numeric or frontend comparison inputs
