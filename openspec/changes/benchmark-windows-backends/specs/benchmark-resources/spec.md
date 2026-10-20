# Spec Delta

## Purpose

让开发基准和后续人工试听能够复用具有可追溯身份的本地模型、转换资源和参考音色，在清理构建目录后仍能运行，且不会覆盖现有资源或使历史测量失去来源依据。

## ADDED Requirements

### Requirement: Persistent identity-bound resources
The benchmark resource workflow SHALL preserve runtime model files, required conversion inputs and reference voices under the resolved Talechime user root, with model/revision identities, file sizes, SHA-256 digests and source/conversion receipts. Existing application resource layouts SHALL remain compatible.

#### Scenario: Import an existing development model
- **WHEN** a required model is found in target, a legacy cache or an explicitly identified external checkpoint directory
- **THEN** the workflow records its identity and verified files under the Talechime resource root and the benchmark no longer depends on the source directory for execution

#### Scenario: Reuse after cleaning build output
- **WHEN** target is removed after a completed resource import
- **THEN** a subsequent benchmark can reuse the archived resources without downloading those files again

### Requirement: Copy-only conflict-safe publication
The resource workflow MUST preserve source files and existing destination data, validate staged files before atomic no-clobber publication, and serialize writes to the same resource identity. A conflicting destination MUST be reported without replacing, renaming or deleting its data.

#### Scenario: Matching destination already exists
- **WHEN** source and destination files match the required identity and digest
- **THEN** the workflow reuses the destination and records the reuse without duplicating or modifying its contents

#### Scenario: Conflicting destination or interrupted copy
- **WHEN** a destination differs, a concurrent importer wins publication, or an import is interrupted
- **THEN** existing destination and source data remain intact, incomplete staging is not accepted as a valid model, and retry verifies the destination before reuse

### Requirement: Explicit pinned acquisition and conversion
The benchmark workflow SHALL acquire missing resources only during an explicit preparation phase using fixed revisions and verified manifests. Measurement SHALL perform no network acquisition or conversion. Conversion inputs, converter identity and output digests MUST be retained; archive metadata alone MUST NOT substitute for production integrity checks.

#### Scenario: Download missing model
- **WHEN** a selected model is absent and resource preparation is requested with acquisition enabled
- **THEN** the workflow downloads the pinned assets to persistent storage, verifies them and records receipts before declaring that model ready

#### Scenario: Converted file fails production manifest
- **WHEN** a local converted file differs from the production manifest or has insufficient provenance
- **THEN** the resource is marked conflicted or unverified, production measurement is blocked, and neither the manifest nor its checks are weakened to admit it

### Requirement: Preserve voice and cache boundaries
The workflow SHALL retain reference audio, exact transcripts, voice descriptions and digests with their model identities. Encoded references and caches MUST remain implementation/model/precision scoped and MUST NOT be reused across incompatible models solely because voice names match.

#### Scenario: Shared reference across model variants
- **WHEN** two variants use the same reference WAV for comparable listening samples
- **THEN** the WAV and transcript identity are recorded, while each model prepares its own compatible encoded reference and cache
