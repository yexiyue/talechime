# Spec Delta

## Purpose

Provide evidence-based regression decisions for repeatable backend performance work. Model-free checks and explicit real-model runs use separate environments and budgets, while compatibility and sample-coverage gates prevent cross-platform differences or incomplete runs from becoming unsupported optimization claims.

## ADDED Requirements

### Requirement: Compatibility before comparison
Comparison SHALL check schema, hardware, platform, provider, effective math implementation, precision, features, resource and workload identities, effective parameters, threads, timing boundaries and observation mode. Intended code changes MUST remain attributable. Incompatible pairs MUST list reasons and cannot produce a regression verdict; exploratory cross-device views MUST be labelled descriptive.

#### Scenario: Compare Windows CUDA with macOS Metal
- **WHEN** runs use different platforms and execution providers
- **THEN** a descriptive report can show both, but the comparator refuses to classify their difference as a code regression

#### Scenario: Compare an optimization revision
- **WHEN** revisions differ while required measurement conditions match
- **THEN** the comparator records the intended source and binary change instead of requiring identical executable hashes

### Requirement: Preserve sample hierarchy and failures
Results SHALL retain warmup exclusions, raw measured rounds, independent process summaries and successful versus expected coverage. Request RTF MUST use total synthesis time divided by actual generated audio duration within the request. Incomplete cases and small-sample percentiles MUST be labelled honestly and MUST NOT yield an unconditional pass.

#### Scenario: A measured round fails
- **WHEN** one required hot round terminates unsuccessfully
- **THEN** the report retains failure evidence, labels reduced coverage and does not silently evaluate only successful rounds as a complete baseline

### Requirement: Explicit regression policy
Each comparison SHALL record its metric direction, absolute and relative practical thresholds, sample policy and evidence rule. Outcomes MUST distinguish pass, regression, inconclusive, incomplete and incomparable. Output-preserving speed claims MUST include the declared output-equivalence evidence; timing alone MUST NOT establish quality equivalence.

#### Scenario: Observed change is smaller than measurement variation
- **WHEN** the evidence does not satisfy the configured decision rule
- **THEN** the result is inconclusive rather than a definitive speedup or regression

### Requirement: Layered verification without model downloads
Ordinary CI SHALL verify contracts and model-free benchmark execution without acquiring models. Real-model regression runs MUST be explicit, serial and tied to named hardware and prepared resources. Native compilation checks MUST remain separate from real-device performance evidence.

#### Scenario: A hosted macOS runner compiles Metal code
- **WHEN** native compilation succeeds without running a prepared model on a verified device
- **THEN** the result is recorded as compilation evidence and does not activate a Mac performance baseline

### Requirement: Portable future baseline onboarding
The workflow SHALL allow a later macOS machine to add independent CPU and Metal baselines using the same evidence schema and comparison protocol. Platform-specific collectors and validation status MUST remain explicit, and absent Mac evidence MUST NOT prevent delivery of verified Windows framework capabilities.

#### Scenario: Developer later onboards a Mac
- **WHEN** native collectors and prepared-resource model runs are verified on that machine
- **THEN** it can produce a separate Mac baseline without replacing Windows history or reinterpreting Windows memory measurements
