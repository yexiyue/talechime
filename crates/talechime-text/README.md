# talechime-text

Native byte-level FST execution and fixed WeText tagged-token serialization.
This crate supplies mechanics; each model frontend owns language selection,
normalization options, grouping, pronunciation controls and source mapping.
There is no global rewrite applied to all backends.

Grammars are provided and verified by resource-owning adapters. Missing or
changed grammar files fail explicitly. Path costs use strict F32 comparisons
to preserve small official grammar preferences. Requests can cancel path
search; negative cycles are rejected.

IndexTTS and CosyVoice3 oracle tests live with their model crates. Ordinary
tests do not fetch resources. See [SOURCE.md](SOURCE.md) for provenance.
