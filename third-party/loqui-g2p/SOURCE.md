# TaleChime loqui-g2p fork

Based on crates.io `loqui-g2p` 0.4.0 from Awakened Labs, MIT; original LICENSE,
NOTICE and bundled data licenses are retained. The package is renamed
`talechime-loqui-g2p` to make the path dependency explicit across Cargo
workspaces. `tokens_with_fallback` injects a caller-supplied fallible callback
at the existing unresolved-token branch, with error propagation. The upstream neural fallback and model weights are
removed; SpellOut remains available. No eSpeak code is linked here.
Unmodified upstream modules retain their original formatting via `rustfmt::skip`.
The original package records upstream commit `717691aadf67408c85d87dc33e5d1d77b2ea4b4d`.
Its published archive omitted `tests/data/num2words-en.tsv`; the test fixture
was restored from that commit, Git blob `b201f2e92c8b6235c4a3828a173f3c4168cd813c`.
