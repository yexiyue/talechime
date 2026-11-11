# talechime-loqui-g2p

TaleChime's local fork of `loqui-g2p` 0.4.0. It keeps the upstream English
lexicon, tokenization, tagging and morphology, removes the neural fallback and
adds a fallible caller-provided OOV callback. Kokoro uses that callback for a
separately distributed eSpeak NG program; this crate does not link it.

Upstream source, changes and licenses are recorded in [SOURCE.md](SOURCE.md).
