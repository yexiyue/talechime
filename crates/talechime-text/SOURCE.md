# Text execution sources

Original Rust mechanics are Apache-2.0, using rustfst 1.3.1 (MIT OR
Apache-2.0). No neural weights or FST grammar binaries are bundled here.

WeText token ordering follows wetext 0.1.0 / WeTextProcessing 1.3.0
(Apache-2.0). Model-owned resource manifests fix the grammar source,
revision, license, size and SHA-256; IndexTTS additionally owns the NeMo
1.2.0 Spanish parser and its independent fixtures. See
`../indextts/SOURCE.md` and `../cosyvoice3/SOURCE.md` for exact upstream
identities and oracle boundaries. These model policies do not belong in
this shared crate.
