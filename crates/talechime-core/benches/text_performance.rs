//! Stable synthetic corpus: no models, user novels or network.
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn text_benches(c: &mut Criterion) {
    let corpus = "第一章\n\n你好，世界。这里是一段用于性能测试的固定文本。\n".repeat(128);
    c.bench_function("preprocess_fixed_corpus", |b| {
        b.iter(|| talechime_core::text::preprocess_text(black_box(&corpus), black_box(200)))
    });
    c.bench_function("sanitize_fixed_corpus", |b| {
        b.iter(|| talechime_core::text::sanitize_for_speech(black_box(&corpus)))
    });
}

criterion_group!(benches, text_benches);
criterion_main!(benches);
