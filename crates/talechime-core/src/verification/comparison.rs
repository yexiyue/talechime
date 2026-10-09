use super::*;
use unicode_normalization::UnicodeNormalization;

fn numeric(text: &str) -> bool {
    !text.is_empty()
        && text.chars().all(|c| {
            c.is_numeric()
                || "零〇一二三四五六七八九十百千万亿两点元角分块年月日时秒百分之.%−-".contains(c)
        })
}
pub(super) fn confirmed(
    normalization: &normalization::Normalizer,
    a: &[ReadbackDifference],
    b: &[ReadbackDifference],
    spoken: &str,
) -> bool {
    a.iter().any(|x| {
        b.iter().any(|y| match x.kind {
            // Substitutions include homophones and ASR ambiguity, even with agreement.
            DifferenceKind::Substitution => false,
            DifferenceKind::Missing => x == y && !numeric(&x.expected),
            // A repeated source phrase may be aligned at different insertion boundaries.
            DifferenceKind::Extra => {
                y.kind == x.kind
                    && x.observed == y.observed
                    && x.observed.chars().count() >= 2
                    && !numeric(&x.observed)
                    // Pronoun spelling cannot mask a repeated phrase; it still remains a substitution in evidence.
                    && normalization.normalize(spoken).replace(['她', '它'], "他")
                        .contains(&x.observed.replace(['她', '它'], "他"))
            }
        })
    })
}

pub(super) fn compare(
    normalization: &normalization::Normalizer,
    request: &ReadbackRequest<'_>,
    observed: &str,
) -> Vec<ReadbackDifference> {
    let expected = normalization.normalize(request.spoken_text);
    let observed = normalization.normalize(observed);
    let a: Vec<char> = expected.chars().collect();
    let mut normalized_offsets: Vec<usize> = expected.char_indices().map(|(at, _)| at).collect();
    normalized_offsets.push(expected.len());
    let b: Vec<char> = observed.chars().collect();
    let cols = b.len() + 1;
    let mut costs = vec![0u16; (a.len() + 1) * cols];
    for i in 0..=a.len() {
        costs[i * cols] = i as u16;
    }
    for (j, cost) in costs.iter_mut().take(cols).enumerate() {
        *cost = j as u16;
    }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            costs[i * cols + j] = (costs[(i - 1) * cols + j - 1] + u16::from(a[i - 1] != b[j - 1]))
                .min(costs[(i - 1) * cols + j] + 1)
                .min(costs[i * cols + j - 1] + 1);
        }
    }
    // Edit operations carry normalized character positions, never advertised as byte ranges.
    let (mut i, mut j) = (a.len(), b.len());
    let mut edits = Vec::new();
    while i > 0 || j > 0 {
        if i > 0
            && j > 0
            && a[i - 1] == b[j - 1]
            && costs[i * cols + j] == costs[(i - 1) * cols + j - 1]
        {
            i -= 1;
            j -= 1;
            edits.push((None, i, j));
        } else if i > 0 && j > 0 && costs[i * cols + j] == costs[(i - 1) * cols + j - 1] + 1 {
            i -= 1;
            j -= 1;
            edits.push((Some(DifferenceKind::Substitution), i, j));
        } else if i > 0 && costs[i * cols + j] == costs[(i - 1) * cols + j] + 1 {
            i -= 1;
            edits.push((Some(DifferenceKind::Missing), i, j));
        } else {
            j -= 1;
            edits.push((Some(DifferenceKind::Extra), i, j));
        }
    }
    edits.reverse();
    let original = &request.source.text()[request.range.start..request.range.end];
    let mut source_map = Vec::new();
    let mut mapped = String::new();
    for (at, c) in original.char_indices() {
        for normalized in c
            .to_string()
            .nfkc()
            .flat_map(char::to_lowercase)
            .filter(|c| c.is_alphanumeric())
        {
            mapped.push(normalized);
            source_map.push(TextRange {
                start: request.range.start + at,
                end: request.range.start + at + c.len_utf8(),
            });
        }
    }
    let exact_mapping = mapped == expected;
    let mut result = Vec::new();
    let mut at = 0;
    while at < edits.len() {
        let (Some(kind), start, _) = edits[at] else {
            at += 1;
            continue;
        };
        let mut end = start;
        let mut missing = String::new();
        let mut extra = String::new();
        while at < edits.len() && edits[at].0 == Some(kind) {
            let (_, i, j) = edits[at];
            if kind != DifferenceKind::Extra {
                missing.push(a[i]);
                end = i + 1;
            }
            if kind != DifferenceKind::Missing {
                extra.push(b[j]);
            }
            at += 1;
        }
        let range = if !exact_mapping {
            request.range
        } else if start == end {
            let byte = source_map.get(start).map_or(request.range.end, |r| r.start);
            TextRange {
                start: byte,
                end: byte,
            }
        } else {
            TextRange {
                start: source_map[start].start,
                end: source_map[end - 1].end,
            }
        };
        result.push(ReadbackDifference {
            kind,
            range,
            normalized_range: TextRange {
                start: normalized_offsets[start],
                end: normalized_offsets[end],
            },
            expected: missing,
            observed: extra,
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    fn differences(source: &str, spoken: &str, observed: &str) -> Vec<ReadbackDifference> {
        let snapshot = SourceSnapshot::new(
            tts_protocol::SourceId {
                namespace: "test".into(),
                book: "".into(),
                chapter: "".into(),
            },
            source,
            &tts_protocol::text_hash(source),
        )
        .unwrap();
        compare(
            &normalization::Normalizer::new().unwrap(),
            &ReadbackRequest {
                source: &snapshot,
                range: TextRange {
                    start: 0,
                    end: source.len(),
                },
                spoken_text: spoken,
                backend: "fake",
                model: None,
                voice: "v",
                style: None,
                attempt: 0,
            },
            observed,
        )
    }
    #[test]
    fn maps_multibyte_source_after_normalizing_punctuation_and_width() {
        assert!(differences("你好，Ａ！", "你好，Ａ！", "你好 a").is_empty());
        let d = differences("你好，世界！", "你好，世界！", "你好，界！");
        assert_eq!(d[0].range, TextRange { start: 9, end: 12 });
        assert_eq!(d[0].kind, DifferenceKind::Missing);
        assert!(confirmed(
            &normalization::Normalizer::new().unwrap(),
            &d,
            &d,
            "你好，世界！"
        ));
    }
    #[test]
    fn agreement_on_substitutions_or_numerals_is_still_suspect() {
        for (a, b) in [("她来了", "他来了"), ("一百元", "百元"), ("123元", "12元")] {
            let d = differences(a, a, b);
            assert!(!confirmed(
                &normalization::Normalizer::new().unwrap(),
                &d,
                &d,
                a
            ));
        }
    }
    #[test]
    fn numbers_do_not_mask_missing_non_numeric_words_or_repeated_pronouns() {
        let d = differences("她点头说会的", "她点头说会的", "");
        assert!(confirmed(
            &normalization::Normalizer::new().unwrap(),
            &d,
            &d,
            "她点头说会的"
        ));
        let d = differences("她点头说会的", "她点头说会的", "他点头说会的他点头说会的");
        assert!(confirmed(
            &normalization::Normalizer::new().unwrap(),
            &d,
            &d,
            "她点头说会的"
        ));
    }
    #[test]
    fn preprocessing_falls_back_to_segment_range_and_repeat_is_detected() {
        let d = differences("待处理原文", "你好世界", "你好");
        assert_eq!(d[0].range, TextRange { start: 0, end: 15 });
        let d = differences("你好世界", "你好世界", "你好世界你好世界");
        assert!(confirmed(
            &normalization::Normalizer::new().unwrap(),
            &d,
            &d,
            "你好世界"
        ));
    }
    #[test]
    fn coarse_source_mapping_does_not_confirm_different_missing_occurrences() {
        let a = differences("待处理原文", "你好世界你好", "世界你好");
        let b = differences("待处理原文", "你好世界你好", "你好世界");
        assert_eq!(a[0].range, b[0].range);
        assert_eq!(a[0].expected, b[0].expected);
        assert_ne!(a[0].normalized_range, b[0].normalized_range);
        assert!(!confirmed(
            &normalization::Normalizer::new().unwrap(),
            &a,
            &b,
            "你好世界你好"
        ));
    }
}
