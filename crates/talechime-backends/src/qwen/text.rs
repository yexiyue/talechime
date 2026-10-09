use tts_core::text::{
    TextSegment, is_decoration_line, is_heading_line, is_non_spoken_line, sanitize_for_speech,
};

pub(super) fn normalize(source: &str) -> String {
    let source = sanitize_for_speech(source);
    let mut result = String::new();
    for line in source.lines() {
        let line = line.trim();
        if line.is_empty() || is_decoration_line(line) {
            continue;
        }
        if result
            .chars()
            .last()
            .is_some_and(|c| c.is_ascii_alphanumeric())
            && line
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphanumeric())
        {
            result.push(' ');
        }
        result.push_str(line);
    }
    result
}

/// Keep soft layout lines together; separate headings and true paragraphs.
pub(super) fn segments(source: &str) -> Vec<TextSegment> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut offset = 0;
    for line in source.split_inclusive('\n') {
        let hard = is_non_spoken_line(line) || is_heading_line(line);
        if hard {
            append(source, start, offset, &mut result);
            if is_heading_line(line) {
                append(source, offset, offset + line.len(), &mut result);
            }
            start = offset + line.len();
        }
        offset += line.len();
    }
    append(source, start, source.len(), &mut result);
    result
}
fn append(source: &str, mut start: usize, end: usize, result: &mut Vec<TextSegment>) {
    // 180 UTF-8 bytes allow up to 60 Chinese characters per synthesis context.
    while start < end {
        let mut limit = start;
        let mut sentence = None;
        let mut clause = None;
        let mut word = None;
        for (relative, c) in source[start..end].char_indices() {
            let at = start + relative;
            let next = at + c.len_utf8();
            if next - start > 180 {
                break;
            }
            limit = next;
            if "。！？.!?".contains(c) || "”’\"'）)]】》」』".contains(c) && sentence == Some(at)
            {
                sentence = Some(next);
            } else if "，,；;：:".contains(c) {
                clause = Some(next);
            } else if c.is_whitespace() {
                word = Some(next);
            }
        }
        let stop = if limit == end {
            end
        } else {
            sentence.or(clause).or(word).unwrap_or(limit)
        };
        let text = normalize(&source[start..stop]);
        if !text.is_empty() {
            result.push(TextSegment {
                text,
                start,
                end: stop,
            });
        }
        start = stop;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn noisy_layout_preserves_hard_boundaries_and_source_ranges() {
        let source = "前句。\n====\u{200B}====\n后句。\n\u{200B}\n\u{FEFF}第一章 开始\n正文。";
        let pieces = segments(source);
        assert_eq!(pieces.len(), 4);
        let spoken: Vec<_> = pieces.iter().map(|p| normalize(&p.text)).collect();
        let expected: Vec<_> = ["前句。", "后句。", "第一章 开始", "正文。"]
            .into_iter()
            .map(normalize)
            .collect();
        assert_eq!(spoken, expected);
        for piece in pieces {
            assert_eq!(
                normalize(&piece.text),
                normalize(&source[piece.start..piece.end])
            );
        }
        assert!(segments("\u{200B}\n\u{0}").is_empty());
    }
    #[test]
    fn inaudible_noise_never_reaches_synthesis() {
        assert_eq!(normalize("你\u{200B}好，世\u{0}界\u{FEFF}"), "你好，世界");
        assert_eq!(normalize("====\u{200B}===="), "");
    }
    #[test]
    fn preserves_source_ranges_and_hard_boundaries() {
        let source = "====\n第一章 开始\n你好，\n世界。\n\n第二段 a=b。";
        let parts = segments(source);
        assert_eq!(parts.len(), 3);
        assert!(parts[0].text.starts_with("第一章"));
        assert_eq!(parts[1].text, "你好，世界。");
        for part in parts {
            assert_eq!(part.text, normalize(&source[part.start..part.end]));
        }
        assert_eq!(normalize("Hello\nworld. a=b"), "Hello world. a=b");
    }
    #[test]
    fn long_unpunctuated_text_is_bounded_and_complete() {
        let source = "汉".repeat(301);
        let parts = segments(&source);
        assert_eq!(
            parts
                .iter()
                .map(|part| part.text.as_str())
                .collect::<String>(),
            source
        );
        assert!(parts.iter().all(|part| part.end - part.start <= 180));
        assert_eq!(parts.last().unwrap().end, source.len());
    }
    #[test]
    fn crlf_and_sentence_quotes_keep_utf8_mapping() {
        let source = format!("{}。\"\r\n{}。", "甲".repeat(30), "乙".repeat(40));
        let parts = segments(&source);
        assert!(parts[0].text.ends_with("。\""));
        for part in parts {
            assert_eq!(part.text, normalize(&source[part.start..part.end]));
        }
    }
}
