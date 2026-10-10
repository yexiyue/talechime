//! Previous text/audio turn in the pinned upstream make_user_prompt format.
pub(super) fn previous_turn(
    prefix: &[u32],
    text: &[u32],
    response: &[u32],
    codes: &[Vec<u32>],
    text_pad: u32,
    audio_pad: u32,
) -> anyhow::Result<Vec<Vec<u32>>> {
    anyhow::ensure!(
        !text.is_empty() && !codes.is_empty(),
        "empty Realtime history"
    );
    anyhow::ensure!(
        codes
            .iter()
            .all(|f| f.len() == 16 && f.iter().all(|id| *id < 1024)),
        "invalid Realtime history codes"
    );
    let delay = text.len().min(12);
    let end = delay + codes.len();
    anyhow::ensure!(
        text.len() <= end + 1,
        "Realtime transcript exceeds audio turn"
    );
    let row = |id| {
        let mut row = vec![audio_pad; 17];
        row[0] = id;
        row
    };
    let mut rows: Vec<_> = prefix.iter().copied().map(row).collect();
    let start = rows.len();
    rows.extend(text.iter().copied().map(row));
    rows.resize(start + end + 1, row(text_pad));
    rows[start + delay - 1][1] = 1025;
    for (index, frame) in codes.iter().enumerate() {
        rows[start + delay + index][1..].copy_from_slice(frame);
    }
    rows[start + end][1] = 1026;
    rows.extend(response.iter().copied().map(row));
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_short_and_delayed_turns_keep_audio_and_transcript_paired() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/realtime-previous-turn.json"
        ))
        .unwrap();
        for (case, length) in fixture.as_array().unwrap().iter().zip([3, 12, 16]) {
            let text: Vec<u32> = (100..100 + length).collect();
            let codes = vec![vec![42; 16]; 20];
            let rows = previous_turn(&[1, 2], &text, &[3, 4], &codes, 9, 1024).unwrap();
            let expected: Vec<Vec<u32>> = serde_json::from_value(case["rows"].clone()).unwrap();
            assert_eq!(rows, expected, "pinned official make_user_prompt");
            let start = 2 + (length as usize).min(12);
            assert_eq!(
                rows[2..2 + text.len()]
                    .iter()
                    .map(|f| f[0])
                    .collect::<Vec<_>>(),
                text
            );
            assert_eq!(rows[start - 1][1], 1025);
            assert_eq!(
                rows[start..start + 20]
                    .iter()
                    .map(|f| f[1..].to_vec())
                    .collect::<Vec<_>>(),
                codes
            );
            assert_eq!(rows[start + 20][1], 1026);
            assert_eq!(rows.last().unwrap()[0], 4);
        }
        assert!(previous_turn(&[], &[1; 20], &[], &[vec![1; 16]], 9, 1024).is_err());
        assert!(previous_turn(&[], &[1], &[], &[vec![1024; 16]], 9, 1024).is_err());
    }
}
