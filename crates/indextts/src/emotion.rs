//! Fixed upstream QwenEmotion classification and ordered output conversion.
//! Inference semantics: IndexTTS d9e41aac, Qwen3 0.6B emotion checkpoint.
mod resources;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{Config, ModelForCausalLM};
use serde_json::Value;
use std::{path::Path, sync::OnceLock};

const LABELS: [(&str, &str); 8] = [
    ("高兴", "happy"),
    ("愤怒", "angry"),
    ("悲伤", "sad"),
    ("恐惧", "afraid"),
    ("反感", "disgusted"),
    ("低落", "melancholic"),
    ("惊讶", "surprised"),
    ("自然", "calm"),
];

/// The checkpoint's actual `chat_template.jinja`, with the fixed system prompt.
pub fn prompt(text: &str) -> String {
    format!("System: 文本情感分类<|endoftext|>\nHuman: {text}<|endoftext|>\nAssistant:")
}

fn label(value: &Value) -> Option<usize> {
    let text = value.as_str()?.trim();
    LABELS
        .iter()
        .position(|(cn, en)| text == *cn || text.eq_ignore_ascii_case(en))
}

/// Parse JSON or the upstream numeric fallback, preserve eight-dimension order,
/// clamp to 0..=1.2, and apply the official sad/melancholic workaround.
pub fn parse(content: &str, text: &str) -> Result<[f32; 8]> {
    let parsed: Value = serde_json::from_str(content).unwrap_or_else(|_| {
        static SCORES: OnceLock<regex::Regex> = OnceLock::new();
        let re = SCORES.get_or_init(|| {
            regex::Regex::new(r#"([^\s\":.,]+?)\"?\s*:\s*([\d.]+)"#).expect("static regex")
        });
        let mut values = serde_json::Map::new();
        for capture in re.captures_iter(content) {
            if let Ok(value) = capture[2].parse::<f64>() {
                values.insert(capture[1].to_owned(), Value::from(value));
            }
        }
        Value::Object(values)
    });
    let mut values = parsed.as_object().cloned().unwrap_or_default();
    let detected = label(&parsed).or_else(|| {
        ["emotion", "emotion_label", "label", "情感", "情绪"]
            .iter()
            .find_map(|key| values.get(*key).and_then(label))
    });
    if !LABELS.iter().any(|(cn, _)| values.contains_key(*cn))
        && let Some(index) = detected
    {
        values.insert(LABELS[index].0.into(), Value::from(1.0));
    }
    for (index, (cn, _)) in LABELS.iter().enumerate() {
        if let Some(detected) = values.get(*cn).and_then(label) {
            values.insert(
                (*cn).into(),
                Value::from(if detected == index { 1.0 } else { 0.0 }),
            );
            if detected != index {
                values.insert(LABELS[detected].0.into(), Value::from(1.0));
            }
        }
    }
    let lower = text.to_lowercase();
    if [
        "低落",
        "melancholy",
        "melancholic",
        "depression",
        "depressed",
        "gloomy",
    ]
    .iter()
    .any(|word| lower.contains(word))
    {
        let sad = values.remove("悲伤").unwrap_or(Value::from(0.0));
        let melancholic = values.remove("低落").unwrap_or(Value::from(0.0));
        values.insert("悲伤".into(), melancholic);
        values.insert("低落".into(), sad);
    }
    let mut vector = [0.0; 8];
    for (index, (cn, _)) in LABELS.iter().enumerate() {
        if let Some(value) = values.get(*cn) {
            let number = value
                .as_f64()
                .or_else(|| value.as_bool().map(|value| if value { 1.0 } else { 0.0 }))
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!("QwenEmotion invalid score for {cn}"))
                })?;
            if !number.is_finite() {
                candle_core::bail!("QwenEmotion non-finite score");
            }
            vector[index] = number.clamp(0.0, 1.2) as f32;
        }
    }
    if vector.iter().all(|value| *value <= 0.0) {
        vector[7] = 1.0;
    }
    Ok(vector)
}

/// Lazily owned by the IndexTTS inference thread; never starts Python.
pub struct QwenEmotion {
    model: ModelForCausalLM,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
}

impl QwenEmotion {
    /// Load and verify the complete fixed classifier under an IndexTTS resource root.
    pub fn load(root: &Path, device: &Device) -> Result<Self> {
        resources::verify(root)?;
        let directory = root.join("qwen0.6bemo4-merge");
        let config: Config = serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        let tokenizer = tokenizers::Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|error| candle_core::Error::Msg(format!("QwenEmotion tokenizer: {error}")))?;
        // CPU uses F32 to avoid unsupported half-precision host kernels. Numeric
        // and output parity must be measured separately from the official FP16.
        let dtype = if device.is_cpu() {
            DType::F32
        } else {
            DType::F16
        };
        let weights = candle_core::safetensors::load(directory.join("model.safetensors"), device)?;
        let vb = VarBuilder::from_tensors(weights, dtype, device);
        Ok(Self {
            model: ModelForCausalLM::new(&config, vb)?,
            tokenizer,
            device: device.clone(),
        })
    }

    /// Greedy decoding matches the fixed generation config (`do_sample=false`).
    /// Truncation and cancellation are errors, never default calm acceptance.
    pub fn infer(&mut self, text: &str, cancelled: &dyn Fn() -> bool) -> Result<[f32; 8]> {
        self.model.clear_kv_cache();
        let encoded = self
            .tokenizer
            .encode(prompt(text), false)
            .map_err(|error| candle_core::Error::Msg(format!("QwenEmotion encode: {error}")))?;
        let mut input = encoded.get_ids().to_vec();
        let mut offset = 0;
        let mut output = Vec::new();
        for _ in 0..32768 {
            if cancelled() {
                candle_core::bail!("QwenEmotion cancelled");
            }
            let tensor = Tensor::new(input.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&tensor, offset)?;
            let next = logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()?;
            offset += input.len();
            if next == 151643 {
                let start = output
                    .iter()
                    .rposition(|id| *id == 151668)
                    .map_or(0, |index| index + 1);
                let content = self
                    .tokenizer
                    .decode(&output[start..], true)
                    .map_err(|error| {
                        candle_core::Error::Msg(format!("QwenEmotion decode: {error}"))
                    })?;
                return parse(&content, text);
            }
            output.push(next);
            input = vec![next];
        }
        candle_core::bail!("QwenEmotion exhausted its token budget without EOS")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires explicitly prepared fixed QwenEmotion weights and independent model oracle"]
    fn fixed_classifier_matches_official_model_outputs() {
        let root =
            std::env::var_os("TALECHIME_INDEXTTS_EMOTION_ROOT").expect("IndexTTS resource root");
        let oracle = std::env::var_os("TALECHIME_INDEXTTS_EMOTION_ORACLE")
            .expect("independent official output JSON");
        let fixture: Value = serde_json::from_slice(&std::fs::read(oracle).unwrap()).unwrap();
        let mut classifier = QwenEmotion::load(Path::new(&root), &Device::Cpu).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let text = case["text"].as_str().unwrap();
            let tokens = classifier.tokenizer.encode(prompt(text), false).unwrap();
            assert_eq!(
                serde_json::to_value(tokens.get_ids()).unwrap(),
                case["input_ids"],
                "prompt: {text}"
            );
            let actual = classifier.infer(text, &|| false).unwrap();
            let expected: [f32; 8] = serde_json::from_value(case["vector"].clone()).unwrap();
            assert_eq!(actual, expected, "classification: {text}");
        }
    }
    #[test]
    fn direct_classifier_loading_rejects_changed_fixed_resources() {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("qwen0.6bemo4-merge");
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(directory.join("chat_template.jinja"), vec![b'x'; 550]).unwrap();
        let error = QwenEmotion::load(root.path(), &Device::Cpu)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("resource identity mismatch"));
    }
    #[test]
    fn conversion_matches_independent_official_fixtures() {
        let fixture: Value =
            serde_json::from_str(include_str!("../tests/fixtures/emotion-oracle.json")).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let actual = parse(
                case["content"].as_str().unwrap(),
                case["text"].as_str().unwrap(),
            )
            .unwrap();
            let expected: [f32; 8] = serde_json::from_value(case["vector"].clone()).unwrap();
            assert_eq!(actual, expected, "{case}");
        }
    }
    #[test]
    fn fixed_template_is_not_the_generic_qwen_chat_template() {
        assert_eq!(
            prompt("你好"),
            "System: 文本情感分类<|endoftext|>\nHuman: 你好<|endoftext|>\nAssistant:"
        );
    }
    #[test]
    fn conversion_handles_aliases_clamps_and_empty_content() {
        assert_eq!(
            parse(r#"{"emotion":"angry"}"#, "text").unwrap(),
            [0., 1., 0., 0., 0., 0., 0., 0.]
        );
        assert_eq!(
            parse(r#"{"高兴":2,"愤怒":-1}"#, "text").unwrap(),
            [1.2, 0., 0., 0., 0., 0., 0., 0.]
        );
        assert_eq!(parse("{}", "text").unwrap()[7], 1.0);
        assert_eq!(parse("高兴: 0.3", "text").unwrap()[0], 0.3);
    }
    #[test]
    fn melancholic_workaround_swaps_before_conversion() {
        assert_eq!(
            parse(r#"{"悲伤":0.8,"低落":0.2}"#, "gloomy").unwrap(),
            [0., 0., 0.2, 0., 0., 0.8, 0., 0.]
        );
        assert!(parse(r#"{"高兴":{}}"#, "text").is_err());
    }
}
