//! A part-of-speech tagger distilled from spaCy's `en_core_web_sm`.
//!
//! misaki reads spaCy's Penn Treebank tags to pick heteronym readings
//! ("record" as NN or VB), to treat "a", "to", "in" and "the" by role, and
//! to decide what is punctuation or a currency sign. This is an averaged
//! perceptron (Honnibal's design, as in NLTK) trained on spaCy's own
//! predictions over public-domain text, so it learns spaCy's decisions
//! rather than a different treebank's. The weights are ours: trained by
//! `examples/train_tagger.rs` from the output of an MIT-licensed model.

use std::collections::HashMap;

use crate::Error;

static WEIGHTS: &[u8] = include_bytes!("../data/tagger-en.bin");

const MAGIC: &[u8; 8] = b"LQTAG001";
const START: [&str; 2] = ["-START-", "-START2-"];
const END: [&str; 2] = ["-END-", "-END2-"];

/// Whitespace tokens get spaCy's `_SP` tag without consulting the model.
pub const SPACE_TAG: &str = "_SP";

pub struct Tagger {
    classes: Vec<String>,
    /// Words the training data always tagged the same way.
    tagdict: HashMap<String, u8>,
    weights: HashMap<String, Vec<(u8, f32)>>,
}

impl std::fmt::Debug for Tagger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tagger").field("classes", &self.classes.len()).field("features", &self.weights.len()).finish()
    }
}

impl Tagger {
    /// Loads the embedded model.
    pub fn english() -> Result<Self, Error> {
        Self::from_bytes(WEIGHTS)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut r = Reader { bytes, at: 0 };
        if r.take(MAGIC.len())? != MAGIC {
            return Err(Error::Weights("tagger: bad magic".into()));
        }
        let classes = (0..r.u8()?).map(|_| r.str8()).collect::<Result<Vec<_>, _>>()?;
        let mut tagdict = HashMap::new();
        for _ in 0..r.u32()? {
            let word = r.str8()?;
            tagdict.insert(word, r.u8()?);
        }
        let mut weights = HashMap::new();
        for _ in 0..r.u32()? {
            let feature = r.str16()?;
            let n = r.u8()?;
            let row = (0..n).map(|_| Ok((r.u8()?, r.f32()?))).collect::<Result<Vec<_>, Error>>()?;
            weights.insert(feature, row);
        }
        if r.at != bytes.len() {
            return Err(Error::Weights("tagger: trailing bytes".into()));
        }
        Ok(Self { classes, tagdict, weights })
    }

    /// Tags a tokenized sentence, one tag per word.
    pub fn tag(&self, words: &[&str]) -> Vec<String> {
        let context = context(words);
        let mut prev = START[0].to_owned();
        let mut prev2 = START[1].to_owned();
        let mut tags = Vec::with_capacity(words.len());
        for (i, word) in words.iter().enumerate() {
            let tag = if word.chars().all(char::is_whitespace) && !word.is_empty() {
                SPACE_TAG.to_owned()
            } else if let Some(&class) = self.tagdict.get(*word) {
                self.classes[class as usize].clone()
            } else {
                let mut scores = vec![0f32; self.classes.len()];
                for feature in features(i, word, &context, &prev, &prev2) {
                    if let Some(row) = self.weights.get(&feature) {
                        for &(class, weight) in row {
                            scores[class as usize] += weight;
                        }
                    }
                }
                // Ties go to the later class, as Python's max over a
                // (score, class) pair would; any fixed rule is fine as
                // long as training used the same one.
                let best = (0..scores.len()).max_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(a.cmp(&b)));
                self.classes[best.unwrap_or(0)].clone()
            };
            prev2 = std::mem::replace(&mut prev, tag.clone());
            tags.push(tag);
        }
        tags
    }
}

/// The normalised sentence, padded with start and end markers, that the
/// features look into.
pub fn context(words: &[&str]) -> Vec<String> {
    START.iter().map(|s| (*s).to_owned()).chain(words.iter().map(|w| normalise(w))).chain(END.iter().map(|s| (*s).to_owned())).collect()
}

/// Folds words that behave alike onto one feature value.
pub fn normalise(word: &str) -> String {
    let first = word.chars().next();
    if word.contains('-') && first != Some('-') {
        "!HYPHEN".to_owned()
    } else if word.len() == 4 && word.bytes().all(|b| b.is_ascii_digit()) {
        "!YEAR".to_owned()
    } else if first.is_some_and(|c| c.is_ascii_digit()) {
        "!DIGITS".to_owned()
    } else {
        word.to_lowercase()
    }
}

/// The features for word `i`. Honnibal's set, plus a word shape and short
/// affixes: spaCy's own tagger leans on shape ("Xxxx", "dd:dd") and case,
/// and the lexicon keys on case for proper nouns.
pub fn features(i: usize, word: &str, context: &[String], prev: &str, prev2: &str) -> Vec<String> {
    let c = i + START.len();
    vec![
        "bias".to_owned(),
        format!("i suffix {}", suffix(word, 3)),
        format!("i suffix2 {}", suffix(word, 2)),
        format!("i suffix1 {}", suffix(word, 1)),
        format!("i pref1 {}", word.chars().next().map(String::from).unwrap_or_default()),
        format!("i pref3 {}", word.chars().take(3).collect::<String>().to_lowercase()),
        format!("i shape {}", shape(word)),
        format!("i-1 tag {prev}"),
        format!("i-2 tag {prev2}"),
        format!("i tag+i-2 tag {prev} {prev2}"),
        format!("i word {}", context[c]),
        format!("i-1 tag+i word {prev} {}", context[c]),
        format!("i-1 word {}", context[c - 1]),
        format!("i-1 suffix {}", suffix(&context[c - 1], 3)),
        format!("i-2 word {}", context[c - 2]),
        format!("i+1 word {}", context[c + 1]),
        format!("i+1 suffix {}", suffix(&context[c + 1], 3)),
        format!("i+2 word {}", context[c + 2]),
    ]
}

fn suffix(word: &str, n: usize) -> String {
    let chars: Vec<char> = word.chars().collect();
    chars[chars.len().saturating_sub(n)..].iter().collect()
}

/// spaCy's word shape: letters become X/x, digits d, runs longer than four
/// are cut to four ("Apple" is "Xxxxx", "1999" is "dddd", "9:30" is "d:dd").
fn shape(word: &str) -> String {
    let mut out = String::new();
    let mut last = None;
    let mut run = 0;
    for c in word.chars() {
        let s = if c.is_alphabetic() {
            if c.is_uppercase() { 'X' } else { 'x' }
        } else if c.is_ascii_digit() {
            'd'
        } else {
            c
        };
        run = if Some(s) == last { run + 1 } else { 1 };
        last = Some(s);
        if run <= 4 {
            out.push(s);
        }
    }
    out
}

/// Writes a model in the format [`Tagger::from_bytes`] reads. Used by the
/// training example.
pub fn encode(classes: &[String], tagdict: &[(String, u8)], weights: &[(String, Vec<(u8, f32)>)]) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    out.push(u8::try_from(classes.len()).expect("fewer than 256 tags"));
    for class in classes {
        push_str8(&mut out, class);
    }
    out.extend(u32::try_from(tagdict.len()).expect("tagdict fits").to_le_bytes());
    for (word, class) in tagdict {
        push_str8(&mut out, word);
        out.push(*class);
    }
    out.extend(u32::try_from(weights.len()).expect("features fit").to_le_bytes());
    for (feature, row) in weights {
        let bytes = feature.as_bytes();
        out.extend(u16::try_from(bytes.len()).expect("feature under 64 KiB").to_le_bytes());
        out.extend(bytes);
        out.push(u8::try_from(row.len()).expect("fewer than 256 tags"));
        for (class, weight) in row {
            out.push(*class);
            out.extend(weight.to_le_bytes());
        }
    }
    out
}

fn push_str8(out: &mut Vec<u8>, s: &str) {
    out.push(u8::try_from(s.len()).expect("short string"));
    out.extend(s.as_bytes());
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], Error> {
        let slice = self.bytes.get(self.at..self.at + n).ok_or_else(|| Error::Weights("tagger: truncated".into()))?;
        self.at += n;
        Ok(slice)
    }
    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, Error> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn f32(&mut self) -> Result<f32, Error> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn str8(&mut self) -> Result<String, Error> {
        let n = self.u8()? as usize;
        self.string(n)
    }
    fn str16(&mut self) -> Result<String, Error> {
        let b = self.take(2)?;
        let n = u16::from_le_bytes([b[0], b[1]]) as usize;
        self.string(n)
    }
    fn string(&mut self, n: usize) -> Result<String, Error> {
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| Error::Weights("tagger: bad utf-8".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_matches_spacy() {
        assert_eq!(shape("Apple"), "Xxxxx");
        assert_eq!(shape("1999"), "dddd");
        assert_eq!(shape("9:30"), "d:dd");
        assert_eq!(shape("NASA"), "XXXX");
        assert_eq!(shape("internationalization"), "xxxx");
    }

    #[test]
    fn round_trips_through_the_binary_format() {
        let classes = vec!["NN".to_owned(), "VB".to_owned()];
        let tagdict = vec![("the".to_owned(), 0u8)];
        let weights = vec![("bias".to_owned(), vec![(1u8, 0.5f32)])];
        let tagger = Tagger::from_bytes(&encode(&classes, &tagdict, &weights)).unwrap();
        assert_eq!(tagger.tag(&["the", "run"]), ["NN", "VB"]);
    }
}
