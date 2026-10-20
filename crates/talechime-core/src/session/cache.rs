//! Persistent, keyed reuse of whole-chapter spools for deterministic runs.
//!
//! Only `AfterChapterReady` executions that start from the beginning of the
//! source, pin their sampling seed and disable readback participate: those are
//! the runs whose audio is reproducible, so serving a cached spool is exactly
//! equivalent to regenerating it. Everything else keeps the disposable staging
//! path. Entries are flat spool files named by a SHA-256 of every
//! generation-relevant input, and the directory is pruned oldest-first to a
//! byte budget.
use super::staging::{self, Storage};
use super::*;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Default reuse budget; chapters are the largest rebuildable artefact.
pub const DEFAULT_MAX_BYTES: u64 = 8 * 1024 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct ChapterCacheOptions {
    /// Existing directory for published spools.
    pub directory: PathBuf,
    /// Total size bound for published spools.
    pub max_bytes: u64,
}

/// Whole-chapter spool cache. Keys fold in every input that can change the
/// generated audio; playback-side settings (volume, speed) are deliberately
/// excluded because replay applies them again.
pub(crate) struct ChapterCache {
    options: ChapterCacheOptions,
}

impl ChapterCache {
    pub(crate) fn new(options: ChapterCacheOptions) -> Self {
        Self { options }
    }

    /// A run may only reuse or populate the cache when its audio is
    /// deterministic and covers the entire source.
    pub(crate) fn eligible(
        &self,
        byte: usize,
        seed: &crate::params::SeedPolicy,
        verification: &crate::verification::VerificationOptions,
    ) -> Option<u64> {
        let crate::params::SeedPolicy::Pinned(seed) = *seed else {
            return None;
        };
        if byte != 0 || verification.policy != crate::verification::VerificationPolicy::Off {
            return None;
        }
        Some(seed)
    }

    pub(crate) fn key(
        identity: &str,
        backend: &str,
        model: Option<&str>,
        text_hash: &str,
        seed: u64,
        input: &PlanInput,
        spans: &[SpeechSpan],
    ) -> String {
        // Structured encoding prevents separator collisions in arbitrary
        // resource identities, parameter strings and model names.
        let canonical = serde_json::json!({
            "version": "chapter-cache-v2",
            "core": env!("CARGO_PKG_VERSION"),
            "identity": identity,
            "backend": backend,
            "model": model,
            "text_hash": text_hash,
            "seed": seed,
            "continuation": input.continuation,
            "boundary": input.continuation_boundary,
            "span": input.continuation_span.unwrap_or(crate::continuation::DEFAULT_CHAIN_SPAN_SECONDS),
            "params": input.params.iter().collect::<std::collections::BTreeMap<_, _>>(),
            "spans": spans.iter().map(|span| serde_json::json!({
                "range": [span.range().start, span.range().end],
                "voice": span.voice(),
                "style": span.style(),
            })).collect::<Vec<_>>(),
        }).to_string();
        format!("{:x}", Sha256::digest(canonical.as_bytes()))
    }

    pub(crate) fn publish_path(&self, key: &str) -> PathBuf {
        self.options.directory.join(format!("{key}.pcm"))
    }

    /// Open and fully validate a published spool. Corrupt or foreign entries
    /// are removed so the next run regenerates instead of failing repeatedly.
    pub(crate) fn lookup(&self, key: &str, text: &str) -> Option<Arc<Storage>> {
        let path = self.publish_path(key);
        if !std::fs::symlink_metadata(&path).is_ok_and(|meta| meta.file_type().is_file()) {
            return None;
        }
        let storage = match staging::open_persistent(&path).and_then(|storage| {
            staging::validate(&storage, text)?;
            Ok(storage)
        }) {
            Ok(storage) => storage,
            Err(error) => {
                let _ = std::fs::remove_file(&path);
                eprintln!("chapter cache discarded {key}: {error}");
                return None;
            }
        };
        touch(&path);
        Some(storage)
    }

    /// Drop oldest-first until the directory fits the budget again. Failures
    /// never fail playback; the next publish retries the sweep.
    pub(crate) fn prune(&self) {
        let _ = std::fs::create_dir_all(&self.options.directory);
        let Ok(entries) = std::fs::read_dir(&self.options.directory) else {
            return;
        };
        let mut published: Vec<(std::time::SystemTime, u64, PathBuf)> = entries
            .flatten()
            .filter(|entry| {
                if entry.file_type().is_ok_and(|kind| kind.is_file()) {
                    staging::cleanup_candidate(&entry.path());
                }
                entry.path().extension().is_some_and(|ext| ext == "pcm")
                    && entry.file_type().is_ok_and(|kind| kind.is_file())
            })
            .filter_map(|entry| {
                let meta = entry.metadata().ok()?;
                Some((meta.modified().ok()?, meta.len(), entry.path()))
            })
            .collect();
        let total: u64 = published.iter().map(|(_, size, _)| size).sum();
        if total <= self.options.max_bytes {
            return;
        }
        published.sort_by_key(|(modified, _, _)| *modified);
        let mut remaining = total;
        for (_, size, path) in published {
            if remaining <= self.options.max_bytes {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                remaining = remaining.saturating_sub(size);
            }
        }
    }
}

fn touch(path: &Path) {
    let now = std::time::SystemTime::now();
    let _ = std::fs::File::options()
        .append(true)
        .open(path)
        .and_then(|file| {
            file.set_modified(now)?;
            Ok(())
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(directory: &Path, max_bytes: u64) -> ChapterCache {
        ChapterCache::new(ChapterCacheOptions {
            directory: directory.to_path_buf(),
            max_bytes,
        })
    }
    fn verification() -> crate::verification::VerificationOptions {
        crate::verification::VerificationOptions::default()
    }

    #[test]
    fn eligibility_requires_pinned_seed_full_chapter_and_no_readback() {
        let directory = tempfile::tempdir().unwrap();
        let cache = cache(directory.path(), DEFAULT_MAX_BYTES);
        let seed = crate::params::SeedPolicy::Pinned(7);
        assert_eq!(cache.eligible(0, &seed, &verification()), Some(7));
        assert_eq!(cache.eligible(3, &seed, &verification()), None);
        assert_eq!(
            cache.eligible(0, &crate::params::SeedPolicy::Auto, &verification()),
            None
        );
        let mut gated = verification();
        gated.policy = crate::verification::VerificationPolicy::ReportOnly;
        assert_eq!(cache.eligible(0, &seed, &gated), None);
    }

    #[test]
    fn keys_separate_every_generation_input() {
        let caps: talechime_protocol::Capabilities = serde_json::from_value(serde_json::json!({
            "backend": "moss",
            "voices": ["A"],
            "default_voice": "A",
            "native_streaming": false,
            "style": true,
            "cloning": false,
            "pronunciation": false,
        }))
        .unwrap();
        let build = |style: Option<&str>| {
            let mut plan = SpeechPlan::new(
                SourceSnapshot::new(
                    talechime_protocol::SourceId {
                        namespace: "t".into(),
                        book: "b".into(),
                        chapter: "c".into(),
                    },
                    "正文",
                    &talechime_protocol::text_hash("正文"),
                )
                .unwrap(),
                VoiceSnapshot::new(
                    &caps.backend,
                    caps.model.as_deref(),
                    &caps,
                    caps.voices.clone(),
                )
                .unwrap(),
                PlaybackPolicy::AfterChapterReady,
            );
            plan.append(vec![SpeechSpan::new(
                TextRange { start: 0, end: 6 },
                "A",
                style.map(str::to_owned),
            )])
            .unwrap();
            let input = PlanInput::new(plan, 0);
            let spans: Vec<_> = input.plan.borrow().spans().to_vec();
            (input, spans)
        };
        let (input, spans) = build(None);
        let base = ChapterCache::key("fixture-v1", "moss", None, "h1", 7, &input, &spans);
        assert_eq!(
            ChapterCache::key("fixture-v1", "moss", None, "h1", 7, &input, &spans),
            base
        );
        assert_ne!(
            ChapterCache::key("fixture-v2", "moss", None, "h1", 7, &input, &spans),
            base
        );
        assert_ne!(
            ChapterCache::key("fixture-v1", "qwen", None, "h1", 7, &input, &spans),
            base
        );
        assert_ne!(
            ChapterCache::key("fixture-v1", "moss", Some("2b"), "h1", 7, &input, &spans),
            base
        );
        assert_ne!(
            ChapterCache::key("fixture-v1", "moss", None, "h2", 7, &input, &spans),
            base
        );
        assert_ne!(
            ChapterCache::key("fixture-v1", "moss", None, "h1", 8, &input, &spans),
            base
        );
        let (mut other, other_spans) = build(None);
        other.continuation = false;
        assert_ne!(
            ChapterCache::key("fixture-v1", "moss", None, "h1", 7, &other, &other_spans),
            base
        );
        let (styled, styled_spans) = build(Some("cheerful"));
        assert_ne!(
            ChapterCache::key("fixture-v1", "moss", None, "h1", 7, &styled, &styled_spans),
            base
        );
    }

    #[test]
    fn lookup_drops_foreign_or_corrupt_entries() {
        let directory = tempfile::tempdir().unwrap();
        let cache = cache(directory.path(), DEFAULT_MAX_BYTES);
        let path = cache.publish_path("k");
        std::fs::write(&path, b"not a spool").unwrap();
        assert!(cache.lookup("k", "正文").is_none());
        assert!(!path.exists());
    }

    #[test]
    fn prune_removes_oldest_entries_until_within_budget() {
        let directory = tempfile::tempdir().unwrap();
        let cache = cache(directory.path(), 10);
        let old = cache.publish_path("old");
        std::fs::write(&old, vec![0u8; 6]).unwrap();
        let past = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        std::fs::File::options()
            .append(true)
            .open(&old)
            .unwrap()
            .set_modified(past)
            .unwrap();
        std::fs::write(cache.publish_path("new"), vec![0u8; 6]).unwrap();
        let orphan = directory.path().join("talechime-candidate-orphan.staging");
        std::fs::write(&orphan, b"TCHSTG01").unwrap();
        cache.prune();
        assert!(!orphan.exists());
        assert!(!old.exists());
        assert!(cache.publish_path("new").exists());
    }
}
