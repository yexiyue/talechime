use super::{
    plan::{Input, Output, build, verified},
    *,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
const HASH: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

#[cfg(unix)]
#[test]
fn downloaded_cross_volume_symlink_becomes_a_regular_verified_input() {
    let directory = tempfile::tempdir().unwrap();
    let cached = directory.path().join("cache");
    let input = directory.path().join("input");
    std::fs::write(&cached, b"abc").unwrap();
    std::os::unix::fs::symlink(&cached, &input).unwrap();
    let control = PreparationControl::default();
    materialize_input(&input, &control).unwrap();
    assert!(std::fs::symlink_metadata(&input).unwrap().is_file());
    assert!(verified(&input, 3, HASH, &control).unwrap());
    assert_eq!(std::fs::read(cached).unwrap(), b"abc");
}
static INPUTS: &[Input] = &[Input {
    url: None,
    path: "source.bin",
    repo: "fixture/model",
    revision: "1111111111111111111111111111111111111111",
    filename: "source.bin",
    size: 3,
    sha256: HASH,
}];
static OUTPUTS: &[Output] = &[Output {
    path: "model.bin",
    size: 3,
    sha256: HASH,
}];
fn recipe() -> Recipe {
    Recipe {
        id: "fixture-v1",
        target: "fixture/models/one/v1".into(),
        inputs: INPUTS,
        outputs: OUTPUTS,
        transform_outputs: &["model.bin"],
        workspace_bytes: 8,
        transform: |work| {
            work.component("model.bin", |input, output, _| {
                std::fs::copy(input.join("source.bin"), output)?;
                Ok(())
            })
        },
    }
}
#[test]
fn plan_is_read_only_and_accounts_for_shared_cache_and_workspace() {
    let root = tempfile::tempdir().unwrap();
    let downloads = crate::downloads::Downloads::test(root.path());
    let before = build(
        &recipe(),
        root.path(),
        &downloads,
        &PreparationControl::default(),
    )
    .unwrap();
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    assert_eq!(
        (before.download_bytes, before.peak_additional_bytes),
        (3, 14)
    );
    let cached = downloads
        .cache_dir()
        .join("models--fixture--model/snapshots")
        .join(INPUTS[0].revision)
        .join("source.bin");
    std::fs::create_dir_all(cached.parent().unwrap()).unwrap();
    std::fs::write(&cached, b"abc").unwrap();
    let after = build(
        &recipe(),
        root.path(),
        &downloads,
        &PreparationControl::default(),
    )
    .unwrap();
    assert_eq!((after.download_bytes, after.peak_additional_bytes), (0, 11));
    assert_ne!(before.fingerprint, after.fingerprint);
    std::fs::write(cached, b"bad").unwrap();
    assert_eq!(
        build(
            &recipe(),
            root.path(),
            &downloads,
            &PreparationControl::default()
        )
        .unwrap()
        .download_bytes,
        3
    );
}
#[test]
fn plans_reject_unsafe_outputs_and_overflow() {
    let root = tempfile::tempdir().unwrap();
    let downloads = crate::downloads::Downloads::test(root.path());
    let mut invalid = recipe();
    invalid.target = "../outside".into();
    assert!(
        build(
            &invalid,
            root.path(),
            &downloads,
            &PreparationControl::default()
        )
        .is_err()
    );
    invalid = recipe();
    invalid.workspace_bytes = u64::MAX;
    assert!(
        build(
            &invalid,
            root.path(),
            &downloads,
            &PreparationControl::default()
        )
        .is_err()
    );
}
#[tokio::test]
async fn transaction_preserves_old_version_and_rejects_conflicting_owner() {
    let root = tempfile::tempdir().unwrap();
    let downloads = crate::downloads::Downloads::test(root.path());
    let recipe = recipe();
    let plan = recipe.plan(root.path(), &downloads).unwrap();
    let transaction = publication::Transaction::new(root.path(), &recipe, &plan).unwrap();
    assert!(publication::Transaction::new(root.path(), &recipe, &plan).is_err());
    let target = root.path().join(&recipe.target);
    std::fs::create_dir_all(&target).unwrap();
    std::fs::write(target.join("model.bin"), b"old").unwrap();
    std::fs::write(transaction.staging.path().join("model.bin"), b"abc").unwrap();
    let (tx, _) = mpsc::channel(1);
    assert!(
        transaction
            .publish(&recipe, &PreparationControl::default(), &tx)
            .is_err()
    );
    assert_eq!(std::fs::read(target.join("model.bin")).unwrap(), b"old");
}
#[tokio::test]
async fn failed_output_is_never_published_and_cancellation_is_checked() {
    let root = tempfile::tempdir().unwrap();
    let downloads = crate::downloads::Downloads::test(root.path());
    let recipe = recipe();
    let plan = recipe.plan(root.path(), &downloads).unwrap();
    let transaction = publication::Transaction::new(root.path(), &recipe, &plan).unwrap();
    std::fs::write(transaction.staging.path().join("model.bin"), b"bad").unwrap();
    let (tx, _) = mpsc::channel(1);
    assert!(
        transaction
            .publish(&recipe, &PreparationControl::default(), &tx)
            .is_err()
    );
    assert!(!root.path().join(&recipe.target).exists());
    let control = PreparationControl::default();
    control.cancel();
    assert!(verified(root.path(), 0, HASH, &control).is_err());
}
#[tokio::test]
async fn cancelling_worker_waits_for_cleanup_before_returning() {
    let control = PreparationControl::default();
    let entered = Arc::new(tokio::sync::Notify::new());
    let exited = Arc::new(AtomicBool::new(false));
    let (release, wait) = std::sync::mpsc::channel();
    let operation = control::worker(control.clone(), {
        let entered = entered.clone();
        let exited = exited.clone();
        move |control| {
            entered.notify_one();
            wait.recv().unwrap();
            exited.store(true, Ordering::SeqCst);
            control.check()?;
            Ok(())
        }
    });
    tokio::pin!(operation);
    tokio::select! { _ = entered.notified() => {}, _ = &mut operation => panic!("worker returned early") };
    control.cancel();
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(20), &mut operation)
            .await
            .is_err()
    );
    assert!(!exited.load(Ordering::SeqCst));
    release.send(()).unwrap();
    assert!(
        operation
            .await
            .unwrap_err()
            .downcast_ref::<PreparationError>()
            .is_some()
    );
    assert!(exited.load(Ordering::SeqCst));
}
#[tokio::test]
async fn completed_worker_does_not_cancel_following_stages() {
    let control = PreparationControl::default();
    assert_eq!(
        control::worker(control.clone(), |_| Ok(42)).await.unwrap(),
        42
    );
    control.check().unwrap();
    assert_eq!(
        control::worker(control.clone(), |_| Ok(43)).await.unwrap(),
        43
    );
    control.check().unwrap();
}
#[test]
fn cancel_and_publication_have_one_commit_boundary() {
    let control = PreparationControl::default();
    control.commit(|| Ok(())).unwrap();
    control.cancel();
    control.check().unwrap();
    let cancelled = PreparationControl::default();
    cancelled.cancel();
    let called = AtomicBool::new(false);
    assert!(
        cancelled
            .commit(|| {
                called.store(true, Ordering::SeqCst);
                Ok(())
            })
            .is_err()
    );
    assert!(!called.load(Ordering::SeqCst));
}
#[tokio::test]
async fn complete_transaction_publishes_only_verified_files() {
    let root = tempfile::tempdir().unwrap();
    let downloads = crate::downloads::Downloads::test(root.path());
    let recipe = recipe();
    let inputs = root.path().join(".native-preparation/inputs/fixture-v1");
    std::fs::create_dir_all(&inputs).unwrap();
    std::fs::write(inputs.join("source.bin"), b"abc").unwrap();
    let plan = recipe.plan(root.path(), &downloads).unwrap();
    let (progress, _) = mpsc::channel(1);
    let (files, _) = mpsc::channel(1);
    let control = PreparationControl::default();
    recipe
        .clone()
        .execute(
            root.path().into(),
            downloads,
            plan.fingerprint,
            control.clone(),
            progress,
            files,
        )
        .await
        .unwrap();
    assert_eq!(
        std::fs::read(root.path().join(&recipe.target).join("model.bin")).unwrap(),
        b"abc"
    );
    assert!(
        root.path()
            .join(&recipe.target)
            .join("preparation.json")
            .is_file()
    );
    assert_eq!(recipe.status(root.path()), crate::ResourceStatus::Ready);
    control.cancel();
    control.check().unwrap();
}
#[cfg(unix)]
#[test]
fn managed_directory_symlink_is_rejected_without_touching_external_files() {
    let root = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(external.path(), root.path().join(".native-preparation")).unwrap();
    let downloads = crate::downloads::Downloads::test(root.path());
    let recipe = recipe();
    let plan = recipe.plan(root.path(), &downloads).unwrap();
    assert!(publication::Transaction::new(root.path(), &recipe, &plan).is_err());
    assert_eq!(std::fs::read_dir(external.path()).unwrap().count(), 0);
}
#[tokio::test]
async fn changed_plan_is_rejected_before_creating_a_transaction() {
    let root = tempfile::tempdir().unwrap();
    let downloads = crate::downloads::Downloads::test(root.path());
    let (progress, _) = mpsc::channel(1);
    let (files, _) = mpsc::channel(1);
    let error = recipe()
        .execute(
            root.path().into(),
            downloads,
            "stale".into(),
            PreparationControl::default(),
            progress,
            files,
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error.downcast_ref::<PreparationError>(),
        Some(PreparationError::PlanChanged)
    ));
    assert!(!root.path().join(".native-preparation").exists());
}
#[test]
fn source_and_cache_changes_invalidate_confirmation() {
    let root = tempfile::tempdir().unwrap();
    let downloads = crate::downloads::Downloads::test(root.path());
    let old = recipe().plan(root.path(), &downloads).unwrap();
    let config = crate::downloads::DownloadConfig {
        cache_dir: downloads.cache_dir().into(),
        endpoint: Some("http://localhost:1234".into()),
        token: None,
    };
    let changed = crate::downloads::Downloads::new(&config).unwrap();
    assert_ne!(
        old.fingerprint,
        recipe().plan(root.path(), &changed).unwrap().fingerprint
    );
}
#[test]
fn space_failure_does_not_touch_an_existing_published_version() {
    let root = tempfile::tempdir().unwrap();
    let downloads = crate::downloads::Downloads::test(root.path());
    let recipe = recipe();
    let target = root.path().join(&recipe.target);
    std::fs::create_dir_all(&target).unwrap();
    std::fs::write(target.join("model.bin"), b"old").unwrap();
    let mut plan = recipe.plan(root.path(), &downloads).unwrap();
    plan.peak_additional_bytes = u64::MAX;
    let error = publication::Transaction::new(root.path(), &recipe, &plan)
        .err()
        .unwrap();
    assert!(matches!(
        error.downcast_ref::<PreparationError>(),
        Some(PreparationError::Space)
    ));
    assert_eq!(std::fs::read(target.join("model.bin")).unwrap(), b"old");
}
#[tokio::test]
async fn equal_size_corruption_is_refreshed_once_through_the_shared_client() {
    let server = httpmock::MockServer::start();
    server.mock(|when, then| {
        when.method("HEAD").path_includes("source.bin");
        then.status(200)
            .header("ETag", format!("\"{HASH}\""))
            .header("X-Repo-Commit", INPUTS[0].revision)
            .header("Content-Length", "3");
    });
    let body = server.mock(|when, then| {
        when.method("GET").path_includes("source.bin");
        then.status(200).body("abc");
    });
    let root = tempfile::tempdir().unwrap();
    let downloads = crate::downloads::Downloads::with_options(
        &crate::downloads::DownloadConfig {
            cache_dir: root.path().join("hf-cache"),
            endpoint: Some(server.base_url()),
            token: None,
        },
        crate::downloads::DownloadOptions {
            transport: crate::downloads::DownloadTransport::Http,
            ..Default::default()
        },
    )
    .unwrap();
    let input = root
        .path()
        .join(".native-preparation/inputs/fixture-v1/source.bin");
    let (tx, _) = mpsc::channel(16);
    downloads
        .ensure(
            &input,
            INPUTS[0].repo,
            INPUTS[0].revision,
            INPUTS[0].filename,
            crate::downloads::FileProgressBridge::new(tx.clone(), "fixture".into(), 3),
        )
        .await
        .unwrap();
    std::fs::write(&input, b"bad").unwrap();
    let recipe = recipe();
    let plan = recipe.plan(root.path(), &downloads).unwrap();
    assert_eq!(plan.download_bytes, 3);
    let (progress, _) = mpsc::channel(16);
    recipe
        .execute(
            root.path().into(),
            downloads,
            plan.fingerprint,
            PreparationControl::default(),
            progress,
            tx,
        )
        .await
        .unwrap();
    assert_eq!(std::fs::read(&input).unwrap(), b"abc");
    body.assert_calls(2);
}

#[tokio::test]
async fn retry_and_another_variant_reuse_only_a_complete_pinned_component() {
    let root = tempfile::tempdir().unwrap();
    let downloads = crate::downloads::Downloads::test(root.path());
    let inputs = root.path().join(".native-preparation/inputs/fixture-v1");
    std::fs::create_dir_all(&inputs).unwrap();
    std::fs::write(inputs.join("source.bin"), b"abc").unwrap();
    let mut failed = recipe();
    failed.transform = |work| {
        work.component("model.bin", |_, dest, _| {
            std::fs::write(dest, b"abc")?;
            Ok(())
        })?;
        anyhow::bail!("injected failure after complete component")
    };
    let plan = failed.plan(root.path(), &downloads).unwrap();
    let (tx, _) = mpsc::channel(1);
    assert!(
        failed
            .clone()
            .execute(
                root.path().into(),
                downloads.clone(),
                plan.fingerprint,
                PreparationControl::default(),
                tx.clone(),
                tx_for_files()
            )
            .await
            .is_err()
    );
    assert!(!root.path().join(&failed.target).exists());
    let cache = root
        .path()
        .join(".native-preparation/components")
        .join(HASH);
    assert_eq!(std::fs::read(&cache).unwrap(), b"abc");
    let mut variant = recipe();
    variant.target = "fixture/models/another/v1".into();
    variant.transform = |work| {
        work.component("model.bin", |_, _, _| {
            anyhow::bail!("cached component must not be converted again")
        })
    };
    let plan = variant.plan(root.path(), &downloads).unwrap();
    variant
        .clone()
        .execute(
            root.path().into(),
            downloads.clone(),
            plan.fingerprint,
            PreparationControl::default(),
            tx,
            tx_for_files(),
        )
        .await
        .unwrap();
    assert_eq!(
        std::fs::read(root.path().join(&variant.target).join("model.bin")).unwrap(),
        b"abc"
    );
    // An interrupted/corrupt cache file cannot skip the converter or publish.
    std::fs::remove_file(&cache).unwrap();
    std::fs::write(&cache, b"bad").unwrap();
    variant.target = "fixture/models/corrupt/v1".into();
    let plan = variant.plan(root.path(), &downloads).unwrap();
    let (tx, _) = mpsc::channel(1);
    assert!(
        variant
            .clone()
            .execute(
                root.path().into(),
                downloads,
                plan.fingerprint,
                PreparationControl::default(),
                tx,
                tx_for_files()
            )
            .await
            .is_err()
    );
    assert!(!root.path().join(&variant.target).exists());
}
fn tx_for_files() -> mpsc::Sender<talechime_protocol::Event> {
    mpsc::channel(1).0
}

#[test]
fn abandoned_cleanup_requires_lease_exact_recipe_and_known_contents() {
    let root = tempfile::tempdir().unwrap();
    let managed = root.path().join(".native-preparation");
    std::fs::create_dir(&managed).unwrap();
    let recipe = recipe();
    let marker = plan::marker(&recipe).unwrap();
    let abandoned = managed.join("staging-fixture-v1-abandoned");
    let unknown = managed.join("staging-fixture-v1-user-files");
    let foreign = managed.join("staging-other-v1-abandoned");
    for path in [&abandoned, &unknown, &foreign] {
        std::fs::create_dir(path).unwrap();
        std::fs::write(path.join("preparation.json"), &marker).unwrap();
        std::fs::write(path.join("model.bin"), b"partial").unwrap();
    }
    std::fs::write(unknown.join("user.notes"), b"preserve").unwrap();
    let downloads = crate::downloads::Downloads::test(root.path());
    let plan = recipe.plan(root.path(), &downloads).unwrap();
    let transaction = publication::Transaction::new(root.path(), &recipe, &plan).unwrap();
    assert!(!abandoned.exists());
    assert_eq!(
        std::fs::read(unknown.join("user.notes")).unwrap(),
        b"preserve"
    );
    assert!(foreign.exists());
    let active_orphan = managed.join("staging-fixture-v1-while-locked");
    std::fs::create_dir(&active_orphan).unwrap();
    std::fs::write(active_orphan.join("preparation.json"), &marker).unwrap();
    assert!(publication::Transaction::new(root.path(), &recipe, &plan).is_err());
    assert!(active_orphan.exists());
    drop(transaction);
    let transaction = publication::Transaction::new(root.path(), &recipe, &plan).unwrap();
    assert!(!active_orphan.exists());
    drop(transaction);
}
#[cfg(unix)]
#[test]
fn cleanup_never_follows_a_marked_stage_symlink() {
    let root = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    std::fs::write(external.path().join("model.bin"), b"user audio").unwrap();
    let managed = root.path().join(".native-preparation");
    let stage = managed.join("staging-fixture-v1-link");
    std::fs::create_dir_all(&stage).unwrap();
    std::fs::write(
        stage.join("preparation.json"),
        plan::marker(&recipe()).unwrap(),
    )
    .unwrap();
    std::os::unix::fs::symlink(external.path().join("model.bin"), stage.join("model.bin")).unwrap();
    let downloads = crate::downloads::Downloads::test(root.path());
    let recipe = recipe();
    let plan = recipe.plan(root.path(), &downloads).unwrap();
    let transaction = publication::Transaction::new(root.path(), &recipe, &plan).unwrap();
    assert!(stage.exists());
    assert_eq!(
        std::fs::read(external.path().join("model.bin")).unwrap(),
        b"user audio"
    );
    drop(transaction);
}
#[tokio::test]
async fn publication_failure_keeps_a_previously_ready_version_readable() {
    let root = tempfile::tempdir().unwrap();
    let downloads = crate::downloads::Downloads::test(root.path());
    let old = recipe();
    let plan = old.plan(root.path(), &downloads).unwrap();
    let transaction = publication::Transaction::new(root.path(), &old, &plan).unwrap();
    std::fs::write(transaction.staging.path().join("model.bin"), b"abc").unwrap();
    let (tx, _) = mpsc::channel(1);
    transaction
        .publish(&old, &PreparationControl::default(), &tx)
        .unwrap();
    assert_eq!(old.status(root.path()), crate::ResourceStatus::Ready);
    let mut new = old.clone();
    new.target = "fixture/models/blocked/v2".into();
    let plan = new.plan(root.path(), &downloads).unwrap();
    let transaction = publication::Transaction::new(root.path(), &new, &plan).unwrap();
    std::fs::write(transaction.staging.path().join("model.bin"), b"abc").unwrap();
    std::fs::write(
        root.path().join("fixture/models/blocked"),
        b"injected disk/path failure",
    )
    .unwrap();
    assert!(
        transaction
            .publish(&new, &PreparationControl::default(), &tx)
            .is_err()
    );
    assert_eq!(old.status(root.path()), crate::ResourceStatus::Ready);
    assert_eq!(
        std::fs::read(root.path().join(&old.target).join("model.bin")).unwrap(),
        b"abc"
    );
}

#[tokio::test]
async fn cancelled_conversion_keeps_the_lease_until_the_blocking_worker_exits() {
    let root = tempfile::tempdir().unwrap();
    let downloads = crate::downloads::Downloads::test(root.path());
    let inputs = root.path().join(".native-preparation/inputs/fixture-v1");
    std::fs::create_dir_all(&inputs).unwrap();
    std::fs::write(inputs.join("source.bin"), b"abc").unwrap();
    let mut blocked = recipe();
    blocked.transform = |work| {
        work.component("model.bin", |input, _, control| {
            std::fs::write(input.join("entered"), b"started")?;
            while !input.join("release").exists() {
                std::thread::park_timeout(std::time::Duration::from_millis(1));
            }
            control.check()?;
            anyhow::bail!("cancelled work must not reach output")
        })
    };
    let plan = blocked.plan(root.path(), &downloads).unwrap();
    let control = PreparationControl::default();
    let (stages, mut events) = mpsc::channel(32);
    let mut operation = tokio::spawn(blocked.clone().execute(
        root.path().into(),
        downloads.clone(),
        plan.fingerprint.clone(),
        control.clone(),
        stages,
        tx_for_files(),
    ));
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        while !inputs.join("entered").exists() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    control.cancel();
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(20), &mut operation)
            .await
            .is_err()
    );
    let error = publication::Transaction::new(root.path(), &blocked, &plan)
        .err()
        .unwrap();
    assert!(matches!(
        error.downcast_ref::<PreparationError>(),
        Some(PreparationError::Busy)
    ));
    assert!(!root.path().join(&blocked.target).exists());
    std::fs::write(inputs.join("release"), b"stop").unwrap();
    let result = operation.await.unwrap().unwrap_err();
    assert!(matches!(
        result.downcast_ref::<PreparationError>(),
        Some(PreparationError::Cancelled)
    ));
    while events.recv().await.is_some() {}
    assert!(events.is_closed());
    assert!(!root.path().join(&blocked.target).exists());
    let transaction = publication::Transaction::new(root.path(), &blocked, &plan).unwrap();
    drop(transaction);
}
#[test]
fn registry_recreation_does_not_resume_an_interrupted_preparation() {
    let root = tempfile::tempdir().unwrap();
    let stage = root
        .path()
        .join(".native-preparation/staging-fixture-v1-interrupted");
    std::fs::create_dir_all(&stage).unwrap();
    std::fs::write(
        stage.join("preparation.json"),
        plan::marker(&recipe()).unwrap(),
    )
    .unwrap();
    std::fs::write(stage.join("model.bin"), b"partial").unwrap();
    let _registry = crate::Registry::new(Some(root.path().into())).unwrap();
    assert_eq!(std::fs::read(stage.join("model.bin")).unwrap(), b"partial");
    assert!(!root.path().join(recipe().target).exists());
    assert!(!root.path().join(".native-preparation/lease").exists());
}

#[test]
fn confirmation_binds_transfer_policy_and_credentials_without_exposing_them() {
    let root = tempfile::tempdir().unwrap();
    let base = crate::downloads::DownloadConfig {
        cache_dir: root.path().join("hf-cache"),
        endpoint: None,
        token: None,
    };
    let defaults = crate::downloads::Downloads::new(&base).unwrap();
    let plan = recipe().plan(root.path(), &defaults).unwrap();
    let http = crate::downloads::Downloads::with_options(
        &base,
        crate::downloads::DownloadOptions {
            connections: 1,
            transport: crate::downloads::DownloadTransport::Http,
        },
    )
    .unwrap();
    assert_ne!(
        recipe().plan(root.path(), &http).unwrap().fingerprint,
        plan.fingerprint
    );
    let secret = "hf_synthetic_never_a_real_credential";
    let config = crate::downloads::DownloadConfig {
        token: Some(secret.into()),
        ..base
    };
    let client = crate::downloads::Downloads::new(&config).unwrap();
    let changed = recipe().plan(root.path(), &client).unwrap();
    assert_ne!(changed.fingerprint, plan.fingerprint);
    assert!(!format!("{config:?} {client:?}").contains(secret));
    assert!(!serde_json::to_string(&changed).unwrap().contains(secret));
}

#[tokio::test]
async fn finished_distribution_reuses_verified_source_components_without_conversion_or_network() {
    static FINISHED: &[Input] = &[Input {
        path: "model.bin",
        ..INPUTS[0]
    }];
    let root = tempfile::tempdir().unwrap();
    let downloads = crate::downloads::Downloads::test(root.path());
    let components = root.path().join(".native-preparation/components");
    std::fs::create_dir_all(&components).unwrap();
    std::fs::write(components.join(HASH), b"abc").unwrap();
    let recipe = recipe().distributed("finished-v3", FINISHED, "fixture/models/one/v3".into());
    let plan = recipe.plan(root.path(), &downloads).unwrap();
    assert_eq!(plan.download_bytes, 0);
    assert!(plan.conversions.is_empty());
    assert_eq!(plan.peak_additional_bytes, 3);
    let (progress, mut stages) = tokio::sync::mpsc::channel(32);
    let (files, _) = tokio::sync::mpsc::channel(32);
    recipe
        .clone()
        .execute(
            root.path().into(),
            downloads,
            plan.fingerprint,
            PreparationControl::default(),
            progress,
            files,
        )
        .await
        .unwrap();
    while let Some(stage) = stages.recv().await {
        assert_ne!(stage.phase, PreparationPhase::Converting);
    }
    assert_eq!(
        std::fs::read(root.path().join(&recipe.target).join("model.bin")).unwrap(),
        b"abc"
    );
    recipe
        .verify_files(
            &root.path().join(&recipe.target),
            &PreparationControl::default(),
        )
        .unwrap();
}

#[tokio::test]
async fn missing_auxiliary_and_mixed_publication_marker_never_publish() {
    static FILES: &[Output] = &[
        Output {
            path: "model.bin",
            size: 3,
            sha256: HASH,
        },
        Output {
            path: "tokenizer.json",
            size: 3,
            sha256: HASH,
        },
    ];
    let root = tempfile::tempdir().unwrap();
    let downloads = crate::downloads::Downloads::test(root.path());
    let mut recipe = recipe();
    recipe.outputs = FILES;
    let preview = recipe.plan(root.path(), &downloads).unwrap();
    let transaction = publication::Transaction::new(root.path(), &recipe, &preview).unwrap();
    std::fs::write(transaction.staging.path().join("model.bin"), b"abc").unwrap();
    let (tx, _) = mpsc::channel(8);
    assert!(
        transaction
            .publish(&recipe, &PreparationControl::default(), &tx)
            .is_err()
    );
    assert!(!root.path().join(&recipe.target).exists());
    let transaction = publication::Transaction::new(root.path(), &recipe, &preview).unwrap();
    for pin in FILES {
        std::fs::write(transaction.staging.path().join(pin.path), b"abc").unwrap();
    }
    let mut foreign = recipe.clone();
    foreign.id = "foreign-recipe-v2";
    std::fs::write(
        transaction.staging.path().join("preparation.json"),
        plan::marker(&foreign).unwrap(),
    )
    .unwrap();
    assert!(
        transaction
            .publish(&recipe, &PreparationControl::default(), &tx)
            .is_err()
    );
    assert!(!root.path().join(&recipe.target).exists());
}

#[tokio::test]
async fn unavailable_pinned_source_reports_failure_without_publishing_or_touching_other_versions() {
    let server = httpmock::MockServer::start();
    let missing = server.mock(|when, then| {
        when.path_includes("source.bin");
        then.status(404);
    });
    let root = tempfile::tempdir().unwrap();
    let old = root.path().join("other-version");
    std::fs::write(&old, b"published old bytes").unwrap();
    let downloads = crate::downloads::Downloads::with_options(
        &crate::downloads::DownloadConfig {
            cache_dir: root.path().join("hf"),
            endpoint: Some(server.base_url()),
            token: None,
        },
        crate::downloads::DownloadOptions {
            transport: crate::downloads::DownloadTransport::Http,
            ..Default::default()
        },
    )
    .unwrap();
    let recipe = recipe();
    let preview = recipe.plan(root.path(), &downloads).unwrap();
    let (stages, _) = mpsc::channel(16);
    let (files, _) = mpsc::channel(16);
    assert!(
        recipe
            .clone()
            .execute(
                root.path().into(),
                downloads,
                preview.fingerprint,
                PreparationControl::default(),
                stages,
                files
            )
            .await
            .is_err()
    );
    assert!(!root.path().join(&recipe.target).exists());
    assert_eq!(std::fs::read(old).unwrap(), b"published old bytes");
    assert!(missing.calls() >= 1);
}

#[cfg(all(feature = "kokoro-g2pw", feature = "cosyvoice3", feature = "indextts"))]
#[test]
fn runtime_distribution_covers_exact_outputs_or_pinned_native_transforms() {
    let recipes = [
        crate::kokoro::native_recipe(crate::kokoro::MODEL).unwrap(),
        crate::kokoro::native_recipe(crate::kokoro::CONTEXT_MODEL).unwrap(),
        crate::cosyvoice3::native_recipe(),
        crate::indextts::native_recipe(),
    ];
    for recipe in recipes {
        if recipe.transform_outputs.is_empty() {
            assert_eq!(recipe.workspace_bytes, 0);
            assert_eq!(recipe.inputs.len(), recipe.outputs.len());
        }
        for output in recipe.outputs {
            if recipe.transform_outputs.contains(&output.path) {
                continue;
            }
            if recipe.id.starts_with("indextts-")
                && matches!(output.path, "SOURCES.json" | "LICENSE-UNIDIC")
            {
                let compiled: &[u8] = if output.path == "SOURCES.json" {
                    include_bytes!("../indextts/native-sources.json")
                } else {
                    include_bytes!("../indextts/native/UNIDIC-BSD")
                };
                use sha2::Digest;
                assert_eq!(compiled.len() as u64, output.size);
                assert_eq!(
                    format!("{:x}", sha2::Sha256::digest(compiled)),
                    output.sha256
                );
                continue;
            }
            let input = recipe
                .inputs
                .iter()
                .find(|pin| pin.path == output.path)
                .unwrap();
            assert_eq!((input.size, input.sha256), (output.size, output.sha256));
            assert!(!input.filename.ends_with(".pt") && !input.filename.ends_with(".pth"));
            assert_eq!(input.revision.len(), 40);
        }
    }
}

#[tokio::test]
async fn equal_size_cache_swaps_require_new_confirmation_before_any_download() {
    static PAIR: &[Input] = &[
        Input {
            path: "first.bin",
            filename: "first.bin",
            ..INPUTS[0]
        },
        Input {
            path: "second.bin",
            filename: "second.bin",
            sha256: "cb8379ac2098aa165029e3938a51da0bcecfc008fd6795f401178647f96c5b34",
            ..INPUTS[0]
        },
    ];
    let root = tempfile::tempdir().unwrap();
    let server = httpmock::MockServer::start();
    let requests = server.mock(|when, then| {
        when.any_request();
        then.status(404);
    });
    let downloads = crate::downloads::Downloads::with_options(
        &crate::downloads::DownloadConfig {
            cache_dir: root.path().join("hf-cache"),
            endpoint: Some(server.base_url()),
            token: None,
        },
        crate::downloads::DownloadOptions {
            transport: crate::downloads::DownloadTransport::Http,
            ..Default::default()
        },
    )
    .unwrap();
    let mut recipe = recipe();
    recipe.inputs = PAIR;
    let snapshots = downloads
        .cache_dir()
        .join("models--fixture--model/snapshots")
        .join(PAIR[0].revision);
    std::fs::create_dir_all(&snapshots).unwrap();
    std::fs::write(snapshots.join("first.bin"), b"abc").unwrap();
    let original = recipe.plan(root.path(), &downloads).unwrap();
    std::fs::remove_file(snapshots.join("first.bin")).unwrap();
    std::fs::write(snapshots.join("second.bin"), b"def").unwrap();
    let changed = recipe.plan(root.path(), &downloads).unwrap();
    assert_eq!(original.download_bytes, changed.download_bytes);
    assert_eq!(
        original.peak_additional_bytes,
        changed.peak_additional_bytes
    );
    assert_eq!(original.conversions, changed.conversions);
    assert_ne!(original.fingerprint, changed.fingerprint);
    let (stages, _) = mpsc::channel(8);
    let (files, _) = mpsc::channel(8);
    let error = recipe
        .clone()
        .execute(
            root.path().into(),
            downloads.clone(),
            original.fingerprint,
            PreparationControl::default(),
            stages,
            files,
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error.downcast_ref::<PreparationError>(),
        Some(PreparationError::PlanChanged)
    ));
    assert!(!root.path().join(".native-preparation").exists());
    requests.assert_calls(0);

    // A published output needs no input downloads, so irrelevant input-cache changes do not invalidate it.
    let published = root.path().join(&recipe.target);
    std::fs::create_dir_all(&published).unwrap();
    std::fs::write(published.join("model.bin"), b"abc").unwrap();
    std::fs::write(
        published.join("preparation.json"),
        plan::marker(&recipe).unwrap(),
    )
    .unwrap();
    let ready = recipe.plan(root.path(), &downloads).unwrap();
    std::fs::remove_file(snapshots.join("second.bin")).unwrap();
    std::fs::write(snapshots.join("first.bin"), b"abc").unwrap();
    let ready_again = recipe.plan(root.path(), &downloads).unwrap();
    assert!(ready.ready && ready_again.ready);
    assert_eq!(ready.fingerprint, ready_again.fingerprint);
}
