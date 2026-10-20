//! Offline catalogue for developer tooling. Never prepares or migrates resources.
use tts_core::paths::AppPaths;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let paths = match args.next() {
        Some(root) => {
            // A custom root is already an application root, not another HOME.
            let root = std::path::PathBuf::from(root);
            if let Some(backend) = args.next() {
                let model = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("model required"))?;
                anyhow::ensure!(
                    args.next().is_none(),
                    "usage: performance_catalog [APP_ROOT [BACKEND MODEL]]"
                );
                let registry = talechime_backends::Registry::new(Some(root.join("resources")))?;
                let receipt = talechime_backends::performance::verify(
                    &registry,
                    backend
                        .to_str()
                        .ok_or_else(|| anyhow::anyhow!("UTF-8 backend required"))?,
                    model
                        .to_str()
                        .ok_or_else(|| anyhow::anyhow!("UTF-8 model required"))?,
                )?;
                println!("{}", serde_json::to_string(&receipt)?);
                return Ok(());
            }
            return catalog(root.clone(), root.join("resources"));
        }
        None => AppPaths::user_default()?,
    };
    catalog(paths.root().to_owned(), paths.resources())
}

fn catalog(root: std::path::PathBuf, resources: std::path::PathBuf) -> anyhow::Result<()> {
    let registry = talechime_backends::Registry::new(Some(resources.clone()))?;
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "schema": "talechime/performance-catalog/1",
            "root": root,
            "resources": resources,
            "catalog": registry.catalog()?,
            "path_resolution": "pure_no_migration",
            "resource_acquisition": false
        }))?
    );
    Ok(())
}
