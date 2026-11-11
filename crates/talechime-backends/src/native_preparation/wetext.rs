//! Shared extraction of fixed WeText wheel data; callers own text policies.
use super::{Input, PreparationControl, Workspace};
use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};
pub(crate) const INPUT: Input = Input {
    path: "wetext-0.1.0.whl",
    repo: "pypi/wetext",
    revision: "0.1.0",
    filename: "wetext-0.1.0-py3-none-any.whl",
    size: 1513305,
    sha256: "cfc595f85d9b0218df6746a9e6bee9240b6e09ee6d803a937ef90cbc03936b3e",
    url: Some(
        "https://files.pythonhosted.org/packages/d4/a6/d5856069473e04968cd04fdd4726a23e78fd925ced0281294a28c5a27913/wetext-0.1.0-py3-none-any.whl",
    ),
};
pub(crate) const OUTPUTS: &[crate::native_preparation::Output] = &[
    crate::native_preparation::Output {
        path: "tn/wetext/zh/tn/tagger.fst",
        size: 811646,
        sha256: "63f3abeafe64fc3e9232099ab8ef6efd44c5198319193854af3ac2716d7c27a7",
    },
    crate::native_preparation::Output {
        path: "tn/wetext/zh/tn/verbalizer.fst",
        size: 87702,
        sha256: "9fcadad76cafddb2e96b92d892d18249f0ba5fbe60e4d4fc51f466cb6bc7ade9",
    },
    crate::native_preparation::Output {
        path: "tn/wetext/en/tn/tagger.fst",
        size: 5220942,
        sha256: "bb804f88030153999f154f01950c18dbfd64de2c4cfc5035e21371d4eb4ead8a",
    },
    crate::native_preparation::Output {
        path: "tn/wetext/en/tn/verbalizer.fst",
        size: 1822714,
        sha256: "33ad65460a3e8f7fe313994ff0204a218c0fcb5dec9c09b24aeeaa30ef21d67b",
    },
    crate::native_preparation::Output {
        path: "LICENSE-WETEXT",
        size: 11357,
        sha256: "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    },
];

pub(crate) fn prepare(work: &Workspace<'_>) -> anyhow::Result<()> {
    for pin in OUTPUTS {
        work.component(pin.path, |input, output, control| {
            extract(input, output, pin.path, pin.size, control)
        })?;
    }
    Ok(())
}
fn extract(
    input: &Path,
    output: &Path,
    name: &str,
    size: u64,
    control: &PreparationControl,
) -> anyhow::Result<()> {
    let entry_name = if let Some(suffix) = name.strip_prefix("tn/wetext/") {
        format!("wetext/fsts/{suffix}")
    } else {
        anyhow::ensure!(name == "LICENSE-WETEXT", "unknown WeText resource");
        "wetext-0.1.0.dist-info/licenses/LICENSE".into()
    };
    let mut archive = zip::ZipArchive::new(File::open(input.join(INPUT.path))?)?;
    let mut entry = archive.by_name(&entry_name)?;
    anyhow::ensure!(entry.size() == size, "WeText resource size mismatch");
    let mut destination = File::create(output)?;
    let mut buffer = [0; 65536];
    loop {
        control.check()?;
        let size = entry.read(&mut buffer)?;
        if size == 0 {
            break;
        }
        destination.write_all(&buffer[..size])?;
    }
    Ok(())
}
