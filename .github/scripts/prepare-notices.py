"""Collect source license/provenance notices for cargo-dist archives."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[2]
for path in (root / 'crates').rglob('*'):
    if path.is_file() and (path.name.startswith(('LICENSE', 'COPYING'))
                          or path.name in ('NOTICE', 'SOURCE.md')):
        destination = root / 'THIRD_PARTY_LICENSES' / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
