"""Collect source license/provenance notices for cargo-dist archives."""
from pathlib import Path
import json
import shutil
import subprocess
import tomllib

root = Path(__file__).resolve().parents[2]
output = root / 'target' / 'third-party-licenses'


def collect(directory, destination):
    for path in directory.rglob('*'):
        if path.is_file() and (path.name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE'))
                              or path.name == 'SOURCE.md'):
            target = destination / path.relative_to(directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


collect(root / 'crates', output / 'crates')
features = tomllib.loads((root / 'dist-workspace.toml').read_text())['dist']['features']
host = next(line.removeprefix('host: ') for line in subprocess.check_output(
    ['rustc', '-vV'], text=True).splitlines() if line.startswith('host: '))
metadata = json.loads(subprocess.check_output([
    'cargo', 'metadata', '--locked', '--format-version', '1', '--filter-platform', host,
    '--features', ','.join('talechime/' + feature for feature in features)], cwd=root))
records = []
for package in metadata['packages']:
    if package['source'] is None:
        continue
    collect(Path(package['manifest_path']).parent,
            output / 'dependencies' / (package['name'] + '-' + package['version']))
    records.append({key: package[key] for key in ['name', 'version', 'license', 'repository', 'source']})
output.mkdir(parents=True, exist_ok=True)
(output / 'dependencies.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
