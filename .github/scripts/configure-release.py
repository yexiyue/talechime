"""Select native release features from the single dist workspace configuration."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib


def configure(root: Path, target: str) -> list[str]:
    path = root / 'dist-workspace.toml'
    source = path.read_text(encoding='utf-8')
    config = tomllib.loads(source)
    if target not in config['dist']['targets']:
        raise ValueError(f'Unsupported distribution target: {target}')
    metal = config['talechime-release']['macos-features']
    features = [feature for feature in config['dist']['features'] if feature not in metal]
    if target.endswith('-apple-darwin'):
        features += metal
    source, count = re.subn(r'^features = .*$', 'features = ' + json.dumps(features),
                            source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError('Expected exactly one dist features declaration')
    path.write_text(source, encoding='utf-8')
    return features


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    target = sys.argv[1] if len(sys.argv) > 1 else next(
        line.removeprefix('host: ') for line in subprocess.check_output(
            ['rustc', '-vV'], text=True).splitlines() if line.startswith('host: '))
    print(f'{target}: {configure(root, target)}')
