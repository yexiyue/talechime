"""Build and smoke-test a standalone CPU archive without downloading models."""
from pathlib import Path
import hashlib
import shutil
import subprocess
import sys
from smoke import smoke

root = Path(__file__).resolve().parents[1]
target = sys.argv[1]
subprocess.run(['rustup', 'target', 'add', target], check=True)
subprocess.run(['cargo', 'build', '--locked', '--release', '--target', target,
                '-p', 'talechime', '--bins'], cwd=root, check=True)
build = root / 'target' / target / 'release'
output = root / 'dist'
output.mkdir(exist_ok=True)
stage = root / 'target' / 'package' / f'talechime-{target}'
stage.mkdir(parents=True, exist_ok=True)
suffix = '.exe' if 'windows' in target else ''
for name in ['talechime', 'novel-tts']:
    program = build / (name + suffix)
    subprocess.run([str(program), '--version'], check=True)
    shutil.copy2(program, stage / program.name)
for path in build.iterdir():
    if path.is_file() and (path.suffix in ['.dll', '.dylib'] or '.so' in path.name):
        shutil.copy2(path, stage / path.name)
for name in ['README.md', 'LICENSE', 'SOURCE.md']:
    shutil.copy2(root / name, stage / name)
for name in ['docs', 'assets', 'dev-notes']:
    shutil.copytree(root / name, stage / name, dirs_exist_ok=True)
for path in (root / 'crates').rglob('*'):
    if path.is_file() and (path.name.startswith(('LICENSE', 'COPYING'))
                          or path.name in ['NOTICE', 'SOURCE.md']):
        destination = stage / 'notices' / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
smoke(stage, suffix)
archive_format = 'zip' if suffix else 'gztar'
archive = Path(shutil.make_archive(str(output / stage.name), archive_format,
                                 root_dir=stage.parent, base_dir=stage.name))
archive.with_name(archive.name + '.sha256').write_text(
    hashlib.sha256(archive.read_bytes()).hexdigest() + '  ' + archive.name + '\n')
print(archive)
