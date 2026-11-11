"""Patch cargo-dist 0.32.0 global installers without replacing their templates."""
import argparse
import hashlib
import json
from pathlib import Path
import re

VERSION = 'espeak-ng-1.52.0'
SHELL_ANCHOR = '    local _arch="$5"\n    for _bin_name in $_bins; do\n'
SHELL_PATCH = '''    local _arch="$5"
    # Talechime: stage the complete separately licensed helper beside the binary.
    local _helper_src="$_src_dir/talechime-helpers/espeak-ng-1.52.0"
    local _helper_parent="$_install_dir/talechime-helpers"
    local _helper_dest="$_helper_parent/espeak-ng-1.52.0"
    [ -f "$_helper_src/manifest.json" ] || err "missing eSpeak helper manifest"
    [ -d "$_helper_src/espeak-ng-data" ] || err "missing eSpeak helper data"
    [ -f "$_helper_src/source/espeak-ng-1.52.0.tar.gz" ] || err "missing eSpeak corresponding source"
    [ -f "$_helper_src/source/COPYING" ] || err "missing eSpeak license"
    [ -f "$_helper_src/source/kokoro_espeak_phonemize.c" ] || err "missing eSpeak phonemizer source"
    if [ ! -f "$_helper_src/kokoro-espeak-phonemize" ] && [ ! -f "$_helper_src/kokoro-espeak-phonemize.exe" ]; then
        err "missing eSpeak phonemizer"
    fi
    if [ ! -f "$_helper_src/espeak-ng" ] && [ ! -f "$_helper_src/espeak-ng.exe" ]; then
        err "missing eSpeak helper executable"
    fi
    ensure mkdir -p "$_helper_parent"
    ensure cp -R "$_helper_src" "$_install_temp/espeak-ng-1.52.0"
    if [ -e "$_helper_dest" ]; then
        ensure mv "$_helper_dest" "$_install_temp/previous-espeak-ng"
    fi
    if ! mv "$_install_temp/espeak-ng-1.52.0" "$_helper_dest"; then
        if [ -e "$_install_temp/previous-espeak-ng" ]; then
            mv "$_install_temp/previous-espeak-ng" "$_helper_dest" || err "cannot restore previous eSpeak helper"
        fi
        err "cannot install eSpeak helper"
    fi
    for _bin_name in $_bins; do
'''
POWERSHELL_ANCHOR = '  # Just copy the binaries from the temp location to the install dir\n'
POWERSHELL_PATCH = '''  # Talechime: stage the complete separately licensed helper beside the binary.
  if (@($artifacts["bin_paths"]).Count -ne 1) { throw "Expected one Talechime binary" }
  $helper_src = Join-Path (Split-Path -Parent $artifacts["bin_paths"][0]) "talechime-helpers\\espeak-ng-1.52.0"
  foreach ($part in @("espeak-ng.exe", "kokoro-espeak-phonemize.exe", "espeak-ng-data", "manifest.json", "source\\espeak-ng-1.52.0.tar.gz", "source\\COPYING", "source\\kokoro_espeak_phonemize.c")) {
    if (-not (Test-Path -LiteralPath (Join-Path $helper_src $part))) { throw "Missing eSpeak helper component: $part" }
  }
  $helper_parent = Join-Path $dest_dir "talechime-helpers"
  $null = New-Item -Force -ItemType Directory -Path $helper_parent -ErrorAction Stop
  $stage = Join-Path $helper_parent (".espeak-stage-" + [guid]::NewGuid().ToString("N"))
  $backup = Join-Path $helper_parent (".espeak-backup-" + [guid]::NewGuid().ToString("N"))
  $helper_dest = Join-Path $helper_parent "espeak-ng-1.52.0"
  try {
    Copy-Item -LiteralPath $helper_src -Destination $stage -Recurse -ErrorAction Stop
    if (Test-Path -LiteralPath $helper_dest) { Move-Item -LiteralPath $helper_dest -Destination $backup -ErrorAction Stop }
    try { Move-Item -LiteralPath $stage -Destination $helper_dest -ErrorAction Stop }
    catch {
      if (Test-Path -LiteralPath $backup) { Move-Item -LiteralPath $backup -Destination $helper_dest -ErrorAction Stop }
      throw
    }
    if (Test-Path -LiteralPath $backup) { Remove-Item -LiteralPath $backup -Recurse -Force -ErrorAction Stop }
  } finally {
    if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction Stop }
  }
  # Just copy the binaries from the temp location to the install dir
'''
BREW_ANCHOR = '    install_binary_aliases!\n\n    # Homebrew will automatically install these'
BREW_PATCH = '''    # Keep the standalone helper and corresponding source adjacent to the executable.
    raise "missing eSpeak helper" unless (buildpath/"talechime-helpers/espeak-ng-1.52.0/manifest.json").file? &&
      (buildpath/"talechime-helpers/espeak-ng-1.52.0/espeak-ng").file? &&
      (buildpath/"talechime-helpers/espeak-ng-1.52.0/kokoro-espeak-phonemize").file? &&
      (buildpath/"talechime-helpers/espeak-ng-1.52.0/espeak-ng-data").directory? &&
      (buildpath/"talechime-helpers/espeak-ng-1.52.0/source/espeak-ng-1.52.0.tar.gz").file? &&
      (buildpath/"talechime-helpers/espeak-ng-1.52.0/source/COPYING").file? &&
      (buildpath/"talechime-helpers/espeak-ng-1.52.0/source/kokoro_espeak_phonemize.c").file?
    bin.install "talechime-helpers"

    install_binary_aliases!

    # Homebrew will automatically install these'''


def patch_once(text, before, after, name):
    if text.count(before) != 1:
        raise ValueError(f'{name}: expected one unmodified cargo-dist 0.32.0 anchor')
    return text.replace(before, after, 1)


def patch_installers(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('dist_version') != '0.32.0':
        raise ValueError('installer patch requires cargo-dist 0.32.0')
    artifacts = manifest['artifacts']
    uploads = {Path(path).resolve() for path in manifest['upload_files']}
    specs = [('talechime-installer.sh', SHELL_ANCHOR, SHELL_PATCH),
             ('talechime-installer.ps1', POWERSHELL_ANCHOR, POWERSHELL_PATCH),
             ('talechime.rb', BREW_ANCHOR, BREW_PATCH)]
    changes = {}
    for name, anchor, replacement in specs:
        artifact = artifacts[name]
        path = Path(artifact['path'])
        if artifact['kind'] != 'installer' or path.name != name or path.resolve() not in uploads:
            raise ValueError(f'{name}: unexpected installer manifest')
        patched = patch_once(path.read_text(encoding='utf-8'), anchor, replacement, name)
        if name == 'talechime.rb':
            patched = patch_once(patched, '  license "MIT"\n',
                                 '  license all_of: ["MIT", "GPL-3.0-or-later"]\n', name)
        changes[path] = patched.encode('utf-8')
    digests = {path.name: hashlib.sha256(data).hexdigest() for path, data in changes.items()}
    checksum_path = Path(artifacts['sha256.sum']['path'])
    if checksum_path.name != 'sha256.sum' or checksum_path.resolve() not in uploads:
        raise ValueError('unexpected unified checksum artifact')
    lines = [line for line in checksum_path.read_text(encoding='utf-8').splitlines() if line]
    names_seen = set()
    for line in lines:
        match = re.fullmatch(r'([0-9a-fA-F]{64}) [ *](.+)', line)
        if not match or match.group(2) in names_seen:
            raise ValueError('invalid or duplicate unified checksum entry')
        names_seen.add(match.group(2))
    lines = [line for line in lines if line[66:] not in digests]
    lines.extend(f'{digest} *{name}' for name, digest in digests.items())
    for name, digest in digests.items():
        artifact = artifacts[name]
        artifact.setdefault('checksums', {})['sha256'] = digest
        if 'checksum' in artifact:
            sidecar = artifacts[artifact['checksum']]
            side_path = Path(sidecar['path'])
            if side_path.resolve() not in uploads or side_path.name != name + '.sha256':
                raise ValueError(f'{name}: unexpected checksum sidecar')
            changes[side_path] = f'{digest} *{name}\n'.encode()
    changes[checksum_path] = ('\n'.join(lines) + '\n').encode()
    changes[manifest_path] = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    for path, contents in changes.items():
        path.write_bytes(contents)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    patch_installers(parser.parse_args().manifest)
