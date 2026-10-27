"""Build cargo-dist artifacts and exercise an isolated, model-free installation."""
import json
import hashlib
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
from threading import Thread
from smoke import protocol_version
import argparse


def run(*args, **kwargs):
    try:
        return subprocess.run(args, check=True, timeout=60, **kwargs)
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, file=sys.stderr)
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        raise


def main(app, target, skip_build=False, build_installers=False):
    if app != 'talechime':
        raise ValueError('Only the Talechime distribution is supported')
    if not skip_build:
        run(sys.executable, '.github/scripts/configure-release.py', target)
        run('dist', 'generate', '--check')
        subprocess.run([sys.executable, '.github/scripts/prepare-notices.py'], check=True)
        subprocess.run(['dist', 'build', '--artifacts', 'local', '--target', target], check=True)
        run('dist', 'build', '--artifacts', 'global')
    elif build_installers:
        # Global artifacts are deterministic from this source/tag and the local
        # manifests. Generate them here while the release's global job runs.
        run('dist', 'build', '--artifacts', 'global', '--tag', os.environ['RELEASE_TAG'])
    archives = list(Path('target/distrib').glob(f'{app}-{target}.*'))
    archive = next(path for path in archives if path.suffix == '.zip' or path.name.endswith('.tar.xz'))
    expected_hash = Path(str(archive) + '.sha256').read_text(encoding='utf-8').split()[0]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected_hash, 'Archive checksum mismatch'
    version = tomllib.loads(Path('crates/talechime/Cargo.toml').read_text(encoding='utf-8'))['package']['version']
    with tempfile.TemporaryDirectory(prefix='isolated install ') as temporary:
        base = Path(temporary)
        shutil.unpack_archive(archive, base / 'install')
        suffix = '.exe' if sys.platform == 'win32' else ''
        programs = [app]
        installed = next((base / 'install').rglob(app + suffix)).parent
        files = list((base / 'install').rglob('*'))
        assert not any((path.suffix.lower() in ['.dll', '.dylib'] or '.so' in path.name)
                       and ('cuda' in path.name.lower() or 'cudnn' in path.name.lower())
                       for path in files), 'Standard archive contains CUDA libraries'
        assert not any(path.name in ['novel-tts', 'novel-tts.exe'] for path in files)
        assert not any(path.suffix in ['.onnx', '.gguf', '.safetensors'] for path in files)
        assert (installed / 'third-party-licenses' / 'dependencies.json').is_file()
        assert (installed / 'docs/legal/onnxruntime/ThirdPartyNotices.txt').is_file()
        home = base / 'home'
        home.mkdir()
        env = dict(os.environ, HOME=str(home), USERPROFILE=str(home))
        for key in ['LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'CUDA_PATH', 'CUDA_HOME']:
            env.pop(key, None)
        for program in programs:
            binary = str(installed / (program + suffix))
            installed_version = run(binary, '--version', env=env, cwd=home,
                                    capture_output=True, text=True, encoding='utf-8').stdout.strip()
            assert installed_version == f'talechime {version}', installed_version
            run(binary, '--help', env=env, cwd=home, capture_output=True,
                text=True, encoding='utf-8')
            catalog = json.loads(run(binary, 'models', '--json', env=env, cwd=home,
                                     capture_output=True, text=True, encoding='utf-8').stdout)
            kokoro = {info['capabilities']['model'] for info in catalog
                      if info['capabilities']['backend'] == 'kokoro'}
            assert kokoro == {'1.1-zh', '1.1-zh-g2pw'}, 'Standard archive lacks Kokoro models'
            assert (installed / 'third-party-licenses/crates/kokoro/licenses/PYTORCH-LICENSE').is_file()
            wire_version = protocol_version()
            requests = ''.join(json.dumps(dict(protocol_version=wire_version, request_id=kind,
                session_id=None, type=kind)) + '\n' for kind in ['hello', 'shutdown'])
            result = run(binary, '--protocol', input=requests, capture_output=True,
                text=True, encoding='utf-8', env=env, cwd=home)
            messages = [json.loads(line) for line in result.stdout.splitlines()]
            assert messages[0]['type'] == 'ready' and messages[0]['protocol_version'] == wire_version
            assert messages[-1]['type'] == 'accepted'
        assert not list(home.rglob('*.onnx')), 'Handshake downloaded weights'
        assert not list(home.rglob('config.json')), 'Handshake persisted configuration'
        # Exercise the generated platform installer against local build artifacts.
        # Its supported download override keeps this smoke independent of releases.
        server = ThreadingHTTPServer(('127.0.0.1', 0), partial(
            SimpleHTTPRequestHandler, directory=str(Path('target/distrib').resolve())))
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        prefix = app.upper()
        installer_env = dict(env)
        installer_env.update({
            f'{prefix}_DOWNLOAD_URL': f'http://127.0.0.1:{server.server_port}',
            f'{prefix}_UNMANAGED_INSTALL': str(base / 'installer-bin'),
            'XDG_CONFIG_HOME': str(home / '.config'),
            'LOCALAPPDATA': str(home / 'AppData/Local'),
        })
        try:
            if sys.platform == 'win32':
                run('pwsh', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                    str(Path(f'target/distrib/{app}-installer.ps1').resolve()), env=installer_env)
            else:
                run('sh', str(Path(f'target/distrib/{app}-installer.sh').resolve()), env=installer_env)
            for program in programs:
                run(str(base / 'installer-bin' / (program + suffix)), '--version', env=env, cwd=home)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    print(f'{app}: archive and platform installer passed for {target}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('app', choices=['talechime'])
    parser.add_argument('target')
    parser.add_argument('--skip-build', action='store_true',
                        help='Verify downloaded release artifacts without rebuilding')
    parser.add_argument('--build-installers', action='store_true',
                        help='Generate installers for downloaded local artifacts')
    arguments = parser.parse_args()
    main(arguments.app, arguments.target, arguments.skip_build, arguments.build_installers)
