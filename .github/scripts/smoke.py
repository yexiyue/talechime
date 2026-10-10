"""Verify the independent CLI and JSON Lines compatibility without model preparation."""
from pathlib import Path
import json
import re
import subprocess
import sys
import tempfile


def protocol_version() -> int:
    root = Path(__file__).resolve().parents[2]
    source = root / 'crates/talechime-protocol/src/lib.rs'
    return int(re.search(r'PROTOCOL_VERSION: u32 = (\d+)', source.read_text()).group(1))


def smoke(directory: Path, suffix: str = '') -> None:
    version = protocol_version()
    requests = ''.join(json.dumps(dict(protocol_version=version, request_id=kind,
                                      session_id=None, type=kind)) + '\n'
                       for kind in ['hello', 'shutdown'])
    with tempfile.TemporaryDirectory() as temporary:
        state = Path(temporary)
        for name in ['talechime']:
            program = directory.resolve() / (name + suffix)
            subprocess.run([str(program), '--version'], check=True)
            subprocess.run([str(program), '--help'], check=True, capture_output=True)
            result = subprocess.run([str(program), '--protocol',
                                     '--config', str(state / 'config.json'),
                                     '--model-dir', str(state / 'models'),
                                     '--checkpoint-dir', str(state / 'positions')],
                                    input=requests, capture_output=True, text=True, encoding='utf-8',
                                    check=True, timeout=15)
            messages = [json.loads(line) for line in result.stdout.splitlines()]
            assert messages[0]['type'] == 'ready' and messages[0]['protocol_version'] == version
            assert messages[-1]['type'] == 'accepted'
        assert not (state / 'config.json').exists(), 'Handshake persisted user configuration'
        assert not any(p.is_file() for p in (state / 'models').rglob('*')), 'Handshake downloaded models'
    print('Talechime passed protocol handshake without model preparation.')


if __name__ == '__main__':
    smoke(Path(sys.argv[1]), '.exe' if sys.platform == 'win32' else '')
