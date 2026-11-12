"""Regenerate pinned cargo-dist CI with a fail-closed publication gate."""
import argparse
from pathlib import Path
import re
import subprocess
import tomllib


def strict_host_gate(workflow, required_jobs):
    jobs = re.findall(r'^  ([\w-]+):$', workflow, re.MULTILINE)
    if len(required_jobs) != len(set(required_jobs)) or 'plan' not in required_jobs:
        raise ValueError('Host prerequisites must be unique and include plan')
    if not set(required_jobs) <= set(jobs):
        raise ValueError('Host prerequisites refer to missing jobs')
    hosts = list(re.finditer(r'^  host:\n.*?(?=^  [\w-]+:|\Z)', workflow,
                             re.DOTALL | re.MULTILINE))
    if len(hosts) != 1:
        raise ValueError('Expected exactly one cargo-dist host job')
    host = hosts[0]
    environment = re.search(r'^    env:\n', host[0], re.MULTILINE)
    if environment is None:
        raise ValueError('Pinned cargo-dist host template changed; review the gate')
    needs = ''.join(f'      - {job}\n' for job in required_jobs)
    checks = ' && '.join(f"needs.{job}.result == 'success'" for job in required_jobs)
    gate = (f'  host:\n    needs:\n{needs}'
            '    # Every validation and artifact job must succeed before publishing.\n'
            "    if: ${{ always() && needs.plan.outputs.publishing == 'true' && "
            + checks + ' }}\n')
    return workflow[:host.start()] + gate + host[0][environment.start():] + workflow[host.end():]


def generate(root, dist, check):
    config_path = root / 'dist-workspace.toml'
    workflow_path = root / '.github/workflows/talechime-v-release.yml'
    config_bytes = config_path.read_bytes()
    previous = workflow_path.read_bytes()
    config = tomllib.loads(config_bytes.decode('utf-8'))
    expected_version = config['dist']['cargo-dist-version']
    version = subprocess.check_output([dist, '--version'], text=True).strip().split()[-1]
    if version != expected_version:
        raise ValueError(f'Expected cargo-dist {expected_version}, found {version}')
    # dist intentionally refuses --mode ci with allow-dirty = ["ci"]. Remove
    # only that setting during generation, restoring both inputs on failure.
    marker = b'allow-dirty = ["ci"]\n'
    if config_bytes.count(marker) != 1:
        raise ValueError('Expected the explicit CI customization setting')
    try:
        config_path.write_bytes(config_bytes.replace(marker, b''))
        subprocess.run([dist, 'generate', '--mode', 'ci'], cwd=root, check=True)
        generated = strict_host_gate(
            workflow_path.read_text(encoding='utf-8'),
            config['talechime-release']['required-host-jobs'],
        ).encode('utf-8')
    finally:
        config_path.write_bytes(config_bytes)
        workflow_path.write_bytes(previous)
    if check:
        if generated != previous:
            raise ValueError('Release workflow is stale; run generate-release.py')
    else:
        workflow_path.write_bytes(generated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--dist', default='dist')
    args = parser.parse_args()
    generate(Path(__file__).resolve().parents[2], args.dist, args.check)
