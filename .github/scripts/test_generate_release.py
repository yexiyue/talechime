"""Publication must reject skipped, cancelled, or failed prerequisite jobs."""
import importlib.util
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'generate_release', Path(__file__).with_name('generate-release.py'))
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)
ROOT = Path(__file__).resolve().parents[2]


class ReleaseGateTests(unittest.TestCase):
    def test_every_prerequisite_must_succeed_in_committed_workflow(self):
        config = tomllib.loads((ROOT / 'dist-workspace.toml').read_text(encoding='utf-8'))
        required = config['talechime-release']['required-host-jobs']
        self.assertEqual(set(required), {'plan', 'custom-ci', 'custom-performance-contracts',
                                        'build-local-artifacts', 'build-global-artifacts',
                                        'custom-release-smoke'})
        workflow = (ROOT / '.github/workflows/talechime-v-release.yml').read_text(encoding='utf-8')
        self.assertEqual(generator.strict_host_gate(workflow, required), workflow)
        host = workflow.split('  host:\n', 1)[1].split('    env:\n', 1)[0]
        needs = re.findall(r'^      - ([\w-]+)$', host, re.MULTILINE)
        self.assertEqual(needs, required)
        expression = host.split('    if: ${{ ', 1)[1].strip().removesuffix(' }}')
        clauses = expression.split(' && ')
        self.assertEqual(clauses[:2], ['always()', "needs.plan.outputs.publishing == 'true'"])
        checks = [re.fullmatch(r"needs\.([\w-]+)\.result == 'success'", c) for c in clauses[2:]]
        self.assertTrue(all(checks), expression)
        checked_jobs = [match[1] for match in checks]
        self.assertEqual(checked_jobs, required)
        states = dict.fromkeys(required, 'success')
        self.assertTrue(all(states[job] == 'success' for job in checked_jobs))
        for job in required:
            for status in ('skipped', 'cancelled', 'failure'):
                with self.subTest(job=job, status=status):
                    changed = states | {job: status}
                    self.assertFalse(all(changed[j] == 'success' for j in checked_jobs))

    def test_missing_prerequisite_and_changed_template_fail_closed(self):
        workflow = '  plan:\n  host:\n    needs:\n      - plan\n    env:\n'
        with self.assertRaises(ValueError):
            generator.strict_host_gate(workflow, ['plan', 'missing-job'])
        with self.assertRaises(ValueError):
            generator.strict_host_gate(workflow.replace('    env:\n', ''), ['plan'])
        with self.assertRaises(ValueError):
            generator.strict_host_gate(
                workflow.replace('    env:\n', '  announce:\n    env:\n'), ['plan'])

    def test_check_and_generation_failure_restore_original_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'dist-workspace.toml'
            workflow = root / '.github/workflows/talechime-v-release.yml'
            workflow.parent.mkdir(parents=True)
            config.write_bytes((ROOT / config.name).read_bytes())
            workflow.write_bytes((ROOT / workflow.relative_to(root)).read_bytes())
            originals = config.read_bytes(), workflow.read_bytes()

            def failed_generation(*args, **kwargs):
                workflow.write_text('incomplete output', encoding='utf-8')
                raise subprocess.CalledProcessError(1, ['dist', 'generate'])

            with patch.object(generator.subprocess, 'check_output', return_value='dist 0.32.0'):
                with patch.object(generator.subprocess, 'run'):
                    generator.generate(root, 'dist', check=True)
                self.assertEqual((config.read_bytes(), workflow.read_bytes()), originals)
                with patch.object(generator.subprocess, 'run', side_effect=failed_generation):
                    with self.assertRaises(subprocess.CalledProcessError):
                        generator.generate(root, 'dist', check=False)
                self.assertEqual((config.read_bytes(), workflow.read_bytes()), originals)

                raw = originals[1].decode('utf-8').replace(
                    "needs.custom-ci.result == 'success'", "needs.custom-ci.result == 'skipped'")

                def write_generated(*args, **kwargs):
                    workflow.write_text(raw, encoding='utf-8')

                with patch.object(generator.subprocess, 'run', side_effect=write_generated):
                    workflow.write_bytes(b'stale workflow')
                    with self.assertRaisesRegex(ValueError, 'stale'):
                        generator.generate(root, 'dist', check=True)
                    self.assertEqual(config.read_bytes(), originals[0])
                    self.assertEqual(workflow.read_bytes(), b'stale workflow')
                    generator.generate(root, 'dist', check=False)
                self.assertEqual((config.read_bytes(), workflow.read_bytes()), originals)


if __name__ == '__main__':
    unittest.main()
