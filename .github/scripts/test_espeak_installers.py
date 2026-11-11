"""Failure-closed cargo-dist installer patching and checksum accounting."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('installers', Path(__file__).with_name('espeak-installers.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class InstallerPatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        names = {'talechime-installer.sh': module.SHELL_ANCHOR,
                 'talechime-installer.ps1': module.POWERSHELL_ANCHOR,
                 'talechime.rb': '  license "MIT"\n' + module.BREW_ANCHOR}
        self.artifacts = {}
        for name, text in names.items():
            path = self.root / name
            path.write_text('prefix\n' + text + '\nsuffix\n')
            self.artifacts[name] = {'path': str(path), 'kind': 'installer'}
        path = self.root / 'sha256.sum'
        path.write_text('a' * 64 + ' *source.tar.gz\n\n')
        self.artifacts[path.name] = {'path': str(path), 'kind': 'unified-checksum'}
        self.manifest = self.root / 'manifest.json'
        self.write_manifest()

    def tearDown(self):
        self.tmp.cleanup()

    def write_manifest(self):
        self.manifest.write_text(json.dumps({'dist_version': '0.32.0', 'artifacts': self.artifacts,
            'upload_files': [entry['path'] for entry in self.artifacts.values()]}))

    def test_each_installer_and_manifest_has_the_current_hash(self):
        module.patch_installers(self.manifest)
        result = json.loads(self.manifest.read_text())
        lines = (self.root / 'sha256.sum').read_text().splitlines()
        for name in ('talechime-installer.sh', 'talechime-installer.ps1', 'talechime.rb'):
            data = (self.root / name).read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            self.assertEqual(result['artifacts'][name]['checksums']['sha256'], digest)
            self.assertIn(f'{digest} *{name}', lines)
            self.assertIn(b'espeak-ng-1.52.0', data)
            self.assertIn(b'COPYING', data)
        self.assertIn('GPL-3.0-or-later', (self.root / 'talechime.rb').read_text())
        self.assertIn('a' * 64 + ' *source.tar.gz', lines)
        with self.assertRaises(ValueError):
            module.patch_installers(self.manifest)

    def test_changed_template_and_missing_upload_fail_before_any_writes(self):
        target = self.root / 'talechime-installer.ps1'
        target.write_text('unexpected template')
        before = {path: path.read_bytes() for path in self.root.iterdir()}
        with self.assertRaises(ValueError):
            module.patch_installers(self.manifest)
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.iterdir()})
        target.write_text(module.POWERSHELL_ANCHOR)
        data = json.loads(self.manifest.read_text())
        data['upload_files'].remove(str(target))
        self.manifest.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            module.patch_installers(self.manifest)

    def test_existing_sha256_sidecar_is_updated(self):
        name = 'talechime-installer.sh'
        sidecar = self.root / (name + '.sha256')
        sidecar.write_text('old')
        self.artifacts[name]['checksum'] = sidecar.name
        self.artifacts[sidecar.name] = {'path': str(sidecar), 'kind': 'checksum'}
        self.write_manifest()
        module.patch_installers(self.manifest)
        digest = hashlib.sha256((self.root / name).read_bytes()).hexdigest()
        self.assertEqual(sidecar.read_text(), f'{digest} *{name}\n')

    def test_version_and_checksum_format_must_match(self):
        data = json.loads(self.manifest.read_text())
        data['dist_version'] = '0.33.0'
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, '0.32.0'):
            module.patch_installers(self.manifest)
        self.write_manifest()
        (self.root / 'sha256.sum').write_text('not a checksum\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            module.patch_installers(self.manifest)

    def test_global_hook_is_required_and_fails_closed_on_template_change(self):
        source = Path(__file__).with_name('generate-release.py')
        generator_spec = importlib.util.spec_from_file_location('release_generator', source)
        generator = importlib.util.module_from_spec(generator_spec)
        generator_spec.loader.exec_module(generator)
        root = Path(__file__).resolve().parents[2]
        workflow = (root / '.github/workflows/talechime-v-release.yml').read_text()
        self.assertEqual(generator.installer_hook(workflow), workflow)
        with self.assertRaises(ValueError):
            generator.installer_hook(workflow.replace('--artifacts=global', '--artifacts=changed'))
        with self.assertRaises(ValueError):
            generator.installer_hook(workflow + '\n          python3 .github/scripts/espeak-installers.py dist-manifest.json\n')

    def test_smoke_patches_the_manifest_from_its_own_global_build(self):
        scripts = Path(__file__).parent
        sys.path.insert(0, str(scripts))
        self.addCleanup(sys.path.remove, str(scripts))
        smoke_spec = importlib.util.spec_from_file_location('distribution_smoke', scripts / 'distribution-smoke.py')
        smoke = importlib.util.module_from_spec(smoke_spec)
        smoke_spec.loader.exec_module(smoke)
        calls = []
        def fake_run(*args, **kwargs):
            calls.append(args)
            if args[0] == 'dist':
                kwargs['stdout'].write('{"generation":"current"}')
            else:
                self.assertEqual(json.loads(Path(args[2]).read_text()), {'generation': 'current'})
        with patch.object(smoke, 'run', side_effect=fake_run):
            smoke.build_global_installers('--tag', 'test-tag')
        self.assertEqual(calls[0], ('dist', 'build', '--artifacts', 'global', '--tag', 'test-tag', '--output-format=json'))
        self.assertEqual(calls[1][1], '.github/scripts/espeak-installers.py')


if __name__ == '__main__':
    unittest.main()
