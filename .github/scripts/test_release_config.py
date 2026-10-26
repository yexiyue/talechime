"""Release feature selection must preserve the platform and runtime boundaries."""
import importlib.util
from pathlib import Path
import tempfile
import tomllib
import unittest

spec = importlib.util.spec_from_file_location(
    'configure_release', Path(__file__).with_name('configure-release.py'))
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


class ReleaseConfigTests(unittest.TestCase):
    def test_platform_switches_are_reversible_and_do_not_duplicate_features(self):
        source = Path(__file__).resolve().parents[2] / 'dist-workspace.toml'
        original = tomllib.loads(source.read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / source.name
            path.write_bytes(source.read_bytes())
            for target in original['dist']['targets'] * 2:
                features = selector.configure(root, target)
                self.assertEqual(len(features), len(set(features)))
                self.assertEqual(any(f.endswith('-metal') for f in features),
                                 target.endswith('-apple-darwin'))
                self.assertFalse(any('cuda' in f or f == 'ort-dynamic' for f in features))
                self.assertIn('asr', features)
                self.assertIn('cli', features)
                changed = tomllib.loads(path.read_text(encoding='utf-8'))
                changed['dist']['features'] = original['dist']['features']
                self.assertEqual(changed, original)
            with self.assertRaises(ValueError):
                selector.configure(root, 'unsupported-target')


if __name__ == '__main__':
    unittest.main()
