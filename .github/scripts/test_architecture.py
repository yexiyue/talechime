"""Architecture boundaries for headless consumers and independent model libraries."""
from pathlib import Path
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[2]


def manifest(name):
    return tomllib.loads((ROOT / 'crates' / name / 'Cargo.toml').read_text())


class ArchitectureTests(unittest.TestCase):
    def test_core_has_no_audio_output_or_http_dependency(self):
        core = manifest('talechime-core')
        self.assertTrue({'rodio', 'cpal', 'reqwest', 'talechime-backends', 'talechime'}.isdisjoint(core['dependencies']))
        protocol = manifest('talechime-protocol')
        self.assertTrue({'rodio', 'cpal', 'reqwest', 'talechime-core', 'talechime-backends', 'talechime'}.isdisjoint(protocol['dependencies']))

    def test_moss_is_candle_only(self):
        retired = {'directml-probe', 'ort-dynamic', 'ort-cuda', 'ort-coreml'}
        for name in ['talechime-backends', 'talechime']:
            features = manifest(name)['features']
            self.assertTrue(retired.isdisjoint(features))
        backends = manifest('talechime-backends')
        self.assertNotIn('dep:ort', backends['features']['moss'])
        self.assertIn('dep:talechime-moss-tts', backends['features']['moss'])
        self.assertIn('dep:ort', backends['features']['asr'])
        self.assertIn('talechime-cosyvoice3/ort', backends['features']['cosyvoice3'])

    def test_default_audio_requires_playback_feature(self):
        app = manifest('talechime')
        self.assertTrue(app['dependencies']['rodio']['optional'])
        self.assertIn('dep:rodio', app['features']['playback'])
        self.assertIn('playback', app['features']['cli'])

    def test_model_runtime_dependencies_do_not_point_to_session_or_application(self):
        for name in ['moss-tts', 'qwen3-tts', 'qwen3-asr', 'voxcpm', 'omnivoice', 'cosyvoice3', 'indextts']:
            with self.subTest(model=name):
                dependencies = manifest(name).get('dependencies', {})
                self.assertTrue({'talechime-core', 'talechime-backends', 'talechime', 'talechime-protocol'}.isdisjoint(dependencies))
