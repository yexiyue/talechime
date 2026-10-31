"""Offline checks for the resumable review batch; no model or network access."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave
import zipfile
import voice_candidates as batch


def wav_bytes():
    out = io.BytesIO()
    with wave.open(out, 'wb') as wav:
        wav.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
        wav.writeframes(struct.pack('<h', 4000) * 8000)
    return out.getvalue()


class BatchTests(unittest.TestCase):
    def test_matrix_is_bounded_unique_and_stable(self):
        rows = batch.matrix('test')
        self.assertEqual(len(rows), 48)
        self.assertEqual(rows, batch.matrix('test'))
        self.assertEqual(len({r['id'] for r in rows}), 48)
        self.assertLessEqual(len(batch.REFERENCE), 60)
        self.assertTrue(80 <= len(batch.PREVIEW) <= 120)

    def test_batch_cannot_escape_output_or_inject_spreadsheet_formulas(self):
        for name in ['../other', 'a/b', '=IMPORTXML', '中文', 'x' * 33]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                batch.matrix(name)

    def test_wav_rejects_truncation_and_nonfinite(self):
        data = wav_bytes()
        self.assertEqual(batch.wav_info(data)['seconds'], 1)
        with self.assertRaises(ValueError):
            batch.wav_info(data[:-1])
        floats = b'RIFF' + struct.pack('<I', 40) + b'WAVEfmt ' + struct.pack('<IHHIIHH', 16, 3, 1, 8000, 32000, 4, 32) + b'data' + struct.pack('<If', 4, float('nan'))
        with self.assertRaises(ValueError):
            batch.wav_info(floats)

    def test_extensible_float_wav_from_rust_exporter(self):
        fmt = struct.pack('<HHIIHHHHI', 65534, 1, 8000, 32000, 4, 32, 22, 32, 1)
        fmt += struct.pack('<I', 3) + bytes.fromhex('00001000800000aa00389b71')
        audio = struct.pack('<f', .5) * 8000
        body = b'WAVEfmt ' + struct.pack('<I', len(fmt)) + fmt + b'data' + struct.pack('<I', len(audio)) + audio
        self.assertEqual(batch.wav_info(b'RIFF' + struct.pack('<I', len(body)) + body)['seconds'], 1)

    def test_asset_conflict_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            row = batch.matrix('test')[0]
            folder = Path(temp) / row['id']
            folder.mkdir()
            (folder / 'voice.json').write_text(json.dumps({'id': 'someone-else', 'transcript': ''}))
            (folder / 'reference.wav').write_bytes(wav_bytes())
            with self.assertRaises((KeyError, ValueError)):
                batch.asset(Path(temp), row)

    def test_resume_after_export_failure_does_not_redesign(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = root / 'binary'
            binary.write_bytes(b'fake frozen build')
            config = root / 'config.json'
            library = root / 'voices'
            args = argparse.Namespace(output=root / 'review', batch='test', resources=root / 'resources',
                binary=binary, preview=binary, config=config, hours=1, moss_verified=False, retry_failed=False)
            rows = batch.matrix('test')[:1]
            calls = []
            def fake_run(command, log, timeout):
                calls.append(command)
                if 'design' in command:
                    def value(flag):
                        return command[command.index(flag) + 1]
                    folder = library / value('--id')
                    folder.mkdir(parents=True)
                    wav = wav_bytes()
                    desc, text = value('--description'), value('--text')
                    record = dict(id=value('--id'), transcript=text, description=desc,
                        source={'kind': 'designed', 'engine': value('--engine')},
                        fingerprint=batch.digest(text.encode()+b'\0'+desc.encode()+b'\0'+wav))
                    (folder / 'voice.json').write_text(json.dumps(record))
                    (folder / 'reference.wav').write_bytes(wav)
                elif 'export' in command:
                    if sum('export' in c for c in calls) == 1:
                        raise RuntimeError('controlled export failure')
                    with zipfile.ZipFile(command[-1], 'w') as archive:
                        archive.writestr('reference.wav', wav_bytes())
                else:
                    Path(command[-1]).write_bytes(wav_bytes())
                return .1
            with patch.object(batch, 'matrix', return_value=rows), patch.object(batch, 'run', side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
                batch.execute(args)
                self.assertEqual(json.loads((args.output/'manifest.json').read_text())['rows'][0]['design'], 'succeeded')
                args.retry_failed = True
                batch.execute(args)
                batch.execute(args)
            self.assertEqual(sum('design' in c for c in calls), 1)
            self.assertEqual(sum('export' in c for c in calls), 2)
            self.assertTrue((args.output / 'index.html').is_file())
            self.assertTrue((args.output / rows[0]['id'] / 'preview.wav').is_file())

    def test_timeout_waits_for_process_exit(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(subprocess.TimeoutExpired):
                batch.run([sys.executable, '-c', 'import time; time.sleep(30)'], Path(temp)/'log', .05)


if __name__ == '__main__':
    unittest.main()
