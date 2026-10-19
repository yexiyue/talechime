"""WAV validation and numeric comparison regressions without model downloads."""
import math
from pathlib import Path
import struct
import tempfile
import unittest

from compare_wav import compare


def write_wav(path, samples):
    data = struct.pack(f"<{len(samples)}f", *samples)
    fmt = struct.pack("<HHIIHH", 3, 1, 48_000, 192_000, 4, 32)
    contents = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    path.write_bytes(b"RIFF" + struct.pack("<I", len(contents)) + contents)


class CompareWavTests(unittest.TestCase):
    def test_identical_invalid_pcm_cannot_bypass_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("a", "b"):
                (root / name).mkdir()
                write_wav(root / name / "001.wav", [math.nan])
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                compare(root / "a", root / "b")

    def test_reports_mean_absolute_error_and_skips_unaligned_lengths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("a", "b"):
                (root / name).mkdir()
            write_wav(root / "a/001.wav", [1.0, 0.0])
            write_wav(root / "b/001.wav", [0.5, 0.0])
            row = compare(root / "a", root / "b")["rows"][0]
            self.assertEqual(row["max_abs"], 0.5)
            self.assertEqual(row["mean_abs"], 0.25)
            self.assertAlmostEqual(row["rms"], math.sqrt(0.125))
            write_wav(root / "b/001.wav", [0.5])
            row = compare(root / "a", root / "b")["rows"][0]
            self.assertNotIn("snr_db", row)
            self.assertEqual(row["samples_b"], 1)


if __name__ == "__main__":
    unittest.main()
