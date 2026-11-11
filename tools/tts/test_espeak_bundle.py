"""Offline bundle integrity gates; never build or execute eSpeak."""
import json
import os
from pathlib import Path
import platform
import tempfile
import unittest
from unittest.mock import patch

import prepare_espeak_bundle as bundle


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "bundle"
        self.output.mkdir()
        names = ["espeak-ng.exe" if os.name == "nt" else "espeak-ng", bundle.WRAPPER, "espeak-ng-data/en_dict",
                 "espeak-ng-data/intonations", "espeak-ng-data/lang/gmw/en", "espeak-ng-data/lang/gmw/en-US",
                 "espeak-ng-data/phondata", "espeak-ng-data/phonindex", "espeak-ng-data/phontab",
                 "source/espeak-ng-1.52.0.tar.gz", "source/COPYING", "source/kokoro_espeak_phonemize.c"]
        entries = []
        for name in names:
            path = self.output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"offline fixture")
            entries.append({"path": name, "size": path.stat().st_size, "sha256": bundle.digest(path)})
        source_hash = bundle.digest(self.output / "source/espeak-ng-1.52.0.tar.gz")
        self.source_pin = patch.object(bundle, "SOURCE_SHA256", source_hash)
        self.source_pin.start()
        self.addCleanup(self.source_pin.stop)
        self.manifest = {"schema": "talechime-espeak-bundle-v1", "source_commit": bundle.COMMIT,
                         "source_sha256": bundle.SOURCE_SHA256, "platform": platform.system(),
                         "machine": platform.machine(), "phonemizer_protocol": bundle.PROTOCOL, "files": entries}
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        (self.output / "manifest.json").write_text(json.dumps(self.manifest))

    def test_missing_changed_and_cross_platform_resources_fail(self):
        bundle.verify_bundle(self.output)
        program = self.output / self.manifest["files"][0]["path"]
        program.write_bytes(b"changed fixture")
        with self.assertRaisesRegex(ValueError, "checksum"):
            bundle.verify_bundle(self.output)
        program.unlink()
        with self.assertRaises(OSError):
            bundle.verify_bundle(self.output)
        self.manifest["platform"] = "another native platform"
        self.save()
        with self.assertRaisesRegex(ValueError, "platform"):
            bundle.verify_bundle(self.output)

    def test_source_mismatch_and_path_escape_fail_before_build(self):
        source = self.root / "source.tar.gz"
        source.write_bytes(b"not the fixed upstream source")
        work = self.root / "work"
        with self.assertRaisesRegex(ValueError, "source checksum"):
            bundle.prepare(source, self.root / "new-bundle", work)
        self.assertFalse(work.exists())
        self.manifest["files"][0]["path"] = "../outside"
        self.save()
        with self.assertRaisesRegex(ValueError, "entry"):
            bundle.verify_bundle(self.output)

    def test_manifest_cannot_substitute_other_corresponding_source(self):
        source = self.output / "source/espeak-ng-1.52.0.tar.gz"
        source.write_bytes(b"different source")
        entry = next(e for e in self.manifest["files"] if e["path"] == "source/espeak-ng-1.52.0.tar.gz")
        entry.update(size=source.stat().st_size, sha256=bundle.digest(source))
        self.save()
        with self.assertRaisesRegex(ValueError, "bundled source checksum"):
            bundle.verify_bundle(self.output)

    def test_runtime_rejects_unlisted_missing_and_duplicate_files(self):
        extra = self.output / "espeak-ng-data/extra"
        extra.write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "unlisted"):
            bundle.verify_bundle(self.output)
        extra.unlink()
        self.manifest["files"] = [entry for entry in self.manifest["files"]
                                  if entry["path"] != "espeak-ng-data/phontab"]
        self.save()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            bundle.verify_bundle(self.output)
        self.manifest["files"].append(self.manifest["files"][0])
        self.save()
        with self.assertRaisesRegex(ValueError, "invalid bundle entry"):
            bundle.verify_bundle(self.output)

    def test_protocol_and_wrapper_are_required(self):
        self.manifest.pop("phonemizer_protocol")
        self.save()
        with self.assertRaisesRegex(ValueError, "identity"):
            bundle.verify_bundle(self.output)
        self.manifest["phonemizer_protocol"] = bundle.PROTOCOL
        self.manifest["files"] = [entry for entry in self.manifest["files"]
                                  if entry["path"] != bundle.WRAPPER]
        self.save()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            bundle.verify_bundle(self.output)

    @unittest.skipIf(os.name == "nt", "symlink creation needs Windows privileges")
    def test_symlinked_directory_is_rejected(self):
        target = self.output / "espeak-ng-data"
        moved = self.root / "data"
        target.rename(moved)
        target.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            bundle.verify_bundle(self.output)


if __name__ == "__main__":
    unittest.main()
