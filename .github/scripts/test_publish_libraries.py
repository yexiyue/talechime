"""Release identity checks against real temporary Git history; no publishing."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "publish_libraries", Path(__file__).with_name("publish-libraries.py")
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class PublishedSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Release Test")
        self.git("config", "user.email", "release@example.invalid")
        (self.root / "crates/example").mkdir(parents=True)
        (self.root / "crates/example/Cargo.toml").write_text(
            '[package]\nname = "example"\nversion = "1.0.0"\n'
        )
        (self.root / "crates/example/lib.rs").write_text("pub fn example() {}\n")
        (self.root / "Cargo.toml").write_text('[workspace]\nmembers = ["crates/example"]\n')
        self.original = self.commit()

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, text=True).strip()

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-qm", "test source")
        return self.git("rev-parse", "HEAD")

    def archive(self, commit=None, path="crates/example", dirty=False):
        data = json.dumps({"git": {"sha1": commit or self.original, "dirty": dirty}, "path_in_vcs": path}).encode()
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            member = tarfile.TarInfo("example-1.0.0/.cargo_vcs_info.json")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        return output.getvalue()

    def verify(self, archive=None):
        return release.verify_published_source(
            "example", self.git("rev-parse", "HEAD"), archive or self.archive(),
            "example", "1.0.0", root=self.root,
        )

    def test_same_commit_is_accepted(self):
        self.assertEqual(self.verify(), "this commit")

    def test_unrelated_package_changes_preserve_existing_version(self):
        (self.root / "other.txt").write_text("another package changed\n")
        self.commit()
        self.assertEqual(self.verify(), f"unchanged ancestor {self.original}")

    def test_package_source_change_requires_new_version(self):
        (self.root / "crates/example/lib.rs").write_text("pub fn changed() {}\n")
        self.commit()
        with self.assertRaisesRegex(RuntimeError, "increment the package version"):
            self.verify()

    def test_workspace_manifest_change_requires_new_version(self):
        with (self.root / "Cargo.toml").open("a") as manifest:
            manifest.write('resolver = "3"\n')
        self.commit()
        with self.assertRaisesRegex(RuntimeError, "increment the package version"):
            self.verify()

    def test_unknown_source_commit_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "not a verified ancestor"):
            self.verify(self.archive("0" * 40))

    def test_other_branch_is_not_release_source(self):
        (self.root / "other.txt").write_text("other branch\n")
        other = self.commit()
        self.git("checkout", "--detach", "-q", self.original)
        (self.root / "different.txt").write_text("release branch\n")
        self.commit()
        with self.assertRaisesRegex(RuntimeError, "not a verified ancestor"):
            self.verify(self.archive(other))

    def test_same_commit_with_wrong_package_path_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "invalid source identity"):
            self.verify(self.archive(path="crates/foreign"))

    def test_dirty_published_source_is_not_proven_by_its_commit(self):
        with self.assertRaisesRegex(RuntimeError, "invalid source identity"):
            self.verify(self.archive(dirty=True))

    def test_yanked_version_never_downloads_or_publishes(self):
        with patch.object(release, "ROOT", self.root), patch.object(release, "PACKAGES", ("example",)), \
                patch.object(release, "get", return_value=b'{"version":{"yanked":true}}') as get, \
                patch.object(release.subprocess, "check_output", return_value=self.original), \
                patch.object(release.subprocess, "run") as publish:
            with self.assertRaisesRegex(RuntimeError, "is yanked"):
                release.main()
        get.assert_called_once()
        publish.assert_not_called()

    def test_uncertain_registry_result_never_publishes(self):
        with patch.object(release, "ROOT", self.root), patch.object(release, "PACKAGES", ("example",)), \
                patch.object(release, "get", side_effect=TimeoutError("unknown registry state")), \
                patch.object(release.subprocess, "check_output", return_value=self.original), \
                patch.object(release.subprocess, "run") as publish:
            with self.assertRaises(TimeoutError):
                release.main()
        publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
