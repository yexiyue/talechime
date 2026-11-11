"""Official evidence cannot outlive its source fixtures or passing check receipt."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import annotate_official_matrix as official


class OfficialEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.addCleanup(patch.stopall)
        patch.object(official.matrix, 'ROOT', self.root).start()
        self.case = {'id':'mixed', 'text':'你好 hello。'}
        self.entry = dict(source_id='mixed', source_text=self.case['text'],
                          source_sha256=hashlib.sha256(self.case['text'].encode()).hexdigest())
        self.fixture = self.root / 'fixture.json'
        self.fixture.write_text(json.dumps({'cases':[self.entry]}))
        self.log = self.root / 'check.log'
        self.log.write_text('check finished')
        self.check = dict(id='independent', exit_code=0, variants=['fixture'],
                          kind='conditioning_checked_synthetic', scope='synthetic inputs only',
                          log=str(self.log), log_sha256=official.matrix.digest(self.log),
                          fixtures=[dict(path='fixture.json', sha256=official.matrix.digest(self.fixture), coverage=True)])

    def evidence(self):
        return official.evidence_index({'checks':[self.check]}, [self.case])

    def test_changed_fixture_or_nonpassing_check_is_not_evidence(self):
        self.check['exit_code'] = 1
        with self.assertRaisesRegex(ValueError, 'did not pass'):
            self.evidence()
        self.check['exit_code'] = 0
        self.fixture.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'fixture changed'):
            self.evidence()

    def test_changed_log_is_not_evidence(self):
        self.log.write_text('a different execution')
        with self.assertRaisesRegex(ValueError, 'log changed'):
            self.evidence()

    def test_fixture_cannot_assign_another_source_hash(self):
        self.entry['source_sha256'] = '0' * 64
        self.fixture.write_text(json.dumps({'cases':[self.entry]}))
        self.check['fixtures'][0]['sha256'] = official.matrix.digest(self.fixture)
        with self.assertRaisesRegex(ValueError, 'source hash'):
            self.evidence()

    def test_official_difference_is_never_promoted_to_match(self):
        self.entry.update(classification='difference')
        self.fixture.write_text(json.dumps({'cases':[self.entry]}))
        self.check['fixtures'][0]['sha256'] = official.matrix.digest(self.fixture)
        self.assertEqual(self.evidence()['fixture/mixed']['status'], 'difference')

    def test_partial_fixture_does_not_invent_case_coverage(self):
        other = dict(id='other', text='另一个输入。')
        result = official.evidence_index({'checks':[self.check]}, [self.case, other])
        self.assertEqual(set(result), {'fixture/mixed'})


if __name__ == '__main__':unittest.main()
