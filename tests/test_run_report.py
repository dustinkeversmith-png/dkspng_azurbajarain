"""Tournament reports land in a visible folder."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orchestrators.run_report import append_attempt, publish_report


class RunReportTest(unittest.TestCase):
    def test_each_publish_writes_latest_and_a_copy(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            log = append_attempt({"source": "test", "content": "{\"c1\":{}}"}, directory)
            first = publish_report("verify one", "slice layer1", directory)
            second = publish_report("verify two", "slice head", directory)
            self.assertTrue(log.is_file())
            self.assertEqual(json.loads(log.read_text(encoding="utf-8"))["content"], "{\"c1\":{}}")
            latest = directory / "latest_report.md"
            self.assertIn("slice head", latest.read_text(encoding="utf-8"))
            self.assertIn("slice layer1", first.read_text(encoding="utf-8"))
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())


if __name__ == "__main__":
    unittest.main()
