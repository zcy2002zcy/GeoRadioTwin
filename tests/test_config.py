from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.config import load_config


class ConfigTest(unittest.TestCase):
    def test_environment_placeholders_are_expanded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"root": "${GEORADIOTWIN_TEST_ROOT}"}), encoding="utf-8")
            with patch.dict(os.environ, {"GEORADIOTWIN_TEST_ROOT": "portable-root"}):
                config, _ = load_config(path)
            self.assertEqual(config["root"], "portable-root")

    def test_environment_placeholder_default_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps({"root": "${GEORADIOTWIN_TEST_DEFAULT:-fallback-root}"}),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                config, _ = load_config(path)
            self.assertEqual(config["root"], "fallback-root")

    def test_missing_environment_variable_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"root": "${GEORADIOTWIN_MISSING_ROOT}"}), encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(EnvironmentError):
                    load_config(path)


if __name__ == "__main__":
    unittest.main()
