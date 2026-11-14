"""Installation must exercise the same lazy dependency as a new daily solve."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import install_optimized_v1_worker as installer
from scripts.smoke_optimized_v1_worker import smoke_test


class WorkerRuntimeTests(unittest.TestCase):
    def test_fresh_board_end_to_end(self):
        record, receipt = smoke_test()
        self.assertEqual(len(record["moves"]), 10)
        self.assertEqual(record["board_sha256"], receipt["board_sha256"])

    def test_smoke_catches_original_missing_numpy_regression(self):
        code = """
import builtins
original = builtins.__import__
def without_numpy(name, *args, **kwargs):
    if name == "numpy" or name.startswith("numpy."):
        raise ModuleNotFoundError("No module named 'numpy'")
    return original(name, *args, **kwargs)
builtins.__import__ = without_numpy
from scripts.smoke_optimized_v1_worker import smoke_test
smoke_test()
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=installer.ROOT,
            env={**os.environ, "PYTHONPATH": str(installer.ROOT)},
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No module named 'numpy'", result.stderr)

    def test_dependencies_and_smoke_precede_launchagent(self):
        calls = []

        def fake_run(*args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0, stdout="test\n")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(
                    sys, "argv", ["install", "--state-dir", str(root / "state")]
                ),
                patch.object(sys, "platform", "darwin"),
                patch.object(Path, "home", return_value=root),
                patch.object(installer, "run", side_effect=fake_run),
                patch.object(installer.shutil, "copyfile"),
            ):
                installer.main()
        pip_index = next(i for i, (a, _) in enumerate(calls) if "pip" in a)
        smoke_index = next(
            i
            for i, (a, _) in enumerate(calls)
            if "scripts.smoke_optimized_v1_worker" in a
        )
        launch_index = next(i for i, (a, _) in enumerate(calls) if a[0] == "launchctl")
        self.assertLess(pip_index, smoke_index)
        self.assertLess(smoke_index, launch_index)
        self.assertEqual(calls[smoke_index][1]["timeout"], 90)
        self.assertTrue(
            any(
                str(a).endswith("requirements-optimized-worker.txt")
                for a in calls[pip_index][0]
            )
        )


if __name__ == "__main__":
    unittest.main()
