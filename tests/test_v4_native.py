import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from solver.push_solver import v4


class NativeRuntimeTest(unittest.TestCase):
    def test_runtime_reports_and_checks_its_actual_implementation(self):
        runtime = v4.runtime_environment()
        compiled = Path(v4.__file__).suffix == ".so"
        self.assertEqual(runtime["implementation"], "cython" if compiled else "python")
        if compiled:
            manifest = runtime["native_build"]
            self.assertEqual(manifest["cython"], "3.3.1")
            self.assertEqual(set(manifest["sources"]), {"v4.py", "path_optimizer.py"})
            self.assertEqual(len(manifest["extensions"]), 2)
        else:
            self.assertIsNone(runtime["native_build"])

    def test_native_build_rejects_changed_sources_and_missing_extensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("v4.py", "path_optimizer.py", "v4.so", "path_optimizer.so")
            for name in names:
                (root / name).write_bytes(name.encode())
            manifest = {
                "sources": {name: sha256((root / name).read_bytes()).hexdigest() for name in names[:2]},
                "extensions": {name: sha256((root / name).read_bytes()).hexdigest() for name in names[2:]},
            }
            path = root / "native-build.json"
            path.write_text(json.dumps(manifest))
            try:
                with patch.object(v4, "__file__", str(root / "v4.so")):
                    v4.native_build_metadata.cache_clear()
                    self.assertEqual(v4.native_build_metadata(), manifest)
                    (root / "v4.py").write_text("changed source")
                    v4.native_build_metadata.cache_clear()
                    with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                        v4.native_build_metadata()
                    manifest["extensions"] = {}
                    path.write_text(json.dumps(manifest))
                    v4.native_build_metadata.cache_clear()
                    with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                        v4.native_build_metadata()
            finally:
                v4.native_build_metadata.cache_clear()
