"""Build the unchanged Python kernels as extensions and record exact inputs."""

import json
import platform
from hashlib import sha256
from pathlib import Path

import Cython
from Cython.Build import cythonize
from setuptools import Extension, setup

ROOT = Path.cwd()
MODULES = ("v4", "path_optimizer")
DIRECTIVES = {"language_level": 3, "annotation_typing": False}
COMPILE_ARGS = ["-O3", "-g0"]
sources = {name + ".py": sha256((ROOT / "solver/push_solver" / (name + ".py")).read_bytes()).hexdigest()
           for name in MODULES}
setup(
    name="tictacgo-native-solver", packages=[],
    script_args=["build_ext", "--inplace"],
    ext_modules=cythonize(
        [Extension("solver.push_solver." + name,
                   ["solver/push_solver/" + name + ".py"],
                   extra_compile_args=COMPILE_ARGS) for name in MODULES],
        compiler_directives=DIRECTIVES,
    ),
)
extensions = {path.name: sha256(path.read_bytes()).hexdigest()
              for path in sorted((ROOT / "solver/push_solver").glob("*.so"))}
if len(extensions) != len(MODULES):
    raise RuntimeError("Expected exactly two native solver extensions")
manifest = {
    "cython": Cython.__version__, "python": platform.python_version(),
    "architecture": platform.machine(), "directives": DIRECTIVES,
    "compile_args": COMPILE_ARGS,
    "builder_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    "sources": sources, "extensions": extensions,
}
(ROOT / "solver/push_solver/native-build.json").write_text(json.dumps(manifest, indent=2) + "\n")
