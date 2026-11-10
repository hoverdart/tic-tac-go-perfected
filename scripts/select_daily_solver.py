"""Select V3/V4 using an immutable, fully qualified Cloud Run image."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATHS = (
    "Dockerfile", ".dockerignore", "pyproject.toml", "apps/api", "solver",
    "scripts/build_solver.py", ".github/workflows/deploy.yml",
    "debug-artifacts/v4-phs-2026-10-26/policy.json",
    "debug-artifacts/v4-phs-2026-10-26/policy-mlp.json",
)


def validate_evidence(directory, revision, *, root=ROOT):
    root = root.resolve()
    directory = (root / directory).resolve()
    if not directory.is_relative_to(root / "debug-artifacts"):
        raise ValueError("Evidence must be inside the repository's debug-artifacts directory")
    identities = []
    for name, expected in (("smoke", 12), ("guards", 18), ("matrix", 180), ("corpus", 397)):
        gate = json.loads((directory / (name + ".gate.json")).read_text())
        metadata = json.loads((directory / (name + ".metadata.json")).read_text())
        rows = [json.loads(line) for line in (directory / (name + ".jsonl")).read_text().splitlines() if line]
        if not gate.get("passed") or len(rows) != expected or metadata["revision"] != revision:
            raise ValueError("Incomplete, failed, or different-revision qualification: " + name)
        if name == "corpus" and gate.get("v4_verified") != 397:
            raise ValueError("The complete corpus must have 397 verified solutions")
        config = metadata["config"]
        runtime = config["provenance"]["runtime"]
        native = runtime.get("native_build")
        if (
            runtime.get("implementation") != "cython" or not native
            or native.get("cython") != "3.3.1" or len(native.get("extensions", {})) != 2
            or metadata.get("routes") != 2
        ):
            raise ValueError("Expected the compiled two-route production configuration")
        sources = config["provenance"]["sources"]
        if not {"solver/push_solver/v4.py", "solver/push_solver/path_optimizer.py", "solver/push_solver/benchmark_v4.py"} <= sources.keys():
            raise ValueError("Missing kernel source provenance")
        for relative, digest in sources.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("Qualification sources differ from this checkout")
        identities.append((metadata["engine_sha256"], metadata["runner_sha256"], native))
        for row in rows:
            result = row["result"]
            if not (
                row.get("independently_verified") and row.get("verified") and result["solved"]
                and row.get("version") == "v4"
                and row["revision"] == revision and row["engine_sha256"] == metadata["engine_sha256"]
                and row["runner_sha256"] == metadata["runner_sha256"]
                and row["limits"] == {"seconds": 30, "nodes": 500000}
                and row["schedule"] == "coverage" and row["routes"] == 2
                and row["quality_timeout_seconds"] == 30 and row["wall_seconds"] <= 30
                and result["nodes_expanded"] <= 500000
                and result["diagnostics"]["exact_hints"] is False
                and result["diagnostics"]["route_limit"] == 2
                and result["diagnostics"]["runtime"]["implementation"] == "cython"
                and result["diagnostics"]["runtime"]["native_build"] == native
                and (row["process"] != "fresh" or 0 < row.get("process_wall_seconds", 31) <= 30)
            ):
                raise ValueError("Invalid qualification receipt: " + name)
    if any(identity != identities[0] for identity in identities):
        raise ValueError("Qualification cohorts use different source or binary identities")
    return identities[0][0]


def gcloud_json(*args):
    return json.loads(subprocess.check_output(["gcloud", *args], text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v3", "v4"), required=True)
    parser.add_argument("--qualified-revision", required=True)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"tic-tac-go-[a-z0-9-]{1,50}", args.qualified_revision):
        raise ValueError("Invalid qualified revision")
    engine = validate_evidence(args.evidence_dir, args.qualified_revision)
    if args.verify_only:
        print("Qualification verified: " + engine)
        return
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    latest = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"], text=True).split()[0]
    if current != latest:
        raise ValueError("Only the current main checkout may select the daily solver")
    data = gcloud_json("run", "revisions", "describe", args.qualified_revision,
                       "--region", "us-west2", "--format=json(status.imageDigest,metadata.labels)")
    image = data["status"]["imageDigest"]
    source = data["metadata"]["labels"]["commit-sha"]
    if not re.fullmatch(r"[a-zA-Z0-9./_-]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("Qualified revision must identify an immutable container digest")
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("Missing qualified image source commit")
    subprocess.run(["git", "diff", "--quiet", source, "HEAD", "--", *RUNTIME_PATHS], check=True)
    subprocess.run([
        "gcloud", "run", "services", "update", "tic-tac-go", "--region", "us-west2",
        "--image", image, "--update-env-vars", "DAILY_PUSH_SOLVER_VERSION=" + args.version, "--quiet",
    ], check=True)
    subprocess.run([
        "gcloud", "run", "services", "update-traffic", "tic-tac-go", "--region", "us-west2",
        "--to-latest", "--quiet",
    ], check=True)
    print(json.dumps({"version": args.version, "qualified_revision": args.qualified_revision,
                      "source_commit": source, "image": image, "engine_sha256": engine}))


if __name__ == "__main__":
    main()
