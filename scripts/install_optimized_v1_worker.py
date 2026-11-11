"""Install the hourly, local optimized-v1 publisher as a macOS LaunchAgent.

Uses an isolated checkout pinned to the current committed revision. No production
secrets, database credentials, or hosted compute are used. Run after pushing the
tested feature branch. Unload with: launchctl bootout gui/$(id -u)/com.tictacgo.optimized-v1
"""

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL = "com.tictacgo.optimized-v1"


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("LaunchAgent installation requires macOS")
    state = args.state_dir.resolve()
    worker = state / "worker"
    revision = run("git", "-C", str(ROOT), "rev-parse", "HEAD").stdout.strip()
    branch = run("git", "-C", str(ROOT), "branch", "--show-current").stdout.strip()
    remote = run("git", "-C", str(ROOT), "remote", "get-url", "origin").stdout.strip()
    if worker.exists():
        parser.error(
            f"Worker already exists at {worker}; inspect it before reinstalling"
        )
    state.mkdir(parents=True, exist_ok=True)
    run(
        "git",
        "clone",
        "--depth",
        "1",
        "--filter=blob:none",
        "--sparse",
        "--branch",
        branch,
        remote,
        str(worker),
    )
    run(
        "git",
        "-C",
        str(worker),
        "sparse-checkout",
        "set",
        "solver",
        "apps/api",
        "scripts",
    )
    run("git", "-C", str(worker), "checkout", "--detach", revision)
    for key in ("user.name", "user.email"):
        value = run("git", "-C", str(ROOT), "config", key).stdout.strip()
        run("git", "-C", str(worker), "config", key, value)
    # Keep the worker's interpreter separate from future workspace venv edits.
    run(sys.executable, "-m", "venv", str(state / "runtime"))
    python = state / "runtime" / "bin" / "python"
    # This publishing/search path needs only the Python standard library.
    environment = {**os.environ, "PYTHONPATH": str(worker)}
    run(
        str(python),
        "-c",
        "from scripts.update_optimized_v1 import load_entries; from solver.optimized_v1 import optimize",
        cwd=worker,
        env=environment,
    )
    feed = state / "optimized-v1.json"
    if not feed.exists():
        shutil.copyfile(ROOT / "apps/web/data/optimized-v1.json", feed)
    logs = state / "logs"
    logs.mkdir(exist_ok=True)
    plist = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    plist.parent.mkdir(parents=True, exist_ok=True)
    if plist.exists():
        parser.error(f"LaunchAgent already exists: {plist}")
    payload = {
        "Label": LABEL,
        "ProgramArguments": [
            str(python),
            str(worker / "scripts/update_optimized_v1.py"),
            "--state-dir",
            str(state),
            "--publish",
        ],
        "WorkingDirectory": str(worker),
        "EnvironmentVariables": {
            "PYTHONPATH": str(worker),
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "GIT_TERMINAL_PROMPT": "0",
        },
        "StartInterval": 3600,
        "RunAtLoad": True,
        "ProcessType": "Background",
        "Nice": 10,
        "StandardOutPath": str(logs / "worker.log"),
        "StandardErrorPath": str(logs / "worker-error.log"),
    }
    plist.write_bytes(plistlib.dumps(payload))
    run("launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist))
    print(
        f"Installed {LABEL} at revision {revision}; hourly checks and catch-up on load."
    )
    print(f"Status: launchctl print gui/{os.getuid()}/{LABEL}")
    print(f"Logs: {logs}")


if __name__ == "__main__":
    main()
