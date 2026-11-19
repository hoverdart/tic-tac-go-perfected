"""Sync the current optimized data branch and refresh both public hostnames."""

import json
import os
import subprocess
from urllib.error import HTTPError
from urllib.request import Request, urlopen

REFRESH_BATCH_SIZE = 50


def request(origin, secret, phase, **payload):
    endpoint = origin + "/api/maintenance/optimized-publication"
    headers = {"Authorization": f"Bearer {secret}", "Content-Type": "application/json"}
    try:
        with urlopen(
            Request(endpoint, data=json.dumps({"phase": phase, **payload}).encode(), headers=headers),
            timeout=35,
        ) as response:
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"{phase} failed at {origin}: HTTP {exc.code}") from None


def main():
    revision, _ = subprocess.check_output(
        ["git", "ls-remote", "--exit-code", "origin", "refs/heads/optimized-v1-data"],
        text=True,
    ).split()
    origins = os.environ["SOLUTION_PUBLICATION_ORIGINS"].split(",")
    secret = os.environ["OPTIMIZED_PUBLICATION_SECRET"]
    dates = request(origins[0], secret, "import", revision=revision)["dates"]
    if dates:
        for origin in origins:
            for start in range(0, len(dates), REFRESH_BATCH_SIZE):
                request(origin, secret, "refresh", dates=dates[start:start + REFRESH_BATCH_SIZE])
        request(origins[0], secret, "ack", revision=revision)
    print(f"Optimized publication {revision}: refreshed {len(dates)} dates.")


if __name__ == "__main__":
    main()
