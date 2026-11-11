"""python -m solver.optimized_paths --input board-and-seeds.json --mode exact"""

import argparse
import json
from pathlib import Path

from . import optimize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--mode", choices=("graph", "repair", "exact"), default="repair"
    )
    parser.add_argument("--seconds", type=float, default=1)
    parser.add_argument("--max-nodes", type=int, default=100_000)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--max-layouts", type=int, default=8192)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output already exists")
    payload = json.loads(args.input.read_text())
    result = optimize(
        payload["board"],
        payload["seeds"],
        mode=args.mode,
        seconds=args.seconds,
        max_nodes=args.max_nodes,
        radius=args.radius,
        max_layouts=args.max_layouts,
    )
    output = json.dumps(result.to_dict(), indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
