"""CLI for exporting learned-search candidate rows."""

from __future__ import annotations

import argparse

from solver.gymnasium_register import ranked_real_boards
from solver.learned_search.training_data import export_expert_rows


BOARD_GROUPS = {
    "five": ranked_real_boards.fiveBoards,
    "six": ranked_real_boards.sixBoards,
    "seven": ranked_real_boards.sevenBoards,
    "eight": ranked_real_boards.eightBoards,
    "nine": ranked_real_boards.nineBoards,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export expert child-path ranking rows as JSONL."
    )
    parser.add_argument("output_path")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(BOARD_GROUPS),
        default=["five"],
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-states", type=int, default=100_000)
    parser.add_argument("--mode", choices=("hybrid", "fast", "exact"), default="hybrid")
    args = parser.parse_args()

    boards = []
    for group in args.groups:
        boards.extend(BOARD_GROUPS[group][: args.limit])

    stats = export_expert_rows(
        boards,
        args.output_path,
        max_states=args.max_states,
        mode=args.mode,
    )
    print(stats)


if __name__ == "__main__":
    main()
