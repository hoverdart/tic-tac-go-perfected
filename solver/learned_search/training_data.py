"""Expert trace export for training learned child-path rankers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from solver import optimized_solver
from solver.board_utils import normalize_board
from solver.learned_search.features import candidate_row, segment_result_key


def expert_rows_for_solution(
    board,
    moves: str,
    *,
    board_id: str | None = None,
) -> list[dict]:
    """Label every candidate child at each expert state.

    The expert path comes from `optimized_solver.solve()`. At each parent state
    we enumerate the same compressed children the optimized solver uses and mark
    the one whose segment matches the remaining expert move string.
    """
    start_board = normalize_board(board)
    geometry = optimized_solver._geometry_for_board(start_board)
    parent_key = optimized_solver._to_key(start_board)
    remaining_moves = moves or ""
    rows: list[dict] = []
    depth = 0

    while remaining_moves:
        candidates = list(optimized_solver._next_states(parent_key, geometry))
        expert_child_key = None
        expert_segment = None

        for child_key, segment in candidates:
            if not remaining_moves.startswith(segment):
                continue
            if segment_result_key(parent_key, geometry, segment) != child_key:
                continue
            expert_child_key = child_key
            expert_segment = segment
            break

        if expert_child_key is None or expert_segment is None:
            raise ValueError(
                f"Could not align expert moves at depth {depth}: {remaining_moves!r}"
            )

        for child_key, segment in candidates:
            label = 1 if child_key == expert_child_key and segment == expert_segment else 0
            rows.append(
                candidate_row(
                    board_id=board_id,
                    depth=depth,
                    parent_key=parent_key,
                    child_key=child_key,
                    segment=segment,
                    geometry=geometry,
                    label=label,
                    remaining_cost=len(remaining_moves),
                )
            )

        parent_key = expert_child_key
        remaining_moves = remaining_moves[len(expert_segment) :]
        depth += 1

    return rows


def export_expert_rows(
    boards: Iterable,
    output_path: str | Path,
    *,
    max_states: int | None = 100_000,
    mode: str = "hybrid",
) -> dict[str, int]:
    """Solve boards with the expert and write candidate rows to JSONL."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    solved = 0
    skipped = 0
    rows_written = 0
    with output.open("w", encoding="utf-8") as file:
        for index, board in enumerate(boards):
            moves, _final_board, _states = optimized_solver.solve(
                board,
                progress_every=0,
                max_states=max_states,
                mode=mode,
            )
            if moves is None:
                skipped += 1
                continue

            solved += 1
            rows = expert_rows_for_solution(board, moves, board_id=str(index))
            for row in rows:
                file.write(json.dumps(row, sort_keys=True) + "\n")
                rows_written += 1

    return {"solved": solved, "skipped": skipped, "rows_written": rows_written}
