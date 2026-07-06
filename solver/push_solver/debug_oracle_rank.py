"""Inspect successor ordering against known verified push paths."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from backfill_solutions import ALL_PAST_DAYS, board_from_entry
from solver.push_solver import core


KNOWN_SOLUTIONS_PATH = (
    Path(__file__).parents[1]
    / "gymnasium_register"
    / "all_boards_heuristic_cnn_solutions.jsonl"
)


def _known_solution(board_id: str) -> str:
    for line in KNOWN_SOLUTIONS_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("id") == board_id:
            solution = row.get("solution")
            if not solution:
                raise ValueError(f"Board id {board_id!r} has no known solution string.")
            return str(solution)
    raise ValueError(f"No known solution found for board id {board_id!r}.")


def _board_entry(board_id: str) -> dict:
    for entry in ALL_PAST_DAYS:
        if entry.get("id") == board_id:
            return entry
    raise ValueError(f"No board found for board id {board_id!r}.")


def _push_path(board, moves: str) -> list[tuple[int, core.State, core.Push]]:
    static, state, _normalized, player = core.parse_board(board)
    os = state.os
    xs = state.xs
    path: list[tuple[int, core.State, core.Push]] = []

    for step, move in enumerate(moves, start=1):
        dr, dc = core.DIRECTION_BY_MOVE[move]
        row, col = static.coord(player)
        nxt = static.index(row + dr, col + dc)
        if nxt in os or nxt in xs:
            piece = "O" if nxt in os else "X"
            current_state = core.normalize_state(player, os, xs, static)
            push = core.Push(piece=piece, cell=nxt, move=move)
            path.append((step, current_state, push))

            dest = static.index(row + (2 * dr), col + (2 * dc))
            if piece == "O":
                os = frozenset((os - {nxt}) | {dest})
            else:
                xs = frozenset((xs - {nxt}) | {dest})
            player = nxt
        else:
            player = nxt

    return path


def _plan_coords(plan: core.LinePlan, static: core.StaticBoard) -> str:
    line = tuple(static.coord(cell) for cell in plan.line)
    target = static.coord(plan.player_target)
    return (
        f"line={line} target={target} score={plan.score:.1f} "
        f"x_line={plan.x_on_line_count} blocked_route={plan.blocked_route_count}"
    )


def inspect_board(board_id: str) -> None:
    entry = _board_entry(board_id)
    board = board_from_entry(entry)
    moves = _known_solution(board_id)
    static, start_state, _normalized, _player = core.parse_board(board)
    path = _push_path(board, moves)

    start_region = core.reachable(start_state.player, start_state.os, start_state.xs, static)
    target_access_penalty = (
        3.0
        if len(start_region) <= 2
        and core._legal_o_push_count(start_state, static, start_region) > 0
        else 0.0
    )
    plan_cache: dict[core.State, core.LinePlan | None] = {}
    top_plan_cache: dict[core.State, tuple[core.LinePlan, ...]] = {}
    o_push_count_cache: dict[core.State, int] = {}

    ranks: list[int] = []
    not_top_plan_helpful = 0
    first_rank_gt_10: tuple[int, int] | None = None
    first_rank_gt_25: tuple[int, int] | None = None

    print(f"{board_id} {entry.get('name', '')}: pushes={len(path)} moves={len(moves)}")
    for depth, (step, state, oracle_push) in enumerate(path, start=1):
        region = core.reachable(state.player, state.os, state.xs, static)
        top_plans = core._top_line_plans_for(
            state,
            static,
            region=region,
            target_access_penalty=target_access_penalty,
            top_plan_cache=top_plan_cache,
        )
        h = top_plans[0].score if top_plans else float("inf")
        parent_o_push_count = core._legal_o_push_count_for(
            state,
            static,
            region,
            o_push_count_cache,
        )
        successors = core.successors(
            state,
            static,
            region=region,
            parent_h=h,
            target_access_penalty=target_access_penalty,
            plan_cache=plan_cache,
            top_plan_cache=top_plan_cache,
            o_push_count_cache=o_push_count_cache,
        )
        ordered = sorted(successors, key=lambda item: (item[3] + item[4], item[3]))
        oracle_key = (oracle_push.piece, oracle_push.cell, oracle_push.move)
        rank = next(
            (
                index
                for index, (push, *_rest) in enumerate(ordered, start=1)
                if (push.piece, push.cell, push.move) == oracle_key
            ),
            0,
        )
        if rank == 0:
            raise RuntimeError(f"Oracle push missing at depth {depth}: {oracle_push}")
        ranks.append(rank)
        if rank > 10 and first_rank_gt_10 is None:
            first_rank_gt_10 = (depth, rank)
        if rank > 25 and first_rank_gt_25 is None:
            first_rank_gt_25 = (depth, rank)

        best_push, _best_state, best_region, _best_h, _best_bias = ordered[0]
        oracle_child = next(item for item in ordered if item[0] == oracle_push)
        _push, _child_state, child_region, _child_h, _bias = oracle_child
        child_o_push_count = core._legal_o_push_count_for(
            _child_state,
            static,
            child_region,
            o_push_count_cache,
        )
        helps_plan = any(
            core._plan_specific_push_bias(
                plan=plan,
                push=oracle_push,
                parent_region=region,
                child_region=child_region,
                board=static,
            )
            < 0
            for plan in top_plans
        )
        if not helps_plan:
            not_top_plan_helpful += 1

        if rank > 10:
            best_plan = top_plans[0] if top_plans else None
            print(
                f"depth={depth} step={step} h={h:.1f} rank={rank} "
                f"oracle={oracle_push} best={best_push} "
                f"succ={len(ordered)} region_delta={len(child_region) - len(region)} "
                f"o_push_delta={child_o_push_count - parent_o_push_count} "
                f"helps_top_plan={helps_plan}"
            )
            if best_plan is not None:
                print(f"  active {_plan_coords(best_plan, static)}")
                for plan_index, plan in enumerate(top_plans[:4], start=1):
                    print(f"  top{plan_index} {_plan_coords(plan, static)}")

    print(f"median_rank={statistics.median(ranks):.1f}")
    print(f"max_rank={max(ranks)}")
    print(f"first_rank_gt_10={first_rank_gt_10}")
    print(f"first_rank_gt_25={first_rank_gt_25}")
    print(f"not_top_plan_helpful={not_top_plan_helpful}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-id", required=True)
    args = parser.parse_args()
    inspect_board(args.board_id)


if __name__ == "__main__":
    main()
