"""Exact move-cost geometry, without V4's heuristic pruning or policy scores."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from hashlib import sha256

from solver.board_utils import normalize_board
from solver.push_solver.models import DIRECTIONS, Push
from solver.push_solver.verify import verify_solution

State = tuple[int, int, int]  # O occupancy, X occupancy, actual player cell
Layout = tuple[int, int]


class LimitReached(Exception):
    pass


@dataclass
class Budget:
    deadline: float
    max_nodes: int
    expanded: int = 0

    def check(self):
        if time.perf_counter() >= self.deadline:
            raise LimitReached("timeout")

    def expand(self):
        self.check()
        if self.expanded >= self.max_nodes:
            raise LimitReached("node_cap")
        self.expanded += 1


def cells(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


class Geometry:
    def __init__(self, board):
        self.board = normalize_board(board)
        self.rows, self.cols = len(self.board), len(self.board[0])
        self.digest = sha256(
            json.dumps(self.board, separators=(",", ":")).encode()
        ).hexdigest()
        self.floor = sum(
            1 << (r * self.cols + c)
            for r, row in enumerate(self.board)
            for c, value in enumerate(row)
            if value != "B"
        )
        masks = {
            piece: sum(
                1 << (r * self.cols + c)
                for r, row in enumerate(self.board)
                for c, value in enumerate(row)
                if value == piece
            )
            for piece in ("O", "X", "U")
        }
        self.start = masks["O"], masks["X"], masks["U"].bit_length() - 1
        self.lines = []
        self.adj = [[] for _ in range(self.rows * self.cols)]
        self.edges = {}
        self.pushes = [[] for _ in self.adj]
        for cell in cells(self.floor):
            r, c = divmod(cell, self.cols)
            for move, dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    target = nr * self.cols + nc
                    if self.floor & (1 << target):
                        self.adj[cell].append((move, target))
                        self.edges[cell, move] = target
                        sr, sc = r - dr, c - dc
                        if 0 <= sr < self.rows and 0 <= sc < self.cols:
                            stand = sr * self.cols + sc
                            if self.floor & (1 << stand):
                                self.pushes[cell].append((move, stand, target))
            for dr, dc in ((0, 1), (1, 0)):
                coords = [(r + i * dr, c + i * dc) for i in range(3)]
                if all(
                    rr < self.rows
                    and cc < self.cols
                    and self.floor & (1 << (rr * self.cols + cc))
                    for rr, cc in coords
                ):
                    self.lines.append(
                        sum(1 << (rr * self.cols + cc) for rr, cc in coords)
                    )
        self.lines = tuple(self.lines)

    def won(self, state):
        os, _xs, player = state
        owned = os | (1 << player)
        return any(owned & line == line for line in self.lines)

    def x_loss(self, xs):
        return any(xs & line == line for line in self.lines)

    def step(self, state, move):
        os, xs, player = state
        target = self.edges.get((player, move))
        if target is None:
            return None
        bit = 1 << target
        if (os | xs) & bit:
            dest = self.edges.get((target, move))
            if dest is None or (os | xs) & (1 << dest):
                return None
            if os & bit:
                os = (os ^ bit) | (1 << dest)
            else:
                xs = (xs ^ bit) | (1 << dest)
                if self.x_loss(xs):
                    return None
        return os, xs, target

    def trace(self, moves):
        state = self.start
        yield state
        for move in moves:
            state = self.step(state, move)
            if state is None:
                raise ValueError("Invalid trajectory")
            yield state

    def walks(self, state):
        os, xs, player = state
        paths = {player: ""}
        queue = deque([player])
        while queue:
            cell = queue.popleft()
            for move, target in self.adj[cell]:
                if target not in paths and not (os | xs) & (1 << target):
                    paths[target] = paths[cell] + move
                    queue.append(target)
        return paths

    def finish(self, state, walks):
        os, _xs, _player = state
        choices = []
        for line in self.lines:
            remaining = line & ~os
            if not remaining:
                return ""  # Three stationary Os; player may be elsewhere.
            if remaining.bit_count() == 1:
                target = remaining.bit_length() - 1
                if target in walks:
                    choices.append(walks[target])
        return min(choices, key=lambda moves: (len(moves), moves)) if choices else None

    def successors(self, state, walks):
        os, xs, _player = state
        for piece, mask in (("O", os), ("X", xs)):
            for cell in cells(mask):
                for move, stand, dest in self.pushes[cell]:
                    if stand not in walks or (os | xs) & (1 << dest):
                        continue
                    changed = (mask ^ (1 << cell)) | (1 << dest)
                    if piece == "X" and self.x_loss(changed):
                        continue
                    child = (changed, xs, cell) if piece == "O" else (os, changed, cell)
                    yield child, walks[stand] + move

    def push_trace(self, moves):
        result = []
        states = self.trace(moves)
        before = next(states)
        for move, after in zip(moves, states):
            target = after[2]
            bit = 1 << target
            if (before[0] | before[1]) & bit:
                result.append(Push("O" if before[0] & bit else "X", target, move))
            before = after
        return tuple(result)

    def verified(self, moves):
        checked = verify_solution(self.board, moves)
        if not checked.ok:
            raise ValueError(f"Invalid solution: {checked.error}")
        return checked.final_board
