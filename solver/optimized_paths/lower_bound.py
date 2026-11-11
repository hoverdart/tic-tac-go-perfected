"""Complete reverse unit-step BFS with Xs removed, for exactly two movable Os."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .model import Budget, Geometry, LimitReached, cells


@dataclass
class PatternDatabase:
    distances: dict[tuple[int, int], int]
    complete: bool
    expanded: int
    reason: str

    def estimate(self, state):
        if not self.complete:
            return 0
        return self.distances.get((state[0], state[2]), math.inf)


def build_pattern_database(geometry: Geometry, budget: Budget, *, max_states=130_000):
    if geometry.start[0].bit_count() != 2:
        return PatternDatabase({}, False, 0, "additional_o_fallback")
    distances = {}
    queue = deque()
    expanded = 0
    try:
        for line in geometry.lines:
            budget.check()
            for player in cells(line):
                key = line ^ (1 << player), player
                if key not in distances:
                    if len(distances) >= max_states:
                        return PatternDatabase({}, False, expanded, "state_cap")
                    distances[key] = 0
                    queue.append(key)
        while queue:
            budget.check()
            os, player = queue.popleft()
            expanded += 1
            for move, previous_player in geometry.adj[player]:
                if os & (1 << previous_player):
                    continue
                # Reverse of a walk: previous player can step into this cell.
                predecessors = [(os, previous_player)]
                opposite = {"U": "D", "D": "U", "L": "R", "R": "L"}[move]
                pushed_to = geometry.edges.get((player, opposite))
                if pushed_to is not None and os & (1 << pushed_to):
                    # Reverse of a push: the O was on the current player cell.
                    predecessors.append(
                        ((os ^ (1 << pushed_to)) | (1 << player), previous_player)
                    )
                for key in predecessors:
                    if key in distances:
                        continue
                    if len(distances) >= max_states:
                        return PatternDatabase({}, False, expanded, "state_cap")
                    distances[key] = distances[os, player] + 1
                    queue.append(key)
    except LimitReached as exc:
        return PatternDatabase({}, False, expanded, str(exc))
    return PatternDatabase(distances, True, expanded, "complete")
