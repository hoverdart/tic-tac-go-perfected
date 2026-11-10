"""Bounded local rewrites of a solution produced by the current invocation.

Piece identities exist only while permuting this incumbent's pushes. Each
candidate is replayed with shortest walks, checked for intermediate X losses,
and independently verified before it replaces the incumbent. No saved paths
or board identities enter the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
import time

from solver.push_solver.models import Push
from solver.push_solver.verify import verify_solution


class RewriteExpired(Exception):
    pass


@dataclass(frozen=True)
class RewriteResult:
    moves: str
    pushes: tuple[Push, ...]
    evaluations: int
    improvements: int


class PathRewriter:
    def __init__(self, context, deadline):
        self.context = context
        self.deadline = deadline
        self.walk_cache = {}
        self.evaluations = 0
        self.improvements = 0
        self.initial = tuple(
            i for mask in (context.os, context.xs)
            for i in range(context.size) if mask & (1 << i)
        )
        self.types = tuple(
            "O" if context.os & (1 << position) else "X" for position in self.initial
        )

    def check(self):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise RewriteExpired

    def walks(self, player, occupied):
        key = player, occupied
        cached = self.walk_cache.get(key)
        if cached is not None:
            return cached
        paths = {player: ""}
        queue = [player]
        for current in queue:
            for move, nxt in self.context.adj[current]:
                if occupied & (1 << nxt) or nxt in paths:
                    continue
                paths[nxt] = paths[current] + move
                queue.append(nxt)
        region = sum(1 << position for position in paths)
        if len(self.walk_cache) >= 1024:
            self.walk_cache.pop(next(iter(self.walk_cache)))
        self.walk_cache[key] = paths, region
        return paths, region

    def label(self, pushes):
        positions = {position: label for label, position in enumerate(self.initial)}
        operations = []
        for push in pushes:
            label = positions.pop(push.cell)
            destination = self.context.push_lookup[push.cell, push.move][1]
            positions[destination] = label
            operations.append((label, push.move))
        return tuple(operations)

    def replay(self, operations, bound):
        self.check()
        self.evaluations += 1
        ctx = self.context
        positions = list(self.initial)
        os, xs, player = ctx.os, ctx.xs, ctx.player
        parts, pushes = [], []
        cost = 0
        for index in range(len(operations) + 1):
            self.check()
            paths, region = self.walks(player, os | xs)
            if ctx.goal((os, xs, region))[0]:
                options = [
                    paths[target] if target is not None else ""
                    for target in ctx.goal_cache[os]
                    if target is None or target in paths
                ]
                walk = min(options, key=lambda path: (len(path), path))
                if cost + len(walk) < bound:
                    return "".join(parts) + walk, tuple(pushes), operations[:index]
            if index == len(operations):
                return None
            label, move = operations[index]
            cell = positions[label]
            edge = ctx.push_lookup.get((cell, move))
            if edge is None:
                return None
            stand, destination = edge
            if (os | xs) & (1 << destination) or stand not in paths:
                return None
            walk = paths[stand]
            cost += len(walk) + 1
            if cost >= bound:
                return None
            piece = self.types[label]
            if piece == "O":
                os = (os ^ (1 << cell)) | (1 << destination)
            else:
                xs = (xs ^ (1 << cell)) | (1 << destination)
                if ctx.x_loss(xs, destination):
                    return None
            positions[label] = destination
            player = cell
            parts.extend((walk, move))
            pushes.append(Push(piece, cell, move))
        return None

    def optimize(self, moves, pushes):
        operations = self.label(pushes)
        best_moves, best_pushes = moves, pushes

        def accept(candidate):
            nonlocal operations, best_moves, best_pushes
            result = self.replay(tuple(candidate), len(best_moves))
            if result is None or not verify_solution(self.context.board, result[0]).ok:
                return False
            best_moves, best_pushes, operations = result
            self.improvements += 1
            return True

        try:
            accept(operations)
            for _ in range(20):
                changed = False
                # Removing an entire piece's work can eliminate unnecessary X
                # detours; subsequent pushes are replayed by identity, not cell.
                for label in sorted({operation[0] for operation in operations}):
                    if accept([operation for operation in operations if operation[0] != label]):
                        changed = True
                        break
                if changed:
                    continue
                blocks = [tuple(group) for _, group in groupby(operations, key=lambda op: op[0])]
                for index in range(len(blocks)):
                    if accept(sum(blocks[:index] + blocks[index + 1:], ())):
                        changed = True
                        break
                if changed:
                    continue
                for index in range(len(operations)):
                    if accept(operations[:index] + operations[index + 1:]):
                        changed = True
                        break
                if changed:
                    continue
                # Bring nearby same-piece runs together when the intervening
                # pushes commute legally. Replay rejects collisions and X loss.
                for index in range(len(blocks)):
                    for other in range(index + 1, min(len(blocks), index + 9)):
                        candidate = sum(
                            blocks[:index] + [blocks[other]]
                            + blocks[index:other] + blocks[other + 1:], ()
                        )
                        if accept(candidate):
                            changed = True
                            break
                    if changed:
                        break
                if not changed:
                    break
        except RewriteExpired:
            pass
        return RewriteResult(best_moves, best_pushes, self.evaluations, self.improvements)
