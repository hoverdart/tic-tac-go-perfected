"""Resumable PHSh/Levin search with probability-aware graph dominance.

PHSh evaluates (g+h)/pi with g = push depth + 1. This implementation uses
nonadmissible V4 h, graph pruning, and finite budgets: no paper guarantee is
claimed. Monotone ancestor priorities and Pareto labels are separate from V4's
cost-only and rank-discrepancy tables. Primitive successor legality is shared.
"""

from __future__ import annotations

import heapq
import math


def log_priority(
    depth, negative_log_probability, heuristic, ancestor=-math.inf, *, aggressive=False
):
    h = max(0.0, heuristic) if math.isfinite(heuristic) else 50.0
    factor = 1 + h / (depth + 1) if aggressive else 1.0
    return max(ancestor, math.log(depth + 1 + h) + factor * negative_log_probability)


def dominates(a, b):
    """Lower depth, negative log probability, and ancestor bound are all needed."""
    return all(x <= y for x, y in zip(a, b, strict=True))


def search(session):
    from solver.push_solver.v4 import Found, Link

    ctx = session.context
    aggressive = session.kind == "phs_star"
    h, _ = yield from ctx.score(ctx.start, limit=1)
    root_priority = log_priority(
        0, 0.0, h if session.kind != "levin" else 0.0, aggressive=aggressive
    )
    serial = next(session.counter)
    labels = {ctx.start: {serial: (0, 0.0, root_priority)}}
    queue = [(root_priority, serial, 0, 0.0, ctx.start, None)]
    label_count = 1
    closed = set()
    while queue:
        yield from ctx.checkpoint()
        priority, serial, depth, nlp, current, link = heapq.heappop(queue)
        if aggressive and current in closed:
            session.counters["pruned_closed_policy_state"] += 1
            continue
        if serial not in labels[current]:
            session.counters["pruned_stale_policy_label"] += 1
            continue
        if aggressive:
            # First-expanded-state graph rule, as in the paper's Algorithm 1.
            # Depth-based Pareto pruning is unsafe with a depth-dependent exponent.
            closed.add(current)
            label_count -= len(labels[current])
            del labels[current]
        session.expand()
        won, target = ctx.goal(current)
        if won:
            yield Found(current, target, link)
            return
        children = yield from ctx.children(current, False)
        for child in children:
            won, target = ctx.goal(child.state)
            if won:
                yield Found(child.state, target, Link(link, child.pushes))
                return
        probabilities = yield from ctx.push_policy.distribution(ctx, current, children)
        session.counters["policy_evaluations"] += 1
        session.counters["policy_actions"] += len(children)
        for child, probability in zip(children, probabilities, strict=True):
            yield from ctx.checkpoint()
            cost, child_nlp = depth + 1, nlp - math.log(probability)
            value = log_priority(
                cost,
                child_nlp,
                child.h if session.kind != "levin" else 0.0,
                priority,
                aggressive=aggressive,
            )
            if aggressive and child.state in closed:
                session.counters["pruned_closed_policy_state"] += 1
                continue
            metric = cost, child_nlp, value
            bucket = labels.setdefault(child.state, {})
            dominated = False
            for index, old in enumerate(bucket.values()):
                if index and index % 32 == 0:
                    yield from ctx.checkpoint()
                if (not aggressive or old[0] == metric[0]) and dominates(old, metric):
                    dominated = True
                    break
            if dominated:
                session.counters["pruned_policy_dominance"] += 1
                continue
            for index, (old_serial, old) in enumerate(tuple(bucket.items())):
                if index and index % 32 == 0:
                    yield from ctx.checkpoint()
                if (not aggressive or old[0] == metric[0]) and dominates(metric, old):
                    del bucket[old_serial]
                    label_count -= 1
            serial = next(session.counter)
            bucket[serial] = metric
            label_count += 1
            heapq.heappush(
                queue,
                (value, serial, cost, child_nlp, child.state, Link(link, child.pushes)),
            )
        session.peak = max(session.peak, label_count + len(closed))
        session.frontier_size = len(queue)
        session.frontier_peak = max(session.frontier_peak, len(queue))
        ctx.peak_frontier = max(ctx.peak_frontier, len(queue))
