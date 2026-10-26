"""Feature-only conditional push probabilities for the opt-in PHS experiment.

Training and search call the same extractor. No board identity, coordinates,
action lookup, or history enters the model. Policies are Markov in V4 state.
"""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path

FEATURES = (
    "child_h",
    "h_delta",
    "bias",
    "piece_o",
    "region_gain",
    "o_mobility_gain",
    "source_degree",
    "dest_degree",
    "clears_line",
    "enters_line",
    "clears_route",
    "enters_route",
    "clears_stand",
    "enters_stand",
    "clears_target",
    "enters_target",
    "target_access_gain",
    "assigned_progress",
    "x_threat_delta",
    "near_o_gain",
    "near_x_gain",
    "goal",
    "sibling_rank",
    "o_h_delta",
    "x_h_delta",
    "o_region_gain",
    "x_region_gain",
    "o_assigned_progress",
    "x_clears_route",
    "x_enters_route",
    "dense_x_region_gain",
    "dense_x_clears_route",
    "narrow_region_gain",
)
SCHEMA = "v4-push-softmax-v1"
MLP_SCHEMA = "v4-push-mlp-v1"


def bounded(value, scale=1.0):
    return max(-5.0, min(5.0, value / scale)) if math.isfinite(value) else 5.0


def action_features(ctx, state, children):
    """Yield a feature tuple per child, preserving the caller's action order."""
    from solver.push_solver.v4 import cells

    parent_h, plans = yield from ctx.score(state, limit=8)
    line = route = stands = targets = 0
    for plan in plans:
        line |= plan.line
        route |= plan.route
        stands |= plan.stands
        targets |= 1 << plan.target
    os, xs, region = state
    floor = max(1, ctx.floor.bit_count())
    density = xs.bit_count() / floor
    narrow = sum(len(ctx.adj[c]) <= 2 for c in cells(ctx.floor)) / floor
    mobility = ctx.o_push_count(state)
    before_threats = sum((xs & mask).bit_count() == 2 for mask in ctx.lines)
    # The rank is a deterministic hand-heuristic feature, not learned ordering.
    ranked = sorted(
        range(len(children)),
        key=lambda i: (children[i].h + children[i].bias, children[i].h),
    )
    ranks = {
        index: rank / max(1, len(children) - 1) for rank, index in enumerate(ranked)
    }
    result = []
    for index, child in enumerate(children):
        yield from ctx.checkpoint()
        if len(child.pushes) != 1:
            raise ValueError("Push policy accepts primitive pushes only")
        push = child.pushes[0]
        dest = ctx.push_lookup[push.cell, push.move][1]
        source_bit, dest_bit = 1 << push.cell, 1 << dest
        o = float(push.piece == "O")
        h_delta = (
            bounded(child.h - parent_h, 10.0)
            if math.isfinite(child.h) and math.isfinite(parent_h)
            else -5.0
            if math.isfinite(child.h)
            else 5.0
            if math.isfinite(parent_h)
            else 0.0
        )
        region_gain = (child.state[2].bit_count() - region.bit_count()) / floor
        assigned = 0.0
        if o and plans:
            for source, target in plans[0].assignment:
                if source == push.cell:
                    before, after = (
                        ctx.distances[target][source],
                        ctx.distances[target][dest],
                    )
                    assigned = (
                        bounded(before - after, 5.0)
                        if before >= 0 and after >= 0
                        else 0.0
                    )
                    break

        def clears(mask, source_bit=source_bit, dest_bit=dest_bit):
            return float(bool(mask & source_bit) and not bool(mask & dest_bit))

        def enters(mask, source_bit=source_bit, dest_bit=dest_bit):
            return float(bool(mask & dest_bit) and not bool(mask & source_bit))

        def adjacent_gain(mask, source_bit=source_bit, dest=dest, source=push.cell):
            mask &= ~source_bit
            return (
                sum(bool(mask & (1 << c)) for _, c in ctx.adj[dest])
                - sum(bool(mask & (1 << c)) for _, c in ctx.adj[source])
            ) / 4.0

        threat_delta = (
            sum((child.state[1] & mask).bit_count() == 2 for mask in ctx.lines)
            - before_threats
        )
        result.append(
            (
                bounded(child.h, 20.0),
                h_delta,
                bounded(child.bias, 10.0),
                o,
                region_gain,
                bounded(ctx.o_push_count(child.state) - mobility, 8.0),
                len(ctx.adj[push.cell]) / 4.0,
                len(ctx.adj[dest]) / 4.0,
                clears(line),
                enters(line),
                clears(route),
                enters(route),
                clears(stands),
                enters(stands),
                clears(targets),
                enters(targets),
                (
                    (child.state[2] & targets).bit_count()
                    - (region & targets).bit_count()
                )
                / max(1, targets.bit_count()),
                assigned,
                bounded(threat_delta, 8.0),
                adjacent_gain(os),
                adjacent_gain(xs),
                float(ctx.goal(child.state)[0]),
                ranks[index],
                o * h_delta,
                (1 - o) * h_delta,
                o * region_gain,
                (1 - o) * region_gain,
                o * assigned,
                (1 - o) * clears(route),
                (1 - o) * enters(route),
                density * region_gain,
                density * clears(route),
                narrow * region_gain,
            )
        )
    return tuple(result)


class PushPolicy:
    def __init__(self, path):
        raw = Path(path).read_bytes()
        self.hash = sha256(raw).hexdigest()
        self.payload = json.loads(raw)
        allowed = {"schema", "name", "weights", "temperature", "uniform_mix"}
        if isinstance(self.payload, dict) and self.payload.get("schema") == MLP_SCHEMA:
            allowed = {
                "schema",
                "name",
                "hidden_weights",
                "hidden_bias",
                "output_weights",
                "temperature",
                "uniform_mix",
            }
        if (
            not isinstance(self.payload, dict)
            or set(self.payload) != allowed
            or self.payload["schema"] not in (SCHEMA, MLP_SCHEMA)
        ):
            raise ValueError(
                "Push policy must match the feature-only schema; action hints are forbidden"
            )
        self.network = None
        self.weights = ()
        if self.payload["schema"] == SCHEMA:
            weights = self.payload["weights"]
            if not isinstance(weights, dict) or set(weights) - set(FEATURES):
                raise ValueError("Unknown push-policy features")
            self.weights = tuple(float(weights.get(name, 0.0)) for name in FEATURES)
        else:
            import numpy as np

            matrix, bias, output = (
                np.asarray(self.payload[key], dtype=np.float64)
                for key in ("hidden_weights", "hidden_bias", "output_weights")
            )
            if (
                matrix.ndim != 2
                or matrix.shape[0] != len(FEATURES)
                or not 1 <= matrix.shape[1] <= 128
                or bias.shape != (matrix.shape[1],)
                or output.shape != bias.shape
            ):
                raise ValueError("Invalid push-policy network dimensions")
            if not all(np.isfinite(value).all() for value in (matrix, bias, output)):
                raise ValueError("Push policy coefficients must be finite")
            self.network = matrix, bias, output
        self.temperature = float(self.payload["temperature"])
        self.uniform_mix = float(self.payload["uniform_mix"])
        if not all(
            math.isfinite(v)
            for v in (*self.weights, self.temperature, self.uniform_mix)
        ):
            raise ValueError("Push policy coefficients must be finite")
        if self.temperature <= 0 or not 1e-6 <= self.uniform_mix <= 1:
            raise ValueError(
                "Push policy needs positive temperature and a nonzero uniform floor"
            )

    def probabilities(self, features):
        if not features:
            return ()
        if self.network is None:
            logits = [
                sum(w * x for w, x in zip(self.weights, row, strict=True))
                for row in features
            ]
        else:
            import numpy as np

            x = np.asarray(features, dtype=np.float64)
            if x.shape != (len(features), len(FEATURES)):
                raise ValueError("Invalid push-policy feature dimensions")
            matrix, bias, output = self.network
            logits = (np.tanh(x @ matrix + bias) @ output).tolist()
        if not all(math.isfinite(v) for v in logits):
            raise ValueError("Nonfinite push-policy logit")
        maximum = max(logits)
        exp = [math.exp(max(-745.0, (v - maximum) / self.temperature)) for v in logits]
        total = sum(exp)
        n = len(exp)
        return tuple(
            (1 - self.uniform_mix) * v / total + self.uniform_mix / n for v in exp
        )

    def distribution(self, ctx, state, children):
        ctx.check()
        key = self.hash, state, tuple(child.pushes for child in children)
        if key in ctx.policy_cache:
            ctx.metrics["policy_cache_hits"] += 1
            ctx.policy_cache.move_to_end(key)
            return ctx.policy_cache[key]
        features = yield from action_features(ctx, state, children)
        probabilities = self.probabilities(features)
        ctx.metrics["policy_computations"] += 1
        return ctx.put(ctx.policy_cache, key, probabilities, 8192)
