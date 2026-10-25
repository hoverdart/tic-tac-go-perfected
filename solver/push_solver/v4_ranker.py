"""Optional offline feature-ranker adapter. Never consults state-action hints."""

from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from solver.push_solver.models import State
from solver.push_solver.policy_features import features_for_child
from solver.push_solver.rank_policy import load_policy
from solver.push_solver.state import parse_board


class FeatureRanker:
    def __init__(self, path, board):
        self.hash = sha256(Path(path).read_bytes()).hexdigest()
        self.policy = load_policy(path)
        if self.policy.state_action_hints or self.policy.priority_recovery:
            raise ValueError(
                "V4 only accepts feature-only candidates without exact action hints"
            )
        static, _, _, _ = parse_board(board)
        self.context = SimpleNamespace(
            static=static,
            region_cache={},
            h_cache={},
            top_plan_cache={},
            o_push_count_cache={},
            target_access_penalty=0.0,
        )
        self.cache = OrderedDict()

    def score(self, ctx, state, child, rank, count, parent_h):
        from solver.push_solver.v4 import cells

        key = state, child.pushes, child.state, rank, count
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        def converted(s):
            return State(
                next(cells(s[2])), frozenset(cells(s[0])), frozenset(cells(s[1]))
            )

        parent, nxt = converted(state), converted(child.state)
        view = self.context
        view.region_cache = {
            parent: frozenset(cells(state[2])),
            nxt: frozenset(cells(child.state[2])),
        }
        view.h_cache = {
            parent: parent_h,
            nxt: child.h,
        }
        features = features_for_child(
            view,
            parent,
            child.pushes,
            nxt,
            view.region_cache[nxt],
            child.h,
            child.bias,
            len(child.pushes),
        )
        features["sibling_rank"] = rank / max(1, count)
        features["sibling_count"] = count / 32.0
        value = self.policy.score(features)
        self.cache[key] = value
        if len(self.cache) > 8192:
            self.cache.popitem(last=False)
        for cache in (view.top_plan_cache, view.o_push_count_cache):
            if len(cache) > 2048:
                cache.clear()
        return value
