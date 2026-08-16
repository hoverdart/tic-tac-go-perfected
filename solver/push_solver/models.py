"""Shared data models and constants for the push solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


DIRECTIONS: tuple[tuple[str, int, int], ...] = (
    ("U", -1, 0),
    ("D", 1, 0),
    ("L", 0, -1),
    ("R", 0, 1),
)
DIRECTION_BY_MOVE = {move: (dr, dc) for move, dr, dc in DIRECTIONS}
EMPTY_ROUTE: tuple[int, ...] = ()
EMPTY_CELL_SET: frozenset[int] = frozenset()


@dataclass(frozen=True)
class StaticBoard:
    rows: int
    cols: int
    walls: frozenset[int]
    floor: frozenset[int]
    win_lines: tuple[tuple[int, int, int], ...]
    dead_cells_for_o: frozenset[int]
    push_distances: Mapping[int, Mapping[int, int]]
    push_predecessors: Mapping[int, Mapping[int, int]]
    push_stand_cells: Mapping[int, Mapping[int, int]]
    push_routes: Mapping[int, Mapping[int, tuple[int, ...]]]
    push_stand_routes: Mapping[int, Mapping[int, tuple[int, ...]]]
    adjacency: tuple[tuple[int, ...], ...]
    push_transitions: tuple[tuple[tuple[str, int, int], ...], ...]
    push_route_sets: Mapping[int, Mapping[int, frozenset[int]]]
    push_stand_route_sets: Mapping[int, Mapping[int, frozenset[int]]]
    reachable_o_pairs: frozenset[tuple[int, int]]

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def index(self, row: int, col: int) -> int:
        return (row * self.cols) + col

    def coord(self, index: int) -> tuple[int, int]:
        return divmod(index, self.cols)

    def neighbor(self, index: int, move: str) -> int | None:
        row, col = self.coord(index)
        dr, dc = DIRECTION_BY_MOVE[move]
        next_row = row + dr
        next_col = col + dc
        if not self.in_bounds(next_row, next_col):
            return None
        next_index = self.index(next_row, next_col)
        if next_index in self.walls:
            return None
        return next_index


@dataclass(frozen=True)
class State:
    player: int
    os: frozenset[int]
    xs: frozenset[int]


@dataclass(frozen=True)
class Push:
    piece: str
    cell: int
    move: str


@dataclass(frozen=True)
class Parent:
    previous: State
    push: Push | None = None
    pushes: tuple[Push, ...] = ()

    def push_tuple(self) -> tuple[Push, ...]:
        if self.pushes:
            return self.pushes
        if self.push is not None:
            return (self.push,)
        return ()


@dataclass(frozen=True)
class GoalInfo:
    line: tuple[int, int, int]
    player_target: int


@dataclass(frozen=True)
class LinePlan:
    score: float
    line: tuple[int, int, int]
    player_target: int
    o_targets: tuple[int, int]
    o_assignment: tuple[tuple[int, int], tuple[int, int]]
    route_cells: frozenset[int]
    stand_cells: frozenset[int]
    player_target_has_x: bool
    player_target_reachable: bool
    x_on_line_count: int
    blocked_route_count: int


@dataclass(frozen=True)
class SearchAttempt:
    strategy: str
    solved: bool
    nodes_expanded: int
    peak_closed_size: int
    elapsed_ms: float
    failure_reason: str | None


@dataclass(frozen=True)
class PushSolveResult:
    solved: bool
    moves: str | None
    final_board: tuple[tuple[str, ...], ...] | None
    pushes: tuple[Push, ...]
    nodes_expanded: int
    peak_closed_size: int
    elapsed_ms: float
    failure_reason: str | None
    strategy: str | None = None
    attempts: tuple[SearchAttempt, ...] = ()
    baseline_keystrokes: int | None = None
    quality_improved: bool = False
    quality_nodes_expanded: int = 0


@dataclass(frozen=True)
class SearchStrategyConfig:
    name: str
    kind: str = "weighted"
    weight: float = 2.0
    g_weight: float = 1.0
    bias_scale: float = 1.0
    use_macros: bool = False
    policy_weight: float = 0.0
    committed_plan: LinePlan | None = None
    commitment_bias_scale: float = 0.0
    relevance_filter: bool = False
    beam_width: int = 128
    beam_max_depth: int = 90
    beam_plan_limit: int = 24
    beam_novelty_per_signature: int = 2
    beam_escape_band: int = 8
    beam_restart_widths: tuple[int, ...] = (48, 96, 160, 256)
    beam_restart_depths: tuple[int, ...] = (60, 90, 120, 160)


@dataclass
class SearchContext:
    static: StaticBoard
    start_state: State
    initial_player: int
    target_access_penalty: float
    region_cache: dict[State, frozenset[int]]
    h_cache: dict[State, float]
    plan_cache: dict[State, LinePlan | None]
    top_plan_cache: dict[State, tuple[LinePlan, ...]]
    o_push_count_cache: dict[State, int]
    deadlock_cache: dict[tuple[frozenset[int], frozenset[int], int | None], bool]
    successor_cache: dict[State, tuple[tuple[Push, State, frozenset[int], float, float], ...]]
    policy_score_cache: dict[tuple[State, tuple[Push, ...], State, bool, bool], float]


StrategyChild = tuple[tuple[Push, ...], State, frozenset[int], float, float, int, float]
