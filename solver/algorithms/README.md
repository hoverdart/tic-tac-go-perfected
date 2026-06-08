# Solver Algorithms

This note tracks how each Tic Tac Go solver works so future changes have a
baseline to compare against.

## Game Model

The board is a rectangular grid with five cell values:

- `""`: empty square
- `"U"`: player piece
- `"O"`: useful piece
- `"X"`: blocker piece
- `"B"`: fixed wall

A board is solved when any horizontal or vertical run of three squares contains
only useful pieces, where `"U"` counts as useful. A board is lost when any run of
three contains only `"X"`.

## Legacy Solver

File: `solver/randomPythonFiles/superTicTacGoSolver.py`

The legacy solver is the original production solver. It is called `bfs` by the
API, but the current implementation is closer to A* over compressed states than
plain breadth-first search.

How it works:

1. Normalize the board into an immutable tuple-of-tuples.
2. Find every square the player can walk to without pushing.
3. Collapse each walking path plus one push into a single search edge.
4. Use a heap priority queue ordered by `moves_so_far + heuristic(board)`.
5. Prune boards that are already lost, have no possible win line, or appear
   soft-locked.
6. Store the full move string in each queued heap entry.

Strengths:

- Reliable and battle-tested.
- Usually finds short solutions.
- The compressed push-state search avoids wasting time on every walking step.

Limits:

- Tuple board copies are relatively expensive.
- Each queued state carries a growing move string.
- The heuristic is useful but recomputed often.

## Optimized Solver

File: `solver/optimized_solver.py`

The optimized solver keeps the same public contract as the legacy solver:
`solve(board, progress_every=0, max_states=None)` returns
`(moves, final_board, states_checked)`.

How it improves the search:

1. Flatten the board into a compact string key.
2. Cache board geometry: neighbors, push landing squares, and win lines.
3. Cache line-score heuristics for repeated states.
4. Keep parent pointers instead of copying full move strings into each heap
   entry.
5. Preserve the legacy idea of collapsing walk paths plus pushes into one edge.
6. Order generated moves by whether they improve the best win-line score and
   whether they push an `"O"`.
7. Replay and validate the returned move string before reporting success.

Modes:

- `hybrid`: weighted best-first search first, then continues looking for a
  shorter solution inside the remaining state budget.
- `fast`: returns the first validated solution.
- `exact`: uses cost-only priority. This is slower, but is useful when checking
  shortest-path behavior on small boards.

API selection:

```bash
SOLVER_IMPL=optimized
SOLVER_MODE=hybrid
```

The service falls back to the legacy solver if the optimized path fails before
the state budget is exhausted. Set `SOLVER_FALLBACK=none` to disable that.

## Training BFS

Files:

- `solver/gymnasium_register/BFStoTrainer.py`
- `solver/gymnasium_register/board_generator.py`
- `solver/gymnasium_register/rank_real_boards.py`

These BFS implementations support Gymnasium training, board generation, and
difficulty ranking. They expand one raw movement at a time instead of collapsing
walking paths into push edges. That makes them simpler for training data and
rank labels, but less efficient as a production solver.

Keep these implementations because they are part of the model-training path and
provide useful comparison data.

## DQN / Gymnasium Solver Path

Files:

- `solver/gymnasium_register/train.py`
- `solver/gymnasium_register/trainGrad8.py`
- `solver/randomPythonFiles/run_dqn_grad10_board.py`

The model path treats Tic Tac Go as a reinforcement-learning environment. A DQN
policy chooses one of four movement actions at each step. This is not currently
the production solver, but it is useful for learning strategies that may later
guide or replace heuristic search.

## Measuring Progress

Use the benchmark runner to compare production solvers:

```bash
python3 -m solver.benchmark_solvers --groups five six seven --limit 3
```

Track these fields:

- `solved`: whether a solution was found under the state cap
- `moves`: length of the returned move string
- `states`: compressed states checked
- `elapsed_ms`: wall-clock time

A solver improvement should normally preserve or reduce move length while
reducing elapsed time or states checked on the ranked board corpus.

## Future ML Solver Plan

The next ML attempt should not try to replace search immediately. The production
recommendation is **learned search guidance first**, with a direct-policy model
as a separate experiment.

### Why The Current Gymnasium Path Is Failing

The current Gymnasium/DQN setup learns raw button presses, while the production
solver wins by searching compressed states where each edge is "walk to a push
position, then push once." That mismatch makes the model learn the wrong level
of abstraction for improving A*.

Current pain points:

- Rewards are mostly terminal, so DQN receives weak guidance about which moves
  make the board easier to solve.
- The model is not trained to estimate remaining solve cost, rank candidate
  pushes, or improve the A* priority function.
- Injected BFS examples are too small and hardcoded to teach broad strategy.
- Raw movement policies can loop, waste steps, or solve one board family while
  failing to generalize to harder layouts.

### Track 1: Learned Search Guide

This is the production path. Keep A*/best-first search in charge of validity,
but train a tiny model to make it expand better states sooner.

Training data should come from offline expert traces generated by
`optimized_solver` and exact runs on ranked/generated boards. Each training row
should describe:

- the current compact board state
- every candidate compressed move from that state
- which child appears on the expert solution path
- remaining move cost or depth-to-solution
- baseline metrics such as states checked and elapsed time

Start with cheap, dependency-light features:

- current heuristic score
- child heuristic delta
- pushed piece type
- compressed segment length
- blocker danger near X lines
- O/U occupancy on candidate win lines
- Manhattan distance from useful pieces to viable win lines

The first model can be a linear ranker, shallow tree model, or generated Python
scoring function. Runtime should avoid heavy ML dependencies so it can fit in
Vercel's CPU/serverless constraints. Torch or ONNX can be tested later, but only
after bundle size and cold-start cost are measured.

Runtime behavior:

- Add a future `SOLVER_IMPL=ml-guided` mode.
- Combine normal `cost + heuristic` priority with the learned score.
- Never let the model prune states by itself.
- Keep replay validation before reporting success.
- Fall back to the pure optimized solver if the guided run fails or exceeds its
  budget.
- Let `SOLVER_MODE=exact` ignore model bias so exact comparisons stay clean.

### Track 2: Direct Policy

This is the research path. Train a model to output moves directly, then use it
as a fast first attempt before falling back to search.

Recommended sequence:

1. Behavior cloning from expert trajectories.
2. DAgger-style correction, where failed policy rollouts are solved by the
   expert and added back to the dataset.
3. Optional RL fine-tuning after supervised learning can already solve many
   boards.

This policy should be evaluated separately from the production search guide. It
is allowed to fail during experiments, but production should only use it as a
quick attempt followed by validated search fallback.

### Data And Evaluation

Split data by board identity and difficulty so validation measures
generalization instead of memorization. The ranked real-board corpus should be
the main benchmark, with generated boards used to increase training coverage.

Compare:

- legacy solver
- optimized `fast`, `hybrid`, and `exact`
- future ML-guided optimized solver
- direct policy plus fallback

Track:

- success rate
- move length
- states checked
- elapsed time
- model inference overhead
- memory use and model size

First acceptance target:

- same success rate as the optimized solver
- same or near-same move length
- at least 30-50% fewer states or lower elapsed time on hard boards
- runtime model small enough for Vercel CPU without heavy ML dependencies

### Useful References

- Stable-Baselines3 imitation learning:
  <https://stable-baselines3.readthedocs.io/en/master/guide/imitation.html>
- Ray RLlib for future scalable/offline RL:
  <https://docs.ray.io/en/latest/rllib/index.html>
- PyTorch compile for offline training/inference experiments:
  <https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler.html>
- ONNX Runtime as a possible later inference option:
  <https://onnxruntime.ai/docs/>
