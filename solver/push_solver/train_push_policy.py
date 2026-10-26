"""Export V4 primitive decisions and fit an offline calibrated softmax policy.

Only the frozen corpus is accepted. Runtime artifacts contain feature weights,
not paths, board IDs or action lookup tables. The original identity determines
train/development/calibration/evaluation splits, including transformations.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from hashlib import sha256
from pathlib import Path

import numpy as np

from apps.api.puzzle_catalog import CatalogEntry
from solver.push_solver.benchmark_v4 import TARGETS, corpus, provenance, read_rows
from solver.push_solver.models import Push
from solver.push_solver.push_policy import FEATURES, MLP_SCHEMA, SCHEMA, action_features
from solver.push_solver.train_v4_candidate import identity
from solver.push_solver.v4 import Context
from solver.push_solver.verify import verify_solution

DEVELOPMENT = frozenset(TARGETS) | {"20250928", "20260301", "20260802", "20260823"}


def consume(iterator):
    while True:
        try:
            next(iterator)
        except StopIteration as done:
            return done.value


def board_for(row):
    return CatalogEntry(
        date.fromisoformat(row["date"]),
        row["name"],
        row["width"],
        row["height"],
        row["puzzle"],
    ).board()


def split_identities(rows):
    groups = {r["id"]: identity(r) for r in rows}
    dev = {group for key, group in groups.items() if key in DEVELOPMENT}
    return {
        key: "development"
        if group in dev
        else "evaluation"
        if int(group[:8], 16) % 10 == 0
        else "calibration"
        if int(group[:8], 16) % 10 == 1
        else "train"
        for key, group in groups.items()
    }


def push_count(board, moves):
    width = len(board[0])
    occupied = {
        r * width + c
        for r, row in enumerate(board)
        for c, value in enumerate(row)
        if value in ("O", "X")
    }
    player = next(
        r * width + c
        for r, row in enumerate(board)
        for c, value in enumerate(row)
        if value == "U"
    )
    count = 0
    for move in moves:
        step = {"U": -width, "D": width, "L": -1, "R": 1}[move]
        nxt = player + step
        if nxt in occupied:
            occupied.remove(nxt)
            occupied.add(nxt + step)
            count += 1
        player = nxt
    return count


def select_paths(rows, paths):
    boards = {r["id"]: r for r in rows}
    selected = {}
    sources = {}
    for path in paths:
        sources[str(path)] = sha256(Path(path).read_bytes()).hexdigest()
        for row in read_rows(path):
            key = str(row.get("id") or row.get("date", "").replace("-", ""))
            original = boards.get(key)
            if (
                original is None
                or row.get("transformation")
                or row.get("board_sha256") != original["board_sha256"]
            ):
                continue
            moves = row.get("solution")
            if moves is None:
                moves = row.get("result", {}).get("moves")
            if moves is None:
                continue
            board = board_for(original)
            if not verify_solution(board, moves).ok:
                continue
            score = push_count(board, moves), len(moves), moves
            if key not in selected or score < selected[key][0]:
                selected[key] = score, moves
    return {key: value[1] for key, value in selected.items()}, sources


def export_board(job):
    row, moves, split = job
    ctx = Context(board_for(row), time.perf_counter() + 300, None, route_limit=3)
    consume(ctx.initialize())
    os, xs, player = ctx.os, ctx.xs, ctx.player
    decisions = []
    for move in moves:
        step = {"U": -ctx.cols, "D": ctx.cols, "L": -1, "R": 1}[move]
        nxt = player + step
        if (os | xs) & (1 << nxt):
            state = os, xs, ctx.reachable(player, os | xs)
            if ctx.goal(state)[0]:
                break  # walking to a reachable win ends search before this push
            expected = Push("O" if os & (1 << nxt) else "X", nxt, move)
            children = consume(ctx.children(state, False))
            labels = [
                i for i, child in enumerate(children) if child.pushes == (expected,)
            ]
            if len(labels) != 1:
                raise ValueError(
                    f"{row['id']}: verified teacher push absent from V4 successors"
                )
            features = consume(action_features(ctx, state, children))
            decisions.append((features, labels[0]))
            if expected.piece == "O":
                os = os ^ (1 << nxt) | (1 << (nxt + step))
            else:
                xs = xs ^ (1 << nxt) | (1 << (nxt + step))
        player = nxt
    return decisions, {
        "id": row["id"],
        "identity": identity(row),
        "split": split,
        "board_sha256": row["board_sha256"],
        "path_sha256": sha256(moves.encode()).hexdigest(),
        "path_pushes": push_count(board_for(row), moves),
        "decisions": len(decisions),
    }


def export_dataset(rows, paths, destination, workers=4, max_boards=None):
    source_provenance = provenance()
    selected, sources = select_paths(rows, paths)
    splits = split_identities(rows)
    jobs = [
        (row, selected[row["id"]], splits[row["id"]])
        for row in rows
        if row["id"] in selected
    ]
    if max_boards is not None:
        jobs = jobs[:max_boards]
    features, offsets, labels, board_indices, manifest = [], [0], [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (decisions, entry) in enumerate(pool.map(export_board, jobs)):
            for xs, gold in decisions:
                features.extend(xs)
                offsets.append(len(features))
                labels.append(gold)
                board_indices.append(index)
            manifest.append(entry)
            print(
                f"export {entry['id']} {entry['split']} decisions={len(decisions)}",
                flush=True,
            )
    if not features:
        raise ValueError("No complete verified training decisions")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        features=np.array(features, dtype=np.float64),
        offsets=np.array(offsets, dtype=np.int64),
        labels=np.array(labels, dtype=np.int64),
        boards=np.array(board_indices, dtype=np.int64),
    )
    report = {
        "schema": SCHEMA,
        "features": FEATURES,
        "sources": sources,
        "manifest": manifest,
        "provenance": source_provenance,
        "decision_count": len(labels),
        "action_count": len(features),
        "split_rule": "canonical original identity; mod10=0 evaluation, 1 calibration, otherwise train; 13 known hard/regression boards and matching identities are development",
        "claim": "Evaluation identities excluded from fitting and temperature calibration; all corpus boards were previously used for solver benchmarking. No unseen-board generalization claim.",
    }
    report["dataset_sha256"] = sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


def probabilities(x, mask, weights, temperature=1.0, mix=0.05):
    logits = (
        np.tanh(x @ weights["hidden_weights"] + weights["hidden_bias"])
        @ weights["output_weights"]
        if isinstance(weights, dict)
        else x @ weights
    )
    logits = np.where(mask, logits, -np.inf)
    exp = (
        np.exp(
            np.maximum(
                -745.0, (logits - np.max(logits, axis=1, keepdims=True)) / temperature
            )
        )
        * mask
    )
    softmax = exp / exp.sum(axis=1, keepdims=True)
    return (1 - mix) * softmax + mix * mask / mask.sum(axis=1, keepdims=True), softmax


def metrics(x, mask, y, board, weights, temperature, mix):
    if len(y) == 0:
        return {"decisions": 0}
    p, _ = probabilities(x, mask, weights, temperature, mix)
    correct = p.argmax(axis=1) == y
    confidence = p.max(axis=1)
    nll = -np.log(p[np.arange(len(y)), y])
    unique = np.unique(board)
    board_nll = float(np.mean([nll[board == key].mean() for key in unique]))
    ece = 0.0
    for lo in np.linspace(0, 0.9, 10):
        in_bin = (confidence >= lo) & (
            confidence < lo + 0.1 + 1e-12 if lo >= 0.9 else confidence < lo + 0.1
        )
        if in_bin.any():
            ece += float(
                in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
            )
    return {
        "decisions": len(y),
        "boards": len(unique),
        "top1_accuracy": float(correct.mean()),
        "mean_teacher_nll": float(nll.mean()),
        "board_mean_teacher_nll": board_nll,
        "ece_10_bins": ece,
    }


def fit(
    dataset, destination, *, epochs=80, seed=20261026, mix=0.05, architecture="linear"
):
    if architecture not in ("linear", "mlp") or epochs < 1 or not 1e-6 <= mix <= 1:
        raise ValueError("Invalid policy training controls")
    if destination.resolve().parent == Path(__file__).resolve().parent:
        raise ValueError(
            "Keep experimental policies outside the production model directory"
        )
    manifest = json.loads(dataset.with_suffix(".manifest.json").read_text())
    if (
        manifest["dataset_sha256"] != sha256(dataset.read_bytes()).hexdigest()
        or tuple(manifest["features"]) != FEATURES
    ):
        raise ValueError("Dataset identity/schema mismatch")
    with np.load(dataset, allow_pickle=False) as data:
        features, offsets, labels, boards = [
            data[key] for key in ("features", "offsets", "labels", "boards")
        ]
    counts = np.diff(offsets)
    choice = np.flatnonzero(counts > 1)
    x = np.zeros((len(choice), int(counts.max()), len(FEATURES)))
    mask = np.zeros(x.shape[:2], dtype=bool)
    for i, key in enumerate(choice):
        x[i, : counts[key]] = features[offsets[key] : offsets[key + 1]]
        mask[i, : counts[key]] = True
    y, board = labels[choice], boards[choice]
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite training features")
    split = np.array([manifest["manifest"][index]["split"] for index in board])
    train = np.flatnonzero(np.isin(split, ("train", "development")))
    calibration = np.flatnonzero(split == "calibration")
    evaluation = np.flatnonzero(split == "evaluation")
    if not len(train) or not len(calibration) or not len(evaluation):
        raise ValueError(
            "Training, calibration and evaluation need separate identities"
        )
    group_counts = np.bincount(board)
    sample_weight = 1.0 / group_counts[board]
    weights = np.zeros(len(FEATURES))
    momentum, variance = np.zeros_like(weights), np.zeros_like(weights)
    rng = np.random.default_rng(seed)
    if architecture == "mlp":
        weights = {
            "hidden_weights": rng.normal(
                0, 1 / math.sqrt(len(FEATURES)), (len(FEATURES), 32)
            ),
            "hidden_bias": np.zeros(32),
            "output_weights": rng.normal(0, 0.1, 32),
        }
        momentum = {k: np.zeros_like(v) for k, v in weights.items()}
        variance = {k: np.zeros_like(v) for k, v in weights.items()}
    history = []
    update = 0
    for epoch in range(epochs):
        for start in range(0, len(train), 256):
            if start == 0:
                order = rng.permutation(train)
            idx = order[start : start + 256]
            p, softmax = probabilities(x[idx], mask[idx], weights, mix=mix)
            # Derivative of log((1-epsilon)*softmax_gold + epsilon/K).
            soft_gold = softmax[np.arange(len(idx)), y[idx]]
            factor = (1 - mix) * soft_gold / p[np.arange(len(idx)), y[idx]]
            delta = softmax.copy()
            delta[np.arange(len(idx)), y[idx]] -= 1
            scale = sample_weight[idx] / sample_weight[idx].sum()
            delta *= (factor * scale)[:, None]
            update += 1
            if architecture == "mlp":
                hidden = np.tanh(
                    x[idx] @ weights["hidden_weights"] + weights["hidden_bias"]
                )
                back = delta[..., None] * weights["output_weights"] * (1 - hidden**2)
                gradient = {
                    "hidden_weights": np.einsum("bkf,bkh->fh", x[idx], back),
                    "hidden_bias": back.sum(axis=(0, 1)),
                    "output_weights": np.einsum("bk,bkh->h", delta, hidden),
                }
                norm = math.sqrt(sum(float(np.sum(v * v)) for v in gradient.values()))
                for key in weights:
                    grad = (
                        gradient[key] * min(1, 10 / max(norm, 1e-12))
                        + 1e-4 * weights[key]
                    )
                    momentum[key] = 0.9 * momentum[key] + 0.1 * grad
                    variance[key] = 0.999 * variance[key] + 0.001 * grad**2
                    weights[key] -= (
                        0.01
                        * (momentum[key] / (1 - 0.9**update))
                        / (np.sqrt(variance[key] / (1 - 0.999**update)) + 1e-8)
                    )
            else:
                gradient = np.einsum("bk,bkf->f", delta, x[idx]) + 1e-4 * weights
                norm = np.linalg.norm(gradient)
                if norm > 10:
                    gradient *= 10 / norm
                momentum = 0.9 * momentum + 0.1 * gradient
                variance = 0.999 * variance + 0.001 * gradient**2
                weights -= (
                    0.03
                    * (momentum / (1 - 0.9**update))
                    / (np.sqrt(variance / (1 - 0.999**update)) + 1e-8)
                )
        if epoch % 10 == 0 or epoch == epochs - 1:
            row = metrics(
                x[train], mask[train], y[train], board[train], weights, 1.0, mix
            )
            history.append({"epoch": epoch + 1, **row})
            print(
                f"fit epoch={epoch + 1} train_nll={row['board_mean_teacher_nll']:.4f}",
                flush=True,
            )
    grid = sorted({float(v) for v in np.geomspace(0.25, 4.0, 61)} | {1.0})
    calibration_grid = [
        {
            "temperature": t,
            **metrics(
                x[calibration],
                mask[calibration],
                y[calibration],
                board[calibration],
                weights,
                t,
                mix,
            ),
        }
        for t in grid
    ]
    temperature = min(
        calibration_grid,
        key=lambda row: (
            row["board_mean_teacher_nll"],
            abs(math.log(row["temperature"])),
        ),
    )["temperature"]
    payload = {
        "schema": SCHEMA,
        "name": "v4-phs-linear-policy-v1",
        "temperature": temperature,
        "uniform_mix": mix,
    }
    if architecture == "mlp":
        payload.update(
            schema=MLP_SCHEMA,
            name="v4-phs-mlp-policy-v1",
            **{key: value.tolist() for key, value in weights.items()},
        )
    else:
        payload["weights"] = dict(zip(FEATURES, weights.tolist(), strict=True))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    report = {
        "dataset_sha256": manifest["dataset_sha256"],
        "model_sha256": sha256(destination.read_bytes()).hexdigest(),
        "epochs": epochs,
        "seed": seed,
        "architecture": architecture,
        "uniform_mix": mix,
        "temperature": temperature,
        "forced_decisions_excluded_from_fit_and_metrics": int((counts == 1).sum()),
        "history": history,
        "calibration_grid": calibration_grid,
        "metrics": {
            key: {
                "uncalibrated": metrics(
                    x[idx], mask[idx], y[idx], board[idx], weights, 1, mix
                ),
                "calibrated": metrics(
                    x[idx], mask[idx], y[idx], board[idx], weights, temperature, mix
                ),
            }
            for key, idx in [
                ("fit", train),
                ("calibration", calibration),
                ("evaluation", evaluation),
            ]
        },
        "claim": manifest["claim"],
    }
    destination.with_suffix(".training.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report["metrics"]["evaluation"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--paths", nargs="+", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--uniform-mix", type=float, default=0.05)
    parser.add_argument("--max-boards", type=int)
    parser.add_argument("--architecture", choices=("linear", "mlp"), default="linear")
    args = parser.parse_args()
    if args.output and args.output.resolve().parent == Path(__file__).resolve().parent:
        parser.error(
            "Keep experimental policies outside the production model directory"
        )
    if args.paths:
        export_dataset(
            corpus(), args.paths, args.dataset, args.workers, args.max_boards
        )
    if args.output:
        fit(
            args.dataset,
            args.output,
            epochs=args.epochs,
            seed=args.seed,
            mix=args.uniform_mix,
            architecture=args.architecture,
        )


if __name__ == "__main__":
    main()
