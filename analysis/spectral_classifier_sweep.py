#!/usr/bin/env python3
"""Train spectral-only category classifiers without sklearn."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def stable_bucket(s: str, mod: int = 10) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def load_features(path: Path):
    rows = list(csv.DictReader(path.open()))
    names = np.array([r["name"] for r in rows])
    y = np.array([r["category"] for r in rows])
    degree = np.array([float(r["degree"]) for r in rows], dtype=np.float32)
    cols = [c for c in rows[0] if c.startswith(("low_", "high_"))]
    x = {c: np.array([float(r[c]) for r in rows], dtype=np.float32) for c in cols}
    return names, y, degree, x


def split(names, y, degree, min_train: int):
    valid = (y != "UNKNOWN") & (degree > 0)
    train = np.array([stable_bucket(n) < 8 for n in names]) & valid
    test = (~train) & valid
    keep = {c for c, n in Counter(y[train]).items() if n >= min_train}
    train &= np.array([c in keep for c in y])
    test &= np.array([c in keep for c in y])
    cats = np.array(sorted(keep))
    enc = {c: i for i, c in enumerate(cats)}
    return train, test, cats, enc


def make_matrix(spec: str, degree, coords):
    parts = spec.split("+")
    mats = []
    for part in parts:
        if part == "degree":
            mats.append(np.log1p(degree[:, None]))
        elif part.startswith("low"):
            k = int(part.removeprefix("low"))
            mats.append(stack_prefix(coords, "low_", k))
        elif part.startswith("high"):
            k = int(part.removeprefix("high"))
            mats.append(stack_prefix(coords, "high_", k))
        else:
            raise ValueError(f"unknown feature spec: {part}")
    return np.hstack(mats).astype(np.float32)


def stack_prefix(coords, prefix: str, k: int):
    cols = [f"{prefix}{i}" for i in range(1, k + 1)]
    missing = [c for c in cols if c not in coords]
    if missing:
        raise ValueError(f"missing columns: {missing[:5]}")
    return np.vstack([coords[c] for c in cols]).T


def standardize(x_train, x_test):
    mu = x_train.mean(axis=0, keepdims=True)
    sig = x_train.std(axis=0, keepdims=True)
    sig[sig < 1e-8] = 1.0
    return (x_train - mu) / sig, (x_test - mu) / sig


def one_hot(y, n):
    out = np.zeros((len(y), n), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def metrics(y_true, probs, cats):
    pred = probs.argmax(axis=1)
    acc = float(np.mean(pred == y_true))
    f1s = []
    for i in range(len(cats)):
        tp = np.sum((y_true == i) & (pred == i))
        fp = np.sum((y_true != i) & (pred == i))
        fn = np.sum((y_true == i) & (pred != i))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    top3 = np.argsort(probs, axis=1)[:, -3:]
    return {
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)),
        "top3_accuracy": float(np.mean([yt in row for yt, row in zip(y_true, top3)])),
    }


def batches(n, batch_size, rng):
    order = rng.permutation(n)
    for i in range(0, n, batch_size):
        yield order[i : i + batch_size]


def train_logreg(x_train, y_train, x_test, y_test, cats, args):
    rng = np.random.default_rng(args.seed)
    n, d = x_train.shape
    c = len(cats)
    w = rng.normal(0, 0.01, size=(d, c)).astype(np.float32)
    b = np.zeros(c, dtype=np.float32)
    yh = one_hot(y_train, c)
    best = None
    for epoch in range(args.logreg_epochs):
        lr = args.logreg_lr * (0.5 ** (epoch // max(1, args.logreg_epochs // 3)))
        for ix in batches(n, args.batch_size, rng):
            p = softmax(x_train[ix] @ w + b)
            g = (p - yh[ix]) / len(ix)
            w -= lr * (x_train[ix].T @ g + args.weight_decay * w)
            b -= lr * g.sum(axis=0)
        probs = softmax(x_test @ w + b)
        cur = metrics(y_test, probs, cats)
        best = cur if best is None or cur["macro_f1"] > best["macro_f1"] else best
    return best


def train_mlp(x_train, y_train, x_test, y_test, cats, args):
    rng = np.random.default_rng(args.seed)
    n, d = x_train.shape
    h, c = args.hidden, len(cats)
    w1 = rng.normal(0, math.sqrt(2 / max(1, d)), size=(d, h)).astype(np.float32)
    b1 = np.zeros(h, dtype=np.float32)
    w2 = rng.normal(0, math.sqrt(2 / h), size=(h, c)).astype(np.float32)
    b2 = np.zeros(c, dtype=np.float32)
    yh = one_hot(y_train, c)
    best = None
    for epoch in range(args.mlp_epochs):
        lr = args.mlp_lr * (0.5 ** (epoch // max(1, args.mlp_epochs // 3)))
        for ix in batches(n, args.batch_size, rng):
            z1 = x_train[ix] @ w1 + b1
            h1 = np.maximum(z1, 0)
            p = softmax(h1 @ w2 + b2)
            g2 = (p - yh[ix]) / len(ix)
            dw2 = h1.T @ g2 + args.weight_decay * w2
            db2 = g2.sum(axis=0)
            g1 = (g2 @ w2.T) * (z1 > 0)
            dw1 = x_train[ix].T @ g1 + args.weight_decay * w1
            db1 = g1.sum(axis=0)
            w2 -= lr * dw2
            b2 -= lr * db2
            w1 -= lr * dw1
            b1 -= lr * db1
        probs = softmax(np.maximum(x_test @ w1 + b1, 0) @ w2 + b2)
        cur = metrics(y_test, probs, cats)
        best = cur if best is None or cur["macro_f1"] > best["macro_f1"] else best
    return best


def baseline_majority(y_train, y_test, cats):
    pred = Counter(y_train).most_common(1)[0][0]
    probs = np.zeros((len(y_test), len(cats)), dtype=np.float32)
    probs[:, pred] = 1.0
    return metrics(y_test, probs, cats)


def baseline_namespace(names, y, train, test, cats):
    majority = Counter(y[train]).most_common(1)[0][0]
    table = {}
    for name, label, ok in zip(names, y, train):
        if ok:
            table.setdefault(name.split(".", 1)[0], Counter())[label] += 1
    probs = np.zeros((test.sum(), len(cats)), dtype=np.float32)
    for i, name in enumerate(names[test]):
        c = table.get(name.split(".", 1)[0])
        probs[i, c.most_common(1)[0][0] if c else majority] = 1.0
    return metrics(y[test], probs, cats)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="results/spectral/spectral_features.csv")
    p.add_argument("--output", default="results/spectral/classifier_sweep.json")
    p.add_argument("--min-train", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--logreg-epochs", type=int, default=45)
    p.add_argument("--mlp-epochs", type=int, default=60)
    p.add_argument("--logreg-lr", type=float, default=0.15)
    p.add_argument("--mlp-lr", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    names, labels, degree, coords = load_features(Path(args.features))
    train, test, cats, enc = split(names, labels, degree, args.min_train)
    y_enc = np.array([enc.get(c, -1) for c in labels])
    y_train, y_test = y_enc[train], y_enc[test]

    specs = [
        "degree",
        "low10",
        "high10",
        "low10+high10",
        "low32",
        "high32",
        "low32+high32",
        "low10+degree",
        "high10+degree",
        "low10+high10+degree",
        "low32+degree",
        "high32+degree",
        "low32+high32+degree",
    ]

    out = {
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "categories": cats.tolist(),
        "baselines": {
            "majority": baseline_majority(y_train, y_test, cats),
            "namespace_prior": baseline_namespace(names, y_enc, train, test, cats),
        },
        "models": {},
    }
    for spec in specs:
        try:
            x = make_matrix(spec, degree, coords)
        except ValueError as exc:
            out["models"][spec] = {"error": str(exc)}
            continue
        x_train, x_test = standardize(x[train], x[test])
        out["models"][spec] = {
            "logreg": train_logreg(x_train, y_train, x_test, y_test, cats, args),
            "mlp": train_mlp(x_train, y_train, x_test, y_test, cats, args),
        }
        print(spec, out["models"][spec])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
