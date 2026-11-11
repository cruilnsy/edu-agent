# scripts/evaluation/metrics.py
# Metric functions shared by every eval script. Pure functions, no I/O, so they are unit-tested.

import math
from collections.abc import Collection, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix as _confusion_matrix, f1_score


# ── Retrieval ──

def hit_at_k(retrieved: Sequence[str], gold: Collection[str], k: int) -> bool:
    """True if any gold chunk appears in the top-k retrieved chunks."""
    return any(chunk_id in gold for chunk_id in retrieved[:k])


def recall_at_k(cases: Sequence[tuple[Sequence[str], Collection[str]]], k: int) -> float:
    """Share of questions whose gold chunk is in the top k (the course's "Recall@K").

    Each case is (retrieved_ids, gold_ids).
    """
    if not cases:
        return math.nan
    return sum(hit_at_k(retrieved, gold, k) for retrieved, gold in cases) / len(cases)


# ── Classification ──

def accuracy(predicted: Sequence[str], gold: Sequence[str]) -> float:
    _check_same_length(predicted, gold)
    if not gold:
        return math.nan
    return sum(p == g for p, g in zip(predicted, gold)) / len(gold)


def macro_f1(predicted: Sequence[str], gold: Sequence[str]) -> float:
    """F1 averaged over labels with equal weight, so rare intents count as much as common ones."""
    _check_same_length(predicted, gold)
    return float(f1_score(gold, predicted, average="macro", zero_division=0))


def confusion(predicted: Sequence[str], gold: Sequence[str], labels: Sequence[str]) -> list[list[int]]:
    """Rows are gold labels, columns are predicted labels, both in `labels` order."""
    _check_same_length(predicted, gold)
    return _confusion_matrix(gold, predicted, labels=list(labels)).tolist()


# ── Scores compared with a human baseline ──

def mae(predicted: Sequence[float], gold: Sequence[float]) -> float:
    """Mean absolute error."""
    _check_same_length(predicted, gold)
    if not gold:
        return math.nan
    return float(np.mean(np.abs(np.asarray(predicted, float) - np.asarray(gold, float))))


def to_level(score: float, full_score: float, cuts: tuple[float, float] = (0.6, 0.85)) -> str:
    """Bucket a score into low / mid / high by its share of the full score."""
    ratio = score / full_score if full_score else 0.0
    if ratio < cuts[0]:
        return "low"
    if ratio < cuts[1]:
        return "mid"
    return "high"


def cohen_kappa(levels_a: Sequence[str], levels_b: Sequence[str]) -> float:
    """Agreement between two raters beyond chance. NaN when it is undefined (one level only)."""
    _check_same_length(levels_a, levels_b)
    if len(set(levels_a) | set(levels_b)) < 2:
        return math.nan
    return float(cohen_kappa_score(levels_a, levels_b))


def ratio(numerator: float, denominator: float) -> float:
    """Safe division used for recall-style metrics (e.g. matched issues / expert issues)."""
    return numerator / denominator if denominator else math.nan


# ── System health ──

def percentile(values: Sequence[float], q: float) -> float:
    """q-th percentile, e.g. q=95 for P95 latency."""
    if not values:
        return math.nan
    return float(np.percentile(np.asarray(values, float), q))


def rate(flags: Sequence[bool]) -> float:
    """Share of True values, e.g. the fallback rate."""
    if not flags:
        return math.nan
    return sum(bool(f) for f in flags) / len(flags)


def spread(values: Sequence[float]) -> float:
    """Max minus min; used to check that repeated runs give consistent scores."""
    if not values:
        return math.nan
    return float(max(values) - min(values))


def _check_same_length(a: Sequence, b: Sequence) -> None:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
