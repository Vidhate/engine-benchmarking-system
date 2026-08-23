"""Binary classification metrics over a trace universe: P/R/F1 + hand-computed
Cohen's kappa. Shared by Scorer 1 (category x trace) and Scorer 2 (known-error x
trace) — both reduce to "which traces got this binary label".
"""

from __future__ import annotations

from typing import NamedTuple


class PRF1Kappa(NamedTuple):
    precision: float
    recall: float
    f1: float
    cohens_kappa: float
    support: int  # count of GT-positive traces (TP + FN)


def binary_prf1_kappa(
    gt_positive: set[str], pred_positive: set[str], universe: list[str]
) -> PRF1Kappa:
    """P/R/F1/kappa for one binary label over `universe` traces.

    Conventions for degenerate denominators (documented, not accidental):
    - precision = 0.0 when nothing was predicted positive.
    - recall = 0.0 when there are no actual positives.
    - f1 = 0.0 whenever precision + recall == 0.
    - kappa = 1.0 when expected agreement pe == 1 (both marginals degenerate to
      a single class, so po == pe == 1 trivially — chance-corrected agreement
      is undefined but "no better than chance" doesn't apply either).
    """
    universe_set = set(universe)
    n = len(universe_set)

    tp = len(gt_positive & pred_positive)
    fp = len(pred_positive - gt_positive)
    fn = len(gt_positive - pred_positive)
    tn = n - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    if n == 0:
        kappa = 1.0
    else:
        po = (tp + tn) / n
        gt_pos_rate = (tp + fn) / n
        pred_pos_rate = (tp + fp) / n
        pe = gt_pos_rate * pred_pos_rate + (1 - gt_pos_rate) * (1 - pred_pos_rate)
        kappa = 1.0 if pe == 1.0 else (po - pe) / (1 - pe)

    return PRF1Kappa(
        precision=precision,
        recall=recall,
        f1=f1,
        cohens_kappa=kappa,
        support=len(gt_positive),
    )
