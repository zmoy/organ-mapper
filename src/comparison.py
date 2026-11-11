"""
Paired comparison of two evaluations on the same volumes (scripts/compare.py).

Fold means of 6 volumes are too noisy to separate settings a few Dice points apart.
Pairing the volumes (same patient, two settings) removes the patient-to-patient
variability: the test is on the per-volume differences.
    - bootstrap 95 % confidence interval of the mean difference (volumes resampled);
    - Wilcoxon signed-rank test (no normality assumption).
Needs the "per_volume" key written by scripts/evaluate.py (evaluations made before it
was added must be run again).
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon

METRICS = ("dice", "hd95", "nsd")
# Direction of improvement of each metric
HIGHER_IS_BETTER = {"dice": True, "hd95": False, "nsd": True}


@dataclass
class PairedResult:
    """Comparison of B against A on the volumes both evaluated (B - A)."""

    n: int
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float
    p_value: float


def per_volume_scores(data: dict, metric: str) -> dict[str, np.ndarray]:
    """
    Args:
        data: Content of an evaluation JSON.
        metric: "dice", "hd95" or "nsd".

    Returns:
        {volume name: per-organ values (13,), NaN where undefined}.
    """
    if "per_volume" not in data:
        raise KeyError("No 'per_volume' in this evaluation: run scripts/evaluate.py again.")
    return {
        v["name"]: np.array([np.nan if x is None else x for x in v[metric]], dtype=float)
        for v in data["per_volume"]
    }


def paired_comparison(
    scores_a: dict[str, np.ndarray],
    scores_b: dict[str, np.ndarray],
    organ: int | None = None,
    n_bootstrap: int = 10000,
    seed: int = 0,
) -> PairedResult | None:
    """
    Args:
        scores_a: Per-volume scores of the baseline (see per_volume_scores).
        scores_b: Per-volume scores of the new setting.
        organ: Organ label (1-13), or None for the mean over the organs of each volume.
        n_bootstrap: Bootstrap resamples of the volumes.
        seed: Seed of the bootstrap.

    Returns:
        The comparison, or None when fewer than 2 volumes have a value in both.
    """
    pairs = []
    for name in sorted(scores_a.keys() & scores_b.keys()):
        a, b = scores_a[name], scores_b[name]
        if organ is None:
            both = ~np.isnan(a) & ~np.isnan(b)  # same organs on both sides
            if both.any():
                pairs.append((a[both].mean(), b[both].mean()))
        elif not (np.isnan(a[organ - 1]) or np.isnan(b[organ - 1])):
            pairs.append((a[organ - 1], b[organ - 1]))
    if len(pairs) < 2:
        return None
    a, b = np.array(pairs).T
    diffs = b - a
    rng = np.random.default_rng(seed)
    boot = diffs[rng.integers(0, len(diffs), size=(n_bootstrap, len(diffs)))].mean(axis=1)
    p_value = 1.0 if np.allclose(diffs, 0) else float(wilcoxon(diffs).pvalue)
    return PairedResult(
        n=len(diffs), mean_a=float(a.mean()), mean_b=float(b.mean()), diff=float(diffs.mean()),
        ci_low=float(np.percentile(boot, 2.5)), ci_high=float(np.percentile(boot, 97.5)), p_value=p_value,
    )
