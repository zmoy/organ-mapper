"""
Paired comparison of two evaluations (results/evaluation_*.json written by evaluate.py)
on the volumes they share: mean difference B - A, bootstrap 95 % confidence interval
and Wilcoxon signed-rank p-value, overall and per organ (see src/comparison.py).

Usage:
    python scripts/compare.py results/evaluation_A.json results/evaluation_B.json
    python scripts/compare.py A.json B.json --metrics dice --organs   # per-organ table too

The per-organ p-values are not corrected for the 13 tests: about one "significant"
organ in 13 is expected by chance at p < 0.05.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.comparison import HIGHER_IS_BETTER, METRICS, PairedResult, paired_comparison, per_volume_scores
from src.data.transforms import ORGAN_NAMES


def verdict(result: PairedResult, metric: str, alpha: float = 0.05) -> str:
    if result.p_value >= alpha or result.diff == 0:
        return "no significant difference"
    better = (result.diff > 0) == HIGHER_IS_BETTER[metric]
    return "B better" if better else "B worse"


def row(label: str, result: PairedResult | None, metric: str) -> str:
    if result is None:
        return f"{label:<26} not enough paired volumes"
    spec = ".1f" if metric == "hd95" else ".3f"
    return (f"{label:<26}{result.n:>4}{result.mean_a:>10{spec}}{result.mean_b:>10{spec}}{result.diff:>+10{spec}}"
            f"   [{result.ci_low:+{spec}}, {result.ci_high:+{spec}}]{result.p_value:>9.3f}   {verdict(result, metric)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired comparison of two evaluations (B - A).")
    parser.add_argument("a", type=Path, help="Baseline evaluation JSON (A).")
    parser.add_argument("b", type=Path, help="New evaluation JSON (B).")
    parser.add_argument("--metrics", nargs="+", default=list(METRICS), choices=METRICS)
    parser.add_argument("--organs", action="store_true", help="Also compare each organ.")
    args = parser.parse_args()

    data_a, data_b = (json.loads(p.read_text(encoding="utf-8")) for p in (args.a, args.b))
    print(f"A = {args.a.name}\nB = {args.b.name}")
    for metric in args.metrics:
        try:
            scores_a, scores_b = per_volume_scores(data_a, metric), per_volume_scores(data_b, metric)
        except KeyError as error:
            print(f"Error: {error.args[0]}")
            return 1
        print(f"\n{metric.upper()} ({'higher' if HIGHER_IS_BETTER[metric] else 'lower'} is better)")
        print(f"{'':<26}{'n':>4}{'A':>10}{'B':>10}{'B - A':>10}   {'95 % CI':<18}{'p':>7}")
        print(row("mean over organs", paired_comparison(scores_a, scores_b), metric))
        if args.organs:
            for organ, name in enumerate(ORGAN_NAMES, start=1):
                print(row(f"{organ} {name}", paired_comparison(scores_a, scores_b, organ), metric))
    print("\nn = paired volumes; CI = bootstrap of the volumes; p = Wilcoxon signed-rank (two-sided).")
    if args.organs:
        print("Per-organ p-values are not corrected for multiple comparisons.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
