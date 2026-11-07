"""
Builds results/REPORT.md from the evaluation files results/evaluation_*.json.

Usage:
    python scripts/report.py            # (re)write results/REPORT.md
    python scripts/report.py --check    # exit 1 if REPORT.md is out of date (CI)
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.reporting import load_results, render_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Markdown report of the evaluation results.")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--output", type=Path, default=None, help="Default: <results-dir>/REPORT.md.")
    parser.add_argument("--check", action="store_true",
                        help="Do not write: exit 1 if the report is missing or out of date.")
    args = parser.parse_args()
    output = args.output or args.results_dir / "REPORT.md"

    results = load_results(args.results_dir)
    if not results:
        print(f"No evaluation_*.json in {args.results_dir}")
        return 1
    report = render_report(results)

    if args.check:
        current = output.read_text(encoding="utf-8") if output.exists() else None
        if current != report:
            print(f"{output} is out of date: run python scripts/report.py")
            return 1
        print(f"{output} is up to date ({len(results)} results).")
        return 0

    output.write_text(report, encoding="utf-8", newline="\n")
    print(f"Wrote {output} ({len(results)} results).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
