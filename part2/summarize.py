"""Summarize P2 pilot results from the saved CSV runs — no notebook session needed.

Usage:
    python3 part2/summarize.py [results_root]

Reads every p2_scores.csv under the given root (default: part2/results/pilot),
groups rows by run folder and by system name, and prints one line per system
with mean and standard deviation of macro-F1 across the saved seed runs.
"""
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def read_scores_csv(path):
    """Load one p2_scores.csv into a list of row dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_by_system(rows):
    """Group score rows by system name within one run folder."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["name"]].append(row)
    return grouped


def system_stats(system_rows):
    """Aggregate one system's saved runs into mean F1 and standard deviation."""
    scores = [float(r["macro_f1"]) for r in system_rows]
    n = len(scores)
    if n == 0:
        return 0, math.nan, math.nan
    sd = statistics.stdev(scores) if n > 1 else 0.0
    return n, statistics.mean(scores), sd


def format_system_line(system, stats, per_run_rows):
    """Format one system's aggregate line, including its per-seed runs."""
    n, mean, sd = stats
    runs_txt = "  ".join(f"s{r['seed']}={float(r['macro_f1']):.4f}"
                         for r in sorted(per_run_rows, key=lambda x: int(x["seed"])))
    return f"{system:<16} runs={n}  mean={mean:.4f}  sd={sd:.4f}   [{runs_txt}]"


def summarize_run_dir(scores_path):
    """Print the per-system table for one run folder's p2_scores.csv."""
    rows = read_scores_csv(scores_path)
    grouped = group_by_system(rows)
    print(f"\n=== {scores_path.parent.name} ===")
    for system in sorted(grouped):
        per_run = grouped[system]
        stats = system_stats(per_run)
        print(format_system_line(system, stats, per_run))


def summarize_results_root(root):
    """Walk every run folder under the root and print its summary table."""
    for scores_path in sorted(root.rglob("p2_scores.csv")):
        summarize_run_dir(scores_path)


if __name__ == "__main__":
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results" / "pilot"
    if arg.is_file() and arg.name == "p2_scores.csv":
        summarize_run_dir(arg)
    else:
        summarize_results_root(arg)
