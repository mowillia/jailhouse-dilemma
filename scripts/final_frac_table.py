# experiment_tabulate.py
# 2025-09-27

import json
import os
import sys
from typing import List, Tuple


def load_results(results_path: str) -> dict:
    with open(results_path, "r") as f:
        return json.load(f)


def compute_final_rates(results: dict) -> List[Tuple[str, float, float]]:
    """
    Returns list of tuples: (experiment_name, final_coop_rate, final_defect_rate)
    final rates are taken as the last value of mean_cooperation_over_time.
    """
    rows: List[Tuple[str, float, float]] = []
    for exp in results.get("experiments", []):
        name = exp.get("name", exp.get("experiment_id", "unknown"))
        coop_over_time = exp.get("mean_cooperation_over_time", [])
        if coop_over_time:
            final_coop = float(coop_over_time[-1])
        else:
            # If missing, default to 0.0
            final_coop = 0.0
        final_defect = 1.0 - final_coop
        rows.append((name, final_coop, final_defect))
    return rows


def render_ascii_table(name_and_coop: List[Tuple[str, float]]) -> str:
    """
    Render an ASCII table with columns: Experiment, Final Cooperation Rate
    """
    headers = ("Experiment", "Final Cooperation Rate")

    # Determine column widths
    name_width = max(len(headers[0]), *(len(name) for name, _ in name_and_coop))
    rate_width = max(len(headers[1]), *(len(f"{rate:.3f}") for _, rate in name_and_coop))

    sep = "+-" + ("-" * name_width) + "-+-" + ("-" * rate_width) + "-+"

    lines = [
        sep,
        f"| {headers[0]:<{name_width}} | {headers[1]:>{rate_width}} |",
        sep,
    ]

    for name, rate in name_and_coop:
        lines.append(f"| {name:<{name_width}} | {rate:>{rate_width}.3f} |")

    lines.append(sep)
    return "\n".join(lines)


def main() -> None:
    # Allow overriding the path via CLI; default to the path the user mentioned
    default_path = os.path.join(
        "data", "run_20250926_185644", "results.json"
    )
    results_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        sys.exit(1)

    results = load_results(results_path)
    rows = compute_final_rates(results)

    # Prepare table with name and final cooperation rate
    name_and_coop = [(name, coop) for name, coop, _ in rows]
    table = render_ascii_table(name_and_coop)
    print(table)

    # Also print overall final defection rate (average across experiments)
    if rows:
        avg_defection = sum(defect for _, _, defect in rows) / len(rows)
        print(f"\nOverall final defection rate (avg across experiments): {avg_defection:.3f}")


if __name__ == "__main__":
    main()