"""
experiment_autocorr.py

Utilities to compute the average action auto-correlation per experiment
from a Jailhouse Dilemma results.json file.

Definition used:
  <Auto-correlation> = (1 / ((N_R - 1) * N_A)) * sum_{alpha=1..N_A} sum_{i=1..N_R-1} sigma_{alpha,i} * sigma_{alpha,i+1},
where sigma = +1 for cooperate, -1 for defect.
"""
from __future__ import annotations

import os
import sys

import json
from typing import Dict, Iterable, List, Optional, Tuple


def _action_to_sigma(action: str) -> int:
    a = (action or "").strip().lower()
    return 1 if a == "cooperate" else -1


def _compute_agent_autocorr(actions: List[str]) -> Tuple[float, int]:
    """
    Compute sum of consecutive products and number of valid pairs for one agent.
    Returns (sum_products, num_pairs).
    """
    if not actions or len(actions) < 2:
        return 0.0, 0

    sigmas = [_action_to_sigma(a) for a in actions]
    sum_products = 0.0
    num_pairs = 0
    for i in range(len(sigmas) - 1):
        sum_products += sigmas[i] * sigmas[i + 1]
        num_pairs += 1
    return sum_products, num_pairs


def _compute_sim_autocorr(sim: Dict) -> Tuple[float, int]:
    """
    Compute (sum_products, num_pairs) accrued over all agents in one simulation.
    sim is a dict from results["experiments"][k]["individual_simulations"][...]
    """
    detailed = sim.get("detailed_results", {})
    agent_histories = detailed.get("agent_histories", [])
    total_sum = 0.0
    total_pairs = 0
    for agent in agent_histories:
        rounds = agent.get("rounds", [])
        actions = [r.get("self_action", "defect") for r in rounds]
        s, n = _compute_agent_autocorr(actions)
        total_sum += s
        total_pairs += n
    return total_sum, total_pairs


def compute_experiment_autocorr(exp: Dict) -> Optional[float]:
    """
    Compute the experiment-level average auto-correlation across all simulations
    (weighted by number of pairs), returning None if no valid pairs exist.

    exp is the experiment dict in results["experiments"].
    """
    sims = exp.get("individual_simulations", [])
    grand_sum = 0.0
    grand_pairs = 0
    for sim in sims:
        s, n = _compute_sim_autocorr(sim)
        grand_sum += s
        grand_pairs += n
    if grand_pairs == 0:
        return None
    return grand_sum / grand_pairs


def compute_all_experiments_autocorr(results: Dict) -> List[Tuple[str, Optional[float]]]:
    """
    Returns a list of (experiment_name, autocorr or None) across all experiments
    in the given results.json dictionary.
    """
    out: List[Tuple[str, Optional[float]]] = []
    for exp in results.get("experiments", []):
        name = exp.get("name", exp.get("experiment_id", "unknown"))
        ac = compute_experiment_autocorr(exp)
        out.append((name, ac))
    return out


def load_results(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


"""
experiment_autocorr_tabulate.py

CLI to print an ASCII table of per-experiment average auto-correlation
from a Jailhouse Dilemma results.json file.
"""



def _render_ascii_table(rows: List[Tuple[str, Optional[float]]]) -> str:
    headers = ("Experiment", "Avg Auto-correlation")

    name_width = max(len(headers[0]), *(len(name) for name, _ in rows)) if rows else len(headers[0])
    # Format values as .3f or "NA"
    def _fmt(val: Optional[float]) -> str:
        return f"{val:.3f}" if val is not None else "NA"

    rate_width = max(len(headers[1]), * (len(_fmt(v)) for _, v in rows)) if rows else len(headers[1])

    sep = "+-" + ("-" * name_width) + "-+-" + ("-" * rate_width) + "-+"
    lines = [
        sep,
        f"| {headers[0]:<{name_width}} | {headers[1]:>{rate_width}} |",
        sep,
    ]

    for name, val in rows:
        lines.append(f"| {name:<{name_width}} | {_fmt(val):>{rate_width}} |")

    lines.append(sep)
    return "\n".join(lines)


def main() -> None:
    default_path = os.path.join("data", "run_20250926_185644", "results.json")
    results_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        sys.exit(1)

    results = load_results(results_path)
    rows = compute_all_experiments_autocorr(results)
    table = _render_ascii_table(rows)
    print(table)


if __name__ == "__main__":
    main()

