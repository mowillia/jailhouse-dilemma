import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay a Prisoner's Dilemma experiment as a grid animation from results.json",
    )
    p.add_argument("results_path", type=str, help="Path to results.json (e.g., data/run_YYYYMMDD_HHMMSS/results.json)")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--experiment-id", type=str, default=None, help="Experiment ID to replay (preferred)")
    sel.add_argument("--experiment-name", type=str, default=None, help="Experiment name to replay")
    p.add_argument("--simulation-index", type=int, default=0, help="Index of simulation to replay (default: 0)")
    p.add_argument("--interval", type=int, default=300, help="Frame interval in ms (default: 300)")
    p.add_argument("--cols", type=int, default=None, help="Number of grid columns to use (default: sqrt(N))")
    p.add_argument("--title", type=str, default=None, help="Optional title displayed at the top of the animation")
    return p.parse_args()


def load_results(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def select_experiment(results: Dict, exp_id: Optional[str], exp_name: Optional[str]) -> Dict:
    experiments = results.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments found in results.json")
    if exp_id is not None:
        for exp in experiments:
            if exp.get("experiment_id") == exp_id:
                return exp
        raise ValueError(f"Experiment with id '{exp_id}' not found")
    if exp_name is not None:
        for exp in experiments:
            if exp.get("name") == exp_name:
                return exp
        raise ValueError(f"Experiment with name '{exp_name}' not found")
    # default: if exactly one experiment, take it
    if len(experiments) == 1:
        return experiments[0]
    raise ValueError("Multiple experiments found. Specify --experiment-id or --experiment-name.")


def extract_agent_actions(sim: Dict) -> Tuple[List[str], List[List[Optional[bool]]]]:
    """
    Returns (agent_ids, actions_by_agent) where actions are booleans per round:
    True=cooperate, False=defect, None=missing.
    """
    detailed = sim.get("detailed_results", {})
    agent_histories = detailed.get("agent_histories", [])
    if not agent_histories:
        raise ValueError("No agent_histories found in simulation detailed_results")

    # Preserve order as listed; alternatively, sort by numeric suffix
    agent_ids: List[str] = [a.get("agent_id", f"Agent-{i+1}") for i, a in enumerate(agent_histories)]

    # Determine max rounds across agents
    max_rounds = 0
    per_agent_rounds: List[List[Dict]] = []
    for a in agent_histories:
        rounds = a.get("rounds", [])
        per_agent_rounds.append(rounds)
        if rounds:
            max_rounds = max(max_rounds, max(r.get("round", 0) for r in rounds))

    # Build boolean action arrays per agent of length max_rounds
    actions_by_agent: List[List[Optional[bool]]] = []
    for rounds in per_agent_rounds:
        # Map round->action for quick lookup
        action_map = {r.get("round"): (r.get("self_action") == "cooperate") for r in rounds}
        series: List[Optional[bool]] = []
        last: Optional[bool] = None
        for i in range(1, max_rounds + 1):
            val = action_map.get(i, last)
            series.append(val)
            if val is not None:
                last = val
        actions_by_agent.append(series)

    return agent_ids, actions_by_agent


def compute_grid_dims(n: int, cols: Optional[int]) -> Tuple[int, int]:
    if cols is None:
        c = int(math.ceil(math.sqrt(n)))
    else:
        c = max(1, cols)
    r = int(math.ceil(n / c))
    return r, c


def build_frame_matrix(actions_by_agent: List[List[Optional[bool]]], t: int, rows: int, cols: int) -> np.ndarray:
    n = len(actions_by_agent)
    mat = np.full((rows, cols), np.nan, dtype=float)
    for idx in range(n):
        r = idx // cols
        c = idx % cols
        if r >= rows:
            break
        a = actions_by_agent[idx][t]
        if a is None:
            # default to previous or defect-like
            val = 0.0
        else:
            val = 1.0 if a else 0.0
        mat[r, c] = val
    return mat


def main() -> None:
    args = parse_args()
    results = load_results(args.results_path)
    exp = select_experiment(results, args.experiment_id, args.experiment_name)

    sims = exp.get("individual_simulations", [])
    if not sims:
        raise ValueError("No individual_simulations found in experiment")
    if args.simulation_index < 0 or args.simulation_index >= len(sims):
        raise IndexError(f"simulation_index out of range (0..{len(sims)-1})")

    sim = sims[args.simulation_index]
    agent_ids, actions_by_agent = extract_agent_actions(sim)
    max_rounds = len(actions_by_agent[0]) if actions_by_agent else 0

    rows, cols = compute_grid_dims(len(agent_ids), args.cols)

    # Set up figure
    fig, ax = plt.subplots(figsize=(6, 6))
    # Reserve more space at the top for title + legend
    fig.subplots_adjust(right=0.95, left=0.08, bottom=0.22, top=0.78)
    if args.title:
        fig.suptitle(args.title, fontsize=20, y=0.98)
    colors = ["lightcoral", "cornflowerblue"]  # 0=defect, 1=cooperate
    cmap = mcolors.ListedColormap(colors)
    im = ax.imshow(np.zeros((rows, cols)), cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])

    title_base = f"{exp.get('name', exp.get('experiment_id', 'Experiment'))} — Sim {args.simulation_index}"
    # Overlay round counter
    step_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        color="white",
        fontsize=12,
        bbox=dict(facecolor='black', alpha=0.35, pad=3),
        va='top',
    )

    # Legend: Blue = Cooperate, Red = Defect
    legend_handles = [
        Patch(facecolor='cornflowerblue', edgecolor='none', label='Cooperate (blue)'),
        Patch(facecolor='lightcoral', edgecolor='none', label='Defect (red)'),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
        fontsize=15,
        framealpha=0.85,
        borderaxespad=0.5,
    )

    # Timeline axis at the bottom: align width with the grid axis
    ax_pos = ax.get_position()  # in figure fraction (x0, y0, x1, y1)
    # Place the timeline just below the grid, with same width
    timeline_height = 0.08
    timeline_bottom = max(ax_pos.y0 - (timeline_height + 0.02), 0.03)
    timeline_ax = fig.add_axes([ax_pos.x0, timeline_bottom, ax_pos.width, timeline_height])
    timeline_ax.set_xlim(1, max_rounds)
    timeline_ax.set_ylim(0, 1)
    timeline_ax.axhline(0.5, color='gray', lw=2, alpha=0.6)
    time_marker = timeline_ax.axvline(1, color='black', lw=4)
    timeline_ax.set_yticks([])
    timeline_ax.set_xlabel('Time (round)', fontsize=15)

    def update(frame: int):
        # frame goes 0..max_rounds-1
        mat = build_frame_matrix(actions_by_agent, frame, rows, cols)
        # Mask NaNs by setting them to 0 and overlaying a hatch or leave as is; simplest: set NaNs to 0 and alpha
        im.set_array(np.nan_to_num(mat, nan=0.0))
        # Title with cooperation count
        valid = ~np.isnan(mat)
        coop = np.nansum(mat)
        total = int(valid.sum())
        ax.set_title(f"{title_base} — Round {frame+1}/{max_rounds}  (coop: {int(coop)}/{total})")
        step_text.set_text(f"Step {frame + 1}")
        # Update timeline marker
        # axvline expects a 2-point x sequence to match its ydata [0, 1]
        x = frame + 1
        time_marker.set_xdata([x, x])
        return [im, step_text, legend, time_marker]

    # Use blit=False for robustness with multiple axes and static legend
    anim = FuncAnimation(fig, update, frames=max_rounds, interval=args.interval, blit=False)
    plt.show()


if __name__ == "__main__":
    main()


