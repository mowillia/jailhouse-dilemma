"""
Regenerate the Fig. 2 panels of the Jailhouse DiLLeMa paper from saved runs.

Reads only the saved results.json files (no LLM API calls) and writes
`gpt-4o-mini-experiment.png` and `deepseek-chat-experiment.png` into
`Jailhouse_Dilemma_Paper/`. Unlike the original figures, all seven
experimental conditions are plotted, including "Constant Random".

Usage (from anywhere):
    .venv/bin/python scripts/make_paper_figures.py
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # plt.show() in plots.py becomes a no-op

# llm_agent_definition reads API keys at import time; plotting never calls an API.
for key in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
    os.environ.setdefault(key, "unused-for-plotting")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "jailhouse" / "agents"))
sys.path.insert(0, str(REPO / "src" / "jailhouse" / "viz"))

import matplotlib.pyplot as plt  # noqa: E402
from plots import plot_selected_experiments  # noqa: E402

# With seven curves, loc="best" (set in llm_agent_definition) covers data.
# Move the legend below the axes just before plots.py saves the figure.
_savefig = plt.savefig


def _savefig_with_legend_below(*args, **kwargs):
    ax = plt.gca()
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=4,
        fontsize=18,
        frameon=False,
    )
    return _savefig(*args, **kwargs)


plt.savefig = _savefig_with_legend_below

RUNS = {
    "gpt-4o-mini": REPO / "data" / "run_20250926_185644",
    "deepseek-chat": REPO / "data" / "run_20250927_072359",
}
TEMPERATURE = 0.7
EXPERIMENT_IDS = [
    "same_as_you",
    "constant_defection",
    "constant_random",
    "us_conservative",
    "us_liberal",
    "human",
    "constant_cooperation",
]
OUT_DIR = REPO / "Jailhouse_Dilemma_Paper"


def main() -> None:
    for model_name, run_folder in RUNS.items():
        plot_selected_experiments(
            str(run_folder),
            EXPERIMENT_IDS,
            model_name=model_name,
            save_path=str(OUT_DIR / f"{model_name}-experiment.png"),
            title=(
                "Comparison of Agent Behaviors Under Different Beliefs with \n"
                f" model: {model_name} & temperature: {TEMPERATURE}"
            ),
            figsize=(15, 8),
        )


if __name__ == "__main__":
    main()
