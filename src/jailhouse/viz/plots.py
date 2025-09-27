# %%
"""
Plotting utilities for analyzing experiment results.
"""

import json
import os
from typing import List, Optional, Dict, Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from llm_agent_definition import (
    MockAgent,
    plot_cooperation_fraction,
    plot_multiple_experiments
)

def load_results(run_folder: str) -> Dict[str, Any]:
    """Load results from a run folder's results.json file."""
    results_path = os.path.join(run_folder, "results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No results.json found in {run_folder}")
    
    with open(results_path, 'r') as f:
        return json.load(f)

def plot_selected_experiments(
    run_folder: str,
    experiment_ids: Sequence[str],
    *,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    model_name: Optional[str] = None,
    colors: Optional[List[str]] = None,
    figsize: tuple[int, int] = (12, 7),
    style: str = 'default',
) -> None:
    """
    Plot selected experiments from a run's results.json file.
    
    Args:
        run_folder: Path to the run folder containing results.json
        experiment_ids: List of experiment IDs to plot
        save_path: Optional path to save the plot
        title: Optional custom title for the plot
        model_name: Optional name of the model used for the plot
        colors: Optional list of colors for the experiment lines
        figsize: Figure size (width, height) in inches
        style: Matplotlib style to use (default: 'default')
    """
    # Load results
    results = load_results(run_folder)
    
    # Filter experiments
    selected_experiments = []
    for exp in results["experiments"]:
        if exp["experiment_id"] in experiment_ids:
            selected_experiments.append(exp)
    
    if not selected_experiments:
        print(f"No matching experiments found. Available IDs:")
        for exp in results["experiments"]:
            print(f"  - {exp['experiment_id']}: {exp['name']}")
        return
    
    # Set up plot
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use default color cycle if no colors provided
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Define marker styles
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot each selected experiment
    for idx, exp in enumerate(selected_experiments):
        # Create a mock agent with the mean cooperation rates
        mock_history = []
        print(f"\nProcessing experiment: {model_name} with model {model_name}")
        
        for i, rate in enumerate(exp["mean_cooperation_over_time"]):
            mock_history.append({
                "round": i + 1,
                "cooperation_rate": rate
            })
        mock_agents = [MockAgent(f"Average_{exp['name']}", mock_history)]
        
        # Plot this experiment
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        experiment_name = f"{exp['name']} (n={exp['n_repeats']})"
        plot_cooperation_fraction(
            mock_agents,
            total_rounds=exp["num_rounds"],
            experiment_name=experiment_name,
            ax=ax,
            color=color,
            linewidth=4.0,  # Increased line thickness
            marker=marker,  
            markersize=15,  # Increased marker size
            show=False,
        )
    
    # Set overall title if provided
    if title:
        ax.set_title(title, fontsize=30, pad=20)
    
    # Add statistics table
    stat_text = "Statistics:\n"
    for exp in selected_experiments:
        stat_text += f"\n{exp['name']}:\n"
        stat_text += f"  Mean cooperation rate: {exp['statistics']['cooperation_rates']['mean']:.1%}\n"
        stat_text += f"  Mean final score: {exp['statistics']['final_scores']['mean']:.1f}\n"
    
    # # Add text box with statistics
    # plt.gcf().text(
    #     1.02, 0.5,  # Position outside the plot
    #     stat_text,
    #     fontsize=10,
    #     verticalalignment='center',
    #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    # )
    
    # Adjust layout to prevent legend and stats cutoff
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_experiment_comparison(
    run_folders: Sequence[str],
    experiment_id: str,
    *,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    figsize: tuple[int, int] = (12, 7),
    style: str = 'default',
) -> None:
    """
    Compare the same experiment across different runs.
    
    Args:
        run_folders: List of run folder paths to compare
        experiment_id: ID of the experiment to compare
        save_path: Optional path to save the plot
        title: Optional custom title for the plot
        colors: Optional list of colors for the run lines
        figsize: Figure size (width, height) in inches
        style: Matplotlib style to use (default: 'default')
    """
    # Load all results
    all_results = []
    for folder in run_folders:
        try:
            results = load_results(folder)
            all_results.append((folder, results))
        except FileNotFoundError:
            print(f"Warning: No results.json found in {folder}")
            continue
    
    if not all_results:
        print("No valid results found to compare")
        return
    
    # Find the experiment in each run
    experiments = []
    for folder, results in all_results:
        for exp in results["experiments"]:
            if exp["experiment_id"] == experiment_id:
                exp["run_folder"] = folder  # Add folder for reference
                experiments.append(exp)
                break
    
    if not experiments:
        print(f"No experiments found with ID: {experiment_id}")
        print("Available IDs in first run:")
        for exp in all_results[0][1]["experiments"]:
            print(f"  - {exp['experiment_id']}: {exp['name']}")
        return
    
    # Plot comparison
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=figsize)
    
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Define marker styles
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for idx, exp in enumerate(experiments):
        # Create mock agent for plotting
        mock_history = []
        run_name = os.path.basename(exp["run_folder"])
        print(f"\nProcessing run: {run_name}")
        
        for i, rate in enumerate(exp["mean_cooperation_over_time"]):
            mock_history.append({
                "round": i + 1,
                "cooperation_rate": rate
            })
        mock_agents = [MockAgent(f"Run_{run_name}", mock_history)]
        
        # Plot this run
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        run_label = f"Run {run_name} with model {model_name}"
        plot_cooperation_fraction(
            mock_agents,
            total_rounds=exp["num_rounds"],
            experiment_name=run_label,
            ax=ax,
            color=color,
            linewidth=4.0,  # Increased line thickness
            marker=marker,
            markersize=15,  # Increased marker size
            show=False,
        )
    
    # Set title
    if title:
        ax.set_title(title, fontsize=20, pad=20)
    else:
        ax.set_title(f"Comparison of {experiments[0]['name']} Across Runs with model {model_name}", fontsize=20, pad=20)
    
    # Add statistics
    stat_text = "Run Statistics:\n"
    for exp in experiments:
        run_name = os.path.basename(exp["run_folder"])
        stat_text += f"\n{run_name}:\n"
        stat_text += f"  Mean cooperation rate: {exp['statistics']['cooperation_rates']['mean']:.1%}\n"
        stat_text += f"  Mean final score: {exp['statistics']['final_scores']['mean']:.1f}\n"
    
    plt.gcf().text(
        1.02, 0.5,
        stat_text,
        fontsize=10,
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
# %%
# Example usage
if __name__ == "__main__":

    # Experiment Run Success
    # Plot all experiments except liberal and conservative
    # run_folder = "data/run_20250926_185644" # gpt-4o-mini
    # model_name = "gpt-4o-mini"
    run_folder = "data/run_20250927_072359" # deepseek-chat
    model_name = "deepseek-chat"
    temperature = 0.7
    experiment_ids = [
        "same_as_you",
        "constant_defection",
        "constant_cooperation",
        # "constant_random",
        "human",
        "us_conservative",
        "us_liberal"
    ]
    plot_selected_experiments(
        run_folder,
        experiment_ids,
        model_name=model_name,
        save_path="plots/behavioral_experiments.png",
        title=f"Comparison of Agent Behaviors Under Different Beliefs with \n model: {model_name} & temperature: {temperature}",
        figsize=(15, 8)  # Larger figure for better readability with 5 experiments
    )
    
    # # Example 2: Compare same experiment across different runs
    # run_folders = [
    #     "data/run_20250926_185259",
    #     "data/run_20250926_185644"
    # ]
    # plot_experiment_comparison(
    #     run_folders,
    #     "same_as_you",
    #     save_path="plots/run_comparison.png",
    #     title="Same-as-You Experiment Across Runs",
    # )

# %%
