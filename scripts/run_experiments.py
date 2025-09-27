# %%
"""
Experiment runner for the Jailhouse Dilemma project.
Loads experiment configurations from YAML and executes them using the core game logic.
"""

from __future__ import annotations
import nest_asyncio
import sys
from pathlib import Path

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml
from langchain_community.chat_models import ChatOpenAI
from langchain_deepseek import ChatDeepSeek

# Ensure 'src' is on sys.path so package imports work when running from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from jailhouse.agents.llm_agent_definition import (
    LLMAgent,
    MockAgent,
    aggregate_cooperation_over_time,
    plot_cooperation_fraction,
    plot_multiple_experiments,
    run_random_experiment_async,
)

# Set your API key
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

def load_experiment_config(config_path: str = "configs/experiments.yaml") -> Dict[str, Any]:
    """Load experiment configurations from YAML file and apply global settings."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Get global settings
    global_settings = config.get("settings", {})
    
    # Apply global settings to each experiment unless overridden
    for experiment in config["experiments"]:
        for key, value in global_settings.items():
            if key not in experiment:
                experiment[key] = value
    
    return config


def create_experiment_agents(
    experiment_config: Dict[str, Any],
    *,
    openai_api_key: Optional[str] = None,
) -> List[LLMAgent]:
    """Create agents for a specific experiment configuration."""
    num_agents = experiment_config.get("num_agents", 25)
    model = experiment_config.get("model", "deepseek-chat")
    temperature = experiment_config.get("temperature", 0.7)
    # Get experiment-specific instructions and ensure they start with "Assume:"
    instructions = experiment_config.get("instructions", 
        "The other entities are given the same instructions as you, and are LLM agents like you."
    )
    if not instructions.strip().startswith("Assume:"):
        instructions = f"Assume: {instructions}"

    print(f"\nCreating agents for experiment with instructions:")
    print(f"  {instructions}")

    return [
        LLMAgent(
            name=f"Agent-{i+1}",
            instructions=instructions,
            model=model,
            temperature=temperature,
            starting_score=0.0,
            openai_api_key=openai_api_key,
        )
        for i in range(num_agents)
    ]


async def run_single_simulation(
    experiment_id: str,
    experiment_config: Dict[str, Any],
    *,
    simulation_index: int,
    seed: Optional[int] = None,
    openai_api_key: Optional[str] = None,
    max_concurrent: Optional[int] = 4,  # Default to 4 concurrent API calls
) -> Dict[str, Any]:
    """
    Run a single simulation of an experiment.
    
    Args:
        experiment_id: Unique identifier for the experiment
        experiment_config: Configuration dictionary for the experiment
        simulation_index: Index of this simulation run
        seed: Optional random seed for reproducibility
        openai_api_key: Optional OpenAI API key
    
    Returns:
        Dictionary containing simulation results
    """
    simulation_start_time = time.perf_counter()
    # Set random seed if provided
    if seed is not None:
        # Use a different seed for each simulation by combining base seed and simulation index
        sim_seed = seed + simulation_index
        random.seed(sim_seed)
    
    # Create agents for this simulation
    agents = create_experiment_agents(
        experiment_config,
        openai_api_key=openai_api_key,
    )

    # Run the simulation
    num_rounds = experiment_config.get("num_rounds", 50)
    experiment_name = experiment_config["name"]
    
    print(f"\nRunning {experiment_name} - Simulation {simulation_index + 1}")
    print(f"Number of agents: {len(agents)}")
    print(f"Number of rounds: {num_rounds}")
    
    outcomes = await run_random_experiment_async(
        agents,
        num_rounds=num_rounds,
        max_concurrent=max_concurrent
    )

    # Compute statistics
    final_scores = [agent.current_score for agent in agents]
    
    # Safely compute cooperation rates
    cooperation_rates = []
    for agent in agents:
        if not agent.history:  # If no history, use 0 as cooperation rate
            cooperation_rates.append(0.0)
        else:
            rate = sum(1 for item in agent.history if item["self_action"] == "cooperate") / len(agent.history)
            cooperation_rates.append(rate)
    
    # Safely compute cooperation rate over time
    cooperation_over_time = aggregate_cooperation_over_time(agents, total_rounds=num_rounds)

    # Safely compute statistics
    def safe_stats(values):
        if not values:
            return {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    results = {
        "simulation_index": simulation_index,
        "experiment_id": experiment_id,
        "name": experiment_name,
        "num_agents": len(agents),
        "num_rounds": num_rounds,
        "seed": seed,
        "statistics": {
            "final_scores": safe_stats(final_scores),
            "cooperation_rates": safe_stats(cooperation_rates),
        },
        "cooperation_over_time": cooperation_over_time,
        "detailed_results": {
            "agent_scores": final_scores,
            "agent_cooperation_rates": cooperation_rates,
            "round_by_round_cooperation": [
                {
                    "round": i + 1,
                    "cooperation_rate": rate,
                    "cooperating_agents": int(rate * len(agents)),
                    "total_agents": len(agents)
                }
                for i, rate in enumerate(cooperation_over_time)
            ],
            "agent_histories": [
                {
                    "agent_id": agent.name,
                    "final_score": agent.current_score,
                    "rounds": [
                        {
                            "round": item["round"],
                            "self_action": item["self_action"],
                            "other_action": item["other_action"],
                            "running_score": item["self_running_score"]
                        }
                        for item in agent.history
                    ]
                }
                for agent in agents
            ]
        }
    }

    # Add execution time to results
    simulation_end_time = time.perf_counter()
    execution_time = simulation_end_time - simulation_start_time
    results["execution_time"] = execution_time
    
    # Print timing information immediately
    print(f"\nSimulation {simulation_index + 1} completed in {execution_time:.2f}s")
    
    return results


async def run_experiment(
    experiment_id: str,
    experiment_config: Dict[str, Any],
    *,
    n_repeats: int = 1,
    seed: Optional[int] = None,
    save_plots: bool = True,
    plots_dir: str = "plots",
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run an experiment multiple times and track timing information."""
    experiment_start_time = time.perf_counter()
    """
    Run an experiment with the given configuration, optionally repeating it multiple times.
    
    Args:
        experiment_id: Unique identifier for the experiment
        experiment_config: Configuration dictionary for the experiment
        n_repeats: Number of times to repeat the experiment (default: 1)
        seed: Optional random seed for reproducibility
        save_plots: Whether to save plots to disk
        plots_dir: Directory to save plots in
        openai_api_key: Optional OpenAI API key
    
    Returns:
        Dictionary containing aggregated experiment results
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")

    # Run all simulations
    simulations = []
    print(f"\nRunning {n_repeats} simulation{'s' if n_repeats > 1 else ''}")
    for i in range(n_repeats):
        print(f"\nSimulation {i + 1}/{n_repeats}")
        try:
            sim_result = await run_single_simulation(
                experiment_id,
                experiment_config,
                simulation_index=i,
                seed=seed,
                openai_api_key=openai_api_key,
            )
            simulations.append(sim_result)
            print(f"Simulation {i + 1} completed successfully:")
            print(f"  Mean cooperation rate: {sim_result['statistics']['cooperation_rates']['mean']:.2%}")
            print(f"  Mean final score: {sim_result['statistics']['final_scores']['mean']:.2f}")
        except Exception as e:
            print(f"Error in simulation {i + 1}:")
            print(f"  Type: {type(e).__name__}")
            print(f"  Message: {str(e)}")
            continue

    # Safely compute aggregate statistics
    def safe_aggregate_stats(simulations, stat_key):
        values = []
        for sim in simulations:
            stats = sim["statistics"][stat_key]
            values.extend([stats["mean"], stats["min"], stats["max"]])
        if not values:
            return {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
            }
        mean_val = sum(values) / len(values)
        return {
            "mean": mean_val,
            "min": min(values),
            "max": max(values),
            "std": (
                sum((x - mean_val) ** 2 for x in values) 
                / len(values)
            ) ** 0.5 if len(values) > 1 else 0.0,
        }

    # Aggregate statistics across simulations
    all_final_scores = safe_aggregate_stats(simulations, "final_scores")
    all_coop_rates = safe_aggregate_stats(simulations, "cooperation_rates")

    # Safely calculate mean cooperation rate over time
    num_rounds = experiment_config.get("num_rounds", 50)
    mean_cooperation_over_time = []
    for round_idx in range(num_rounds):
        try:
            round_rates = [sim["cooperation_over_time"][round_idx] for sim in simulations]
            if round_rates:
                mean_cooperation_over_time.append(sum(round_rates) / len(round_rates))
            else:
                mean_cooperation_over_time.append(0.0)
        except (IndexError, KeyError):
            mean_cooperation_over_time.append(0.0)

    # Generate plots if requested
    if save_plots:
        # Create plots directory if it doesn't exist
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(
            plots_dir,
            f"{experiment_id}_{n_repeats}runs_{timestamp}.png"
        )
        
        # Create a mock agent list with the mean cooperation rates for plotting
        
        # Create a history that will preserve the exact cooperation rates
        mock_history = []
        for i, rate in enumerate(mean_cooperation_over_time):
            # Store the actual rate in the history
            mock_history.append({
                "round": i + 1,
                "cooperation_rate": rate  # Store the actual rate
            })
        mock_agents = [MockAgent("AveragedAgent", mock_history)]
        
        # Plot average results
        plot_cooperation_fraction(
            mock_agents,
            total_rounds=num_rounds,
            experiment_name=f"{experiment_config['name']} (avg of {n_repeats} runs)",
            save_path=plot_path,
        )
    else:
        plot_path = None

    # Aggregate results
    results = {
        "experiment_id": experiment_id,
        "name": experiment_config["name"],
        "n_repeats": n_repeats,
        "num_rounds": num_rounds,
        "seed": seed,
        "statistics": {
            "final_scores": all_final_scores,
            "cooperation_rates": all_coop_rates,
        },
        "mean_cooperation_over_time": mean_cooperation_over_time,
        "plot_path": plot_path if save_plots else None,
        "individual_simulations": simulations,
    }

    # Add timing information
    experiment_end_time = time.perf_counter()
    total_execution_time = experiment_end_time - experiment_start_time
    total_simulation_time = sum(sim["execution_time"] for sim in simulations)
    overhead_time = total_execution_time - total_simulation_time
    
    results["timing"] = {
        "total_execution_time": total_execution_time,
        "total_simulation_time": total_simulation_time,
        "average_simulation_time": total_simulation_time / len(simulations) if simulations else 0,
        "overhead_time": overhead_time,
    }

    # Print experiment timing summary
    print(f"\nExperiment '{experiment_config['name']}' completed:")
    print(f"  Total execution time: {total_execution_time:.2f}s")
    print(f"  Total simulation time: {total_simulation_time:.2f}s")
    print(f"  Overhead time: {overhead_time:.2f}s")
    print(f"  Average simulation time: {results['timing']['average_simulation_time']:.2f}s")
    
    return results


async def run_all_experiments(
    config_path: str = "configs/experiments.yaml",
    *,
    n_repeats: Optional[int] = None,
    seed: Optional[int] = None,
    save_results: bool = True,
    data_dir: str = "data",
    combined_plot: bool = True,
    individual_plots: bool = False,
    openai_api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run all experiments defined in the config file and save results.
    
    Args:
        config_path: Path to YAML config file
        n_repeats: Optional number of times to repeat each experiment (overrides config file)
        seed: Optional random seed for reproducibility
        save_results: Whether to save results and plots to disk
        data_dir: Base directory to save results in (creates timestamped subdirectory)
        combined_plot: Whether to create a combined plot of all experiments (default: True)
        individual_plots: Whether to create individual plots for each experiment (default: False)
        openai_api_key: Optional OpenAI API key
    
    Returns:
        List of results dictionaries, one per experiment
    """
    overall_start_time = time.perf_counter()
    
    # Create timestamped directory for this run if saving results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(data_dir, f"run_{timestamp}")
        plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
    config = load_experiment_config(config_path)
    results = []

    for experiment in config["experiments"]:
        experiment_id = experiment["id"]
        try:
            # Use n_repeats from function argument if provided, otherwise from config
            experiment_n_repeats = n_repeats if n_repeats is not None else experiment.get("n_repeats", 1)
            
            print(f"\nStarting experiment: {experiment['name']}")
            print(f"Configuration:")
            print(f"  ID: {experiment_id}")
            print(f"  Repeats: {experiment_n_repeats}")
            print(f"  Model: {experiment.get('model', 'default')}")
            print(f"  Temperature: {experiment.get('temperature', 'default')}")
            print(f"  Num agents: {experiment.get('num_agents', 'default')}")
            print(f"  Num rounds: {experiment.get('num_rounds', 'default')}")
            print(f"Plots will be saved to: {plots_dir if save_results else 'nowhere (disabled)'}")
            
            try:
                result = await run_experiment(
                    experiment_id,
                    experiment,
                    n_repeats=experiment_n_repeats,
                    seed=seed,
                    save_plots=save_results and individual_plots,  # Save individual plots if requested
                    plots_dir=plots_dir if save_results else None,
                    openai_api_key=openai_api_key,
                )
                results.append(result)
                print(f"\nExperiment {experiment_id} completed successfully.")
            except asyncio.TimeoutError:
                print(f"\nError: Experiment {experiment_id} timed out")
                continue
            except Exception as e:
                print(f"\nError running experiment {experiment_id}:")
                print(f"  Type: {type(e).__name__}")
                print(f"  Message: {str(e)}")
                continue
        except Exception as e:
            print(f"\nError setting up experiment {experiment_id}:")
            print(f"  Type: {type(e).__name__}")
            print(f"  Message: {str(e)}")
            continue

    # Generate combined plot if requested
    if save_results and combined_plot and results:
        combined_plot_path = os.path.join(
            plots_dir,
            f"combined_experiments_{n_repeats}runs.png"
        )
        print(f"\nGenerating combined plot at: {combined_plot_path}")
        plot_multiple_experiments(
            results,
            save_path=combined_plot_path,
            title=f"Cooperation Rate Comparison (n={n_repeats} runs per experiment)"
        )
        print("Combined plot generated.")

    # Save experiment results to JSON
    if save_results:
        # Create a results summary that includes timing
        overall_end_time = time.perf_counter()
        overall_time = overall_end_time - overall_start_time
        total_experiment_time = sum(result["timing"]["total_execution_time"] for result in results)
        total_simulation_time = sum(result["timing"]["total_simulation_time"] for result in results)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config_file": config_path,
            "seed": seed,
            "timing": {
                "overall_execution_time": overall_time,
                "total_experiment_time": total_experiment_time,
                "total_simulation_time": total_simulation_time,
                "total_overhead_time": overall_time - total_simulation_time,
            },
            "experiments": results
        }
        
        # Save results to JSON
        import json
        results_file = os.path.join(run_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {run_dir}")

    return results

# %%
async def main():
    # Example usage - use n_repeats from config file
    save_results = True
    data_dir = "data"

    results = await run_all_experiments(
        config_path="configs/experiments.yaml",
        n_repeats=None,  # Use value from config file
        save_results=True,
        data_dir="data",
        combined_plot=True,
        individual_plots=True,
        openai_api_key=None,  # Use environment variable
    )
    return results

if __name__ == "__main__":
    import asyncio
    nest_asyncio.apply()
    results = asyncio.run(main())
    
    # Example - override n_repeats from config
    # results = run_all_experiments(
    #     n_repeats=5,  # Override config file setting
    #     save_plots=True,
    #     plots_dir="plots",
    #     combined_plot=True,
    #     individual_plots=False,
    #     seed=42,
    # )
    
    # Example of generating both combined and individual plots:
    # results = run_all_experiments(
    #     n_repeats=3,
    #     save_plots=True,
    #     plots_dir="plots",
    #     combined_plot=True,
    #     individual_plots=True,
    #     seed=42,
    # )
    
    # Example of generating only individual plots:
    # results = run_all_experiments(
    #     n_repeats=3,
    #     save_plots=True,
    #     plots_dir="plots",
    #     combined_plot=False,
    #     individual_plots=True,
    #     seed=42,
    # )
    
    # Calculate overall execution time
    overall_time = sum(result["timing"]["total_execution_time"] for result in results)
    total_experiment_time = overall_time
    total_simulation_time = sum(result["timing"]["total_simulation_time"] for result in results)

    # Print summary statistics
    print("\nExperiment Results Summary:")
    print("-" * 50)
    print(f"\nOverall Execution Time: {overall_time:.2f}s")
    print(f"Total Experiment Time: {total_experiment_time:.2f}s")
    print(f"Total Simulation Time: {total_simulation_time:.2f}s")
    print(f"Total Overhead Time: {(overall_time - total_simulation_time):.2f}s")
    print("-" * 50)

    for result in results:
        print(f"\nExperiment: {result['name']} ({result['n_repeats']} runs)")
        print("Final Scores:")
        print(f"  Mean: {result['statistics']['final_scores']['mean']:.2f}")
        print(f"  Std Dev: {result['statistics']['final_scores']['std']:.2f}")
        print(f"  Min: {result['statistics']['final_scores']['min']:.2f}")
        print(f"  Max: {result['statistics']['final_scores']['max']:.2f}")
        
        print("\nCooperation Rates:")
        print(f"  Mean: {result['statistics']['cooperation_rates']['mean']:.2%}")
        print(f"  Std Dev: {result['statistics']['cooperation_rates']['std']:.2%}")
        print(f"  Min: {result['statistics']['cooperation_rates']['min']:.2%}")
        print(f"  Max: {result['statistics']['cooperation_rates']['max']:.2%}")
        
        print("\nTiming Information:")
        print(f"  Total Execution Time: {result['timing']['total_execution_time']:.2f}s")
        print(f"  Total Simulation Time: {result['timing']['total_simulation_time']:.2f}s")
        print(f"  Average Simulation Time: {result['timing']['average_simulation_time']:.2f}s")
        print(f"  Overhead Time: {result['timing']['overhead_time']:.2f}s")
        
        if result['plot_path']:
            print(f"\nPlot saved to: {result['plot_path']}")
            
        # Print individual simulation results
        print("\nIndividual Simulation Results:")
        for sim in result['individual_simulations']:
            print(f"\n  Simulation {sim['simulation_index'] + 1}:")
            print(f"    Mean final score: {sim['statistics']['final_scores']['mean']:.2f}")
            print(f"    Mean cooperation rate: {sim['statistics']['cooperation_rates']['mean']:.2%}")
            print(f"    Execution time: {sim['execution_time']:.2f}s")

# %%
