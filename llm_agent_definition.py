# LLM agent definition
# %% 
from __future__ import annotations

import concurrent.futures
import os
import random
import time
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict, Literal, cast
from uuid import uuid4

import matplotlib.pyplot as plt
from langchain_community.chat_models import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
import asyncio
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
Action = Literal["cooperate", "defect"]


# Set your API key
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

class GameHistoryItem(TypedDict):
    """
    One game round record.

    - round: integer round index (starting at 1)
    - self_action: 'cooperate' | 'defect'
    - other_action: 'cooperate' | 'defect'
    - self_running_score: cumulative score for self after this round
    """

    round: int
    self_action: Action
    other_action: Action
    self_running_score: float


class MockAgent:
    """Mock agent for plotting averaged results."""
    def __init__(self, name: str, history: list):
        self.name = name
        self.history = history
        self.current_score = 0.0  # Add this to match LLMAgent interface


PAYOFF_MATRIX: dict[tuple[Action, Action], tuple[float, float]] = {
    ("cooperate", "cooperate"): (-5.0, -5.0),
    ("cooperate", "defect"): (-20.0, 0.0),
    ("defect", "cooperate"): (0.0, -20.0),
    ("defect", "defect"): (-10.0, -10.0),
}


BASE_SYSTEM_PROMPT = """You are an LLM agent that engages in repeated two-entity interactions with other agents like you. In each interaction you must choose one action: "cooperate" or "defect".

Payoff rules (your score change per round):
- If you "cooperate" and the other agent "defects": -20
- If you "defect" and the other agent "cooperates": 0
- If both "defect": -10
- If both "cooperate": -5

After the first round, you will be given a history of the "rounds" of your past interactions with other agents, where "self_action" is your action, "other_action" is the action of the other agent, and "self_running_score" is your total score that resulted from that action.

Goal:
- Over all the rounds, your goal is to achieve the least negative sum score possible

{instructions}

Output format (strict):
- Return exactly one word in lowercase: cooperate or defect
- Do not include any additional text, punctuation, or explanation.
"""

GAME_STATE_TEMPLATE = """Current total score: {current_score}

Game history:
{game_history}

Choose your next action (cooperate/defect):"""


class LLMAgent:
    """
    An LLM-powered agent with simple score tracking and conversation history.
    Uses LangChain + OpenAI chat models under the hood.
    """

    def __init__(
        self,
        name: str,
        instructions: str,
        *,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_output_tokens: int = 8,
        history: Optional[Sequence[GameHistoryItem]] = None,
        starting_score: float = 0.0,
        openai_api_key: Optional[str] = None,
    ) -> None:
        self.name: str = name
        self.instructions: str = instructions
        self.history: List[GameHistoryItem] = list(history) if history is not None else []
        self.current_score: float = float(starting_score)

        # Format the base system prompt with the experiment-specific instructions
        formatted_system_prompt = BASE_SYSTEM_PROMPT.format(instructions=instructions)

        self._llm = ChatDeepSeek(
            model_name="deepseek-chat",
            temperature=temperature,
            request_timeout=30.0,  # Add timeout
            max_retries=2,  # Add retry limit
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    formatted_system_prompt,
                ),
                ("human", "{game_state}"),
            ]
        )
        self._output_parser = StrOutputParser()
        self._default_max_output_tokens = max_output_tokens

    # ----- state management -----
    def update_history(self, record: GameHistoryItem) -> None:
        """Append a round record to the agent's history."""
        self.history.append(record)

    def record_round(
        self,
        *,
        round_index: int,
        self_action: Action,
        other_action: Action,
        self_running_score: float,
    ) -> None:
        """Convenience method to append a structured round record."""
        self.history.append(
            {
                "round": round_index,
                "self_action": self_action,
                "other_action": other_action,
                "self_running_score": self_running_score,
            }
        )

    def update_score(self, delta: float) -> float:
        """Increment the agent's score by delta (can be negative). Returns the new score."""
        self.current_score += float(delta)
        return self.current_score

    # ----- acting -----
    async def get_action_async(self) -> str:
        """
        Async version: Return exactly one word: 'cooperate' or 'defect', using the provided game prompt,
        current score, and internal game history.
        """
        # Format game history
        if self.history:
            history_lines = [
                (
                    f"Round {item['round']}: "
                    f"You chose {item['self_action']}, opponent chose {item['other_action']}. "
                    f"Your score: {item['self_running_score']}"
                )
                for item in self.history[-50:]  # Show last 50 rounds max
            ]
            history_text = "\n".join(history_lines)
        else:
            history_text = "No previous rounds played."

        # Format current game state
        game_state = GAME_STATE_TEMPLATE.format(
            current_score=self.current_score,
            game_history=history_text,
        )

        llm = self._llm.bind()
        chain = self._prompt | llm | self._output_parser

        try:
            raw = await chain.ainvoke(
                {
                    "game_state": game_state,
                },
                config=RunnableConfig(configurable={"tags": [self.name]})
            )
            action = self._sanitize_action(raw)
            print(f"{self.name} chose: {action}")  # Add debug output
            return action
        except Exception as e:
            print(f"Error getting action for {self.name}: {str(e)}")
            # Default to cooperate on error
            return "cooperate"
        
    def get_action(self) -> str:
        """
        Synchronous wrapper for get_action_async.
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.get_action_async())

    @staticmethod
    def _sanitize_action(text: str) -> str:
        t = (text or "").strip().lower()
        # quick normalization to enforce strict format
        if "cooperate" in t and "defect" in t:
            # pick the first matching token if both appear
            first = t.find("cooperate")
            second = t.find("defect")
            return "cooperate" if (first != -1 and (second == -1 or first < second)) else "defect"
        if "cooperate" in t:
            return "cooperate"
        if "defect" in t:
            return "defect"
        # fallback to first token or default
        token = (t.split() or ["cooperate"])[0]
        return "cooperate" if token not in {"cooperate", "defect"} else token


def _generate_agent_code() -> str:
    """Return a short unique identifier for an agent."""
    return uuid4().hex[:8].upper()


def create_agents(
    num_agents: int = 10,
    instructions: str = "Assume: The other entities are given the same instructions as you, and are LLM agents like you.",
) -> List[LLMAgent]:
    """
    Create agents with identical instructions and unique IDs.
    
    Args:
        num_agents: Number of agents to create
        instructions: Experiment-specific instructions for the agents
    """
    return [
        LLMAgent(
            name=f"Agent-{_generate_agent_code()}",
            instructions=instructions,
            model="deepseek-chat",
            temperature=0.7,
            starting_score=0.0,
        )
        for _ in range(num_agents)
    ]


class RoundOutcome(TypedDict):
    """Structured result for a single encounter between two agents."""

    experiment_round: int
    agents: Tuple[str, str]
    actions: Tuple[Action, Action]
    payoffs: Tuple[float, float]
    new_scores: Tuple[float, float]


def generate_random_pairs(
    agents: Sequence[LLMAgent], *, rng: Optional[random.Random] = None
) -> List[Tuple[LLMAgent, LLMAgent]]:
    """
    Generate random pairs from the list of agents.
    If there's an odd number of agents, one agent will be left unpaired.
    
    Returns:
        List of (agent_a, agent_b) pairs for this round
    """
    if len(agents) < 2:
        raise ValueError("At least two agents are required to generate pairs.")

    # Create a copy of the agent list that we can shuffle
    available_agents = list(agents)
    if rng is not None:
        rng.shuffle(available_agents)
    else:
        random.shuffle(available_agents)

    # Create pairs, leaving out one agent if the count is odd
    pairs = []
    for i in range(0, len(available_agents) - 1, 2):
        pairs.append((available_agents[i], available_agents[i + 1]))
    
    return pairs


async def play_round_async(
    agent_a: LLMAgent,
    agent_b: LLMAgent,
    *,
    experiment_round: int,
) -> RoundOutcome:
    """Execute one competitive interaction between ``agent_a`` and ``agent_b`` asynchronously."""

    # Get both actions concurrently
    action_a, action_b = await asyncio.gather(
        agent_a.get_action_async(),
        agent_b.get_action_async()
    )
    action_a = cast(Action, action_a)
    action_b = cast(Action, action_b)

    payoff_key = (action_a, action_b)
    if payoff_key not in PAYOFF_MATRIX:
        raise ValueError(f"Unsupported action pair: {payoff_key}")
    delta_a, delta_b = PAYOFF_MATRIX[payoff_key]

    round_index_a = len(agent_a.history) + 1
    round_index_b = len(agent_b.history) + 1

    new_score_a = agent_a.update_score(delta_a)
    new_score_b = agent_b.update_score(delta_b)

    agent_a.record_round(
        round_index=round_index_a,
        self_action=action_a,
        other_action=action_b,
        self_running_score=new_score_a,
    )
    agent_b.record_round(
        round_index=round_index_b,
        self_action=action_b,
        other_action=action_a,
        self_running_score=new_score_b,
    )

    return {
        "experiment_round": experiment_round,
        "agents": (agent_a.name, agent_b.name),
        "actions": (action_a, action_b),
        "payoffs": (delta_a, delta_b),
        "new_scores": (new_score_a, new_score_b),
    }

def play_round(
    agent_a: LLMAgent,
    agent_b: LLMAgent,
    *,
    experiment_round: int,
) -> RoundOutcome:
    """Synchronous wrapper for play_round_async."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(play_round_async(agent_a, agent_b, experiment_round=experiment_round))


def compute_cooperation_fraction(history: Sequence[GameHistoryItem]) -> float:
    """Return the fraction of rounds in which the agent cooperated."""

    if not history:
        return 0.0

    cooperations = sum(1 for item in history if item["self_action"] == "cooperate")
    return cooperations / len(history)


def aggregate_cooperation_over_time(
    agents: Sequence[LLMAgent], *, total_rounds: int
) -> List[float]:
    """
    Compute the fraction of cooperation across agents for each experiment round.
    
    For regular agents:
        Computes the fraction of cooperative actions in each round.
    For mock agents with pre-computed rates:
        Returns the stored cooperation rates directly.
    """
    # Check if we're dealing with mock agents that have pre-computed rates
    if hasattr(agents[0], 'history') and agents[0].history and 'cooperation_rate' in agents[0].history[0]:
        return [item['cooperation_rate'] for item in agents[0].history]
    
    # Regular agent processing
    cooperation_by_round: dict[int, list[bool]] = defaultdict(list)
    
    # Go through each agent's history
    for agent in agents:
        for item in agent.history:
            # The round number in the history is the experiment round
            experiment_round = item["round"]
            cooperation_by_round[experiment_round].append(
                item["self_action"] == "cooperate"
            )
    
    # Calculate cooperation rate for each round
    fractions: List[float] = []
    print("\nCooperation rate calculations:")
    
    for round_index in range(1, total_rounds + 1):
        round_actions = cooperation_by_round.get(round_index, [])
        if not round_actions:
            print(f"\nRound {round_index}: No actions found!")
            fractions.append(0.0)
        else:
            # Calculate fraction of cooperative actions in this round
            cooperation_rate = sum(round_actions) / len(round_actions)
            fractions.append(cooperation_rate)
            
            # Debug print to verify calculations
            print(f"\nRound {round_index}:")
            print(f"  Actions recorded: {len(round_actions)}")
            print(f"  Actions by agent:")
            for agent in agents:
                round_items = [item for item in agent.history if item["round"] == round_index]
                if round_items:
                    for item in round_items:
                        print(f"    {agent.name}: {item['self_action']}")
            print(f"  Cooperative actions: {sum(round_actions)}")
            print(f"  Rate: {cooperation_rate:.1%}")
    
    return fractions


def plot_cooperation_fraction(
    agents: Sequence[LLMAgent],
    *,
    total_rounds: int,
    experiment_name: str,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    linewidth: Optional[float] = None,
    marker: Optional[str] = None,
    markersize: Optional[float] = None,
    show: bool = True,
) -> Optional[plt.Axes]:
    """Plot how cooperation fraction evolves over the course of the experiment.
    
    Args:
        agents: Sequence of LLMAgent instances to analyze
        total_rounds: Total number of rounds in the experiment
        experiment_name: Name of the experiment being plotted
        save_path: Optional path to save the plot (e.g. "plots/exp1.png")
        title: Optional custom title (if None, uses experiment_name)
        ax: Optional matplotlib axes to plot on (if None, creates new figure)
        color: Optional color for the plot line
        linewidth: Optional linewidth for the plot line
        marker: Optional marker for the plot line
        markersize: Optional markersize for the plot line
        show: Whether to call plt.show() (default: True)

    Returns:
        The matplotlib axes object if ax was provided or show=False, None otherwise
    """
    plt.style.use('default')
    
    fractions = aggregate_cooperation_over_time(agents, total_rounds=total_rounds)
    rounds = list(range(1, total_rounds + 1))

    # Create new figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # Slightly larger figure
    
    # Plot the data
    plot_kwargs = {"marker": "o", "linewidth": 2, "label": f"{experiment_name}".replace("Presumed ", "")}
    if linewidth is not None:
        plot_kwargs["linewidth"] = linewidth
    if marker is not None:
        plot_kwargs["marker"] = marker
    if markersize is not None:
        plot_kwargs["markersize"] = markersize
    if color is not None:
        plot_kwargs["color"] = color
    ax.plot(rounds, fractions, **plot_kwargs)
    
    # Use experiment name in title if no custom title provided and no existing title
    if ax.get_title() == "" and title is None:
        ax.set_title(f"Cooperation Rate Over Time - {experiment_name}", fontsize=14, pad=20)
    elif title is not None:
        ax.set_title(title, fontsize=14, pad=20)
    
    ax.set_xlabel("Round", fontsize=18)
    ax.set_ylabel("Fraction Cooperating", fontsize=18)
    ax.set_ylim(-0.01, 1.2)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    # ax.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend(fontsize=18, loc='best')
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(pad=1.5)
    
    # Save plot if path provided
    if save_path:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    return ax


def plot_multiple_experiments(
    experiment_results: List[Dict[str, Any]],
    *,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    linewidth: Optional[float] = None,
    marker: Optional[str] = None,
    markersize: Optional[float] = None,
) -> None:
    """Plot multiple experiments on the same graph for comparison.
    
    Args:
        experiment_results: List of experiment result dictionaries
        save_path: Optional path to save the plot
        title: Optional custom title for the plot
        colors: Optional list of colors for the experiment lines
        linewidth: Optional linewidth for the plot line
        marker: Optional marker for the plot line
        markersize: Optional markersize for the plot line
    """
    if not experiment_results:
        return

    # Use default color cycle if no colors provided
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    fig, ax = plt.subplots(figsize=(12, 7))  # Larger figure for multiple lines
    
    for idx, result in enumerate(experiment_results):
        # Create a mock agent with the mean cooperation rates
        
        # Create a mock agent that stores the actual cooperation rates
        mock_history = []
        print(f"\nProcessing experiment: {result['name']}")
        for i, rate in enumerate(result["mean_cooperation_over_time"]):
            print(f"  Round {i+1}: Rate = {rate:.2%}")
            mock_history.append({
                "round": i + 1,
                "cooperation_rate": rate  # Store the actual rate
            })
        mock_agents = [MockAgent(f"Average_{result['name']}", mock_history)]
        
        # Debug print
        print(f"\nPlotting {result['name']}:")
        for i, rate in enumerate(result["mean_cooperation_over_time"]):
            print(f"  Round {i+1}: {rate:.1%} cooperation rate")
        
        # Plot this experiment
        color = colors[idx % len(colors)]
        experiment_name = f"{result['name']} (n={result['n_repeats']})"
        print(f"Plotting with experiment name: {experiment_name}")  # Debug print
        plot_cooperation_fraction(
            mock_agents,
            total_rounds=result["num_rounds"],
            experiment_name=experiment_name,
            ax=ax,
            color=color,
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
            show=False,
        )
    
    # Set overall title if provided
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


async def play_round_for_pair_async(pair_and_round: tuple[tuple[LLMAgent, LLMAgent], int]) -> RoundOutcome:
    """Helper function to play a round for a single pair asynchronously."""
    (agent_a, agent_b), round_number = pair_and_round
    return await play_round_async(agent_a, agent_b, experiment_round=round_number)

async def run_random_experiment_async(
    agents: Sequence[LLMAgent],
    *,
    num_rounds: int,
    rng: Optional[random.Random] = None,
    max_concurrent: Optional[int] = None,  # None means no limit
) -> List[RoundOutcome]:
    """
    Run num_rounds of interactions where in each round, all agents are randomly paired
    and interact with their assigned partner. All pairs are processed concurrently.
    
    Args:
        agents: Sequence of agents to participate in the experiment
        num_rounds: Number of rounds to run
        rng: Optional random number generator for reproducibility
        max_concurrent: Maximum number of concurrent API calls (None for no limit)
    """
    if num_rounds < 1:
        return []

    outcomes: List[RoundOutcome] = []
    
    for round_number in range(1, num_rounds + 1):
        start_time = time.perf_counter()
        
        # Generate random pairs for this round
        pairs = generate_random_pairs(agents, rng=rng)
        
        # Create list of (pair, round_number) tuples for concurrent processing
        pair_rounds = [(pair, round_number) for pair in pairs]
        
        # Process all pairs concurrently with a semaphore to control concurrency
        round_outcomes = []
        max_concurrent = max_concurrent or min(4, len(pair_rounds))  # Default to 4 concurrent calls or number of pairs if less
        sem = asyncio.Semaphore(max_concurrent)
        
        async def process_pair_with_semaphore(pair_round):
            async with sem:
                return await play_round_for_pair_async(pair_round)
        
        # Create tasks with semaphore control
        tasks = [process_pair_with_semaphore(pair_round) for pair_round in pair_rounds]
        
        # Process all pairs
        results = await asyncio.gather(*tasks)
        round_outcomes.extend(results)
        
        # Print results
        for outcome in results:
            print(
                f"Round {round_number}/{num_rounds} - Pair: {outcome['agents']} -> {outcome['actions']}"
                f" | payoffs={outcome['payoffs']}"
            )
        
        outcomes.extend(round_outcomes)
        
        # Print round summary
        elapsed = time.perf_counter() - start_time
        n_cooperations = sum(
            outcome['actions'].count('cooperate')
            for outcome in round_outcomes
        )
        total_actions = len(round_outcomes) * 2  # 2 actions per pair
        cooperation_rate = n_cooperations / total_actions if total_actions > 0 else 0
        print(
            f"\nRound {round_number} Summary:"
            f"\n  Pairs played: {len(round_outcomes)}"
            f"\n  Cooperation rate: {cooperation_rate:.1%}"
            f"\n  Time elapsed: {elapsed:.3f}s\n"
        )

    return outcomes

def run_random_experiment(
    agents: Sequence[LLMAgent],
    *,
    num_rounds: int,
    rng: Optional[random.Random] = None,
    max_concurrent: Optional[int] = None,
) -> List[RoundOutcome]:
    """Synchronous wrapper for run_random_experiment_async."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_random_experiment_async(
        agents,
        num_rounds=num_rounds,
        rng=rng,
        max_concurrent=max_concurrent
    ))

# %%
# Example random experiment loop. This will invoke live LLM calls.
if __name__ == "__main__":
    agents = create_agents(num_agents=25)
    rounds_to_run = 3
    experiment_outcomes = run_random_experiment(agents, num_rounds=rounds_to_run)
    for item in experiment_outcomes:
        print(item)
    plot_cooperation_fraction(
        agents,
        total_rounds=rounds_to_run,
        experiment_name="Test Run",
        linewidth=3.0,
        marker="o",
        markersize=10,
    )


# %%
