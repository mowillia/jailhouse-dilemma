import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from typing import Dict, Tuple, List, Optional
from jailhouse.agents.llm_agent import LLMAgent
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from functools import partial
import atexit

# Global pool for multiprocessing
_process_pool = None

def get_process_pool():
    """Get or create the global process pool"""
    global _process_pool
    if _process_pool is None:
        _process_pool = Pool(processes=cpu_count())
    return _process_pool

def cleanup_pool():
    """Cleanup function to properly close the process pool"""
    global _process_pool
    if _process_pool is not None:
        _process_pool.close()
        _process_pool.join()
        _process_pool = None

# Register the cleanup function to run at exit
atexit.register(cleanup_pool)

class PrisonersDilemmaGrid:
    def __init__(self, grid_size=5):
        """
        Initialize the Prisoner's Dilemma grid simulation
        
        Parameters:
        grid_size (int): Size of the square grid
        """
        self.grid_size = grid_size
        
        # Payoff matrix values
        self.PAYOFF = {
            (True, True): -5,    # both cooperate
            (True, False): -20,  # cooperate while other defects
            (False, True): 0,    # defect while other cooperates
            (False, False): -10  # both defect
        }
        
        # Initialize grid of LLM agents
        self.agents = {}
        for i in range(grid_size):
            for j in range(grid_size):
                self.agents[(i,j)] = LLMAgent(position=(i,j))
        
        # Initialize visualization
        self.fig, (self.ax_grid, self.ax_scores) = plt.subplots(1, 2, figsize=(12, 5))
        self.colors = ['lightcoral', 'cornflowerblue']  # coral red for defect, cornflowerblue for cooperate
        self.cmap = mcolors.ListedColormap(self.colors)
        
        # Initialize decision grid for visualization
        self.decision_grid = np.zeros((grid_size, grid_size))
        # Add vmin and vmax to fix the color mapping
        self.im = self.ax_grid.imshow(self.decision_grid, cmap=self.cmap, vmin=0, vmax=1)
        
        # Initialize score visualization
        # Create a custom colormap for scores (purple to yellow)
        score_colors = ['mediumpurple', 'lavender', 'ivory', 'khaki', 'palegoldenrod'][::-1]
        self.score_cmap = mcolors.LinearSegmentedColormap.from_list('custom', score_colors)
        
        self.score_im = self.ax_scores.imshow(
            np.zeros((grid_size, grid_size)), 
            cmap=self.score_cmap
        )
        plt.colorbar(self.score_im, ax=self.ax_scores)
        
        # Setup axes
        self.ax_grid.set_title('Decisions (Blue=Cooperate, Coral=Defect)')
        self.ax_scores.set_title('Cumulative Scores')
        for ax in [self.ax_grid, self.ax_scores]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Ensure we start with a fresh process pool
        self.pool = get_process_pool()
        
        # Overlay step counter on the grid axis
        self.step_text = self.ax_grid.text(
            0.02,
            0.95,
            "",
            transform=self.ax_grid.transAxes,
            color="white",
            fontsize=12,
            bbox=dict(facecolor='black', alpha=0.35, pad=3),
            va='top',
        )
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get positions of neighboring agents (with periodic boundary conditions)"""
        i, j = pos
        return [
            ((i-1) % self.grid_size, j),  # up
            ((i+1) % self.grid_size, j),  # down
            (i, (j-1) % self.grid_size),  # left
            (i, (j+1) % self.grid_size)   # right
        ]

    @staticmethod
    def process_agent(agent):
        """Static method for parallel processing of agent decisions"""
        return agent.position, agent.decide_action()

    def update(self, frame):
        """Update the grid for one time step"""
        # First, have all agents make their decisions in parallel
        decisions = {}
        agents_list = list(self.agents.values())
        
        # Use the global process pool
        results = self.pool.map(self.process_agent, agents_list)
        decisions = dict(results)
        
        # Update the decision grid for visualization
        for (i, j), decision in decisions.items():
            # Convert boolean to float for proper color mapping
            self.decision_grid[i, j] = float(decision)
        
        # Now process interactions and update scores
        for pos, agent in self.agents.items():
            neighbors = self.get_neighbors(pos)
            for neighbor_pos in neighbors:
                # Record interaction in both agents' histories
                agent.record_interaction(neighbor_pos, decisions[neighbor_pos], decisions[pos])
                self.agents[neighbor_pos].record_interaction(pos, decisions[pos], decisions[neighbor_pos])
                
                # Update scores
                agent.score += self.PAYOFF[(decisions[pos], decisions[neighbor_pos])]
        
        # Update visualizations
        self.im.set_array(self.decision_grid)
        
        # Update score visualization
        score_grid = np.array([[self.agents[(i,j)].score 
                              for j in range(self.grid_size)]
                             for i in range(self.grid_size)])
        self.score_im.set_array(score_grid)
        self.score_im.set_clim(score_grid.min(), score_grid.max())
        
        # Update step counter text (frame is 0-indexed)
        self.step_text.set_text(f"Step {frame + 1}")
        
        return [self.im, self.score_im, self.step_text]
    
    def animate(self, frames=1000, interval=500, title: Optional[str] = None):
        """
        Create and display the animation
        
        Parameters:
        frames (int): Number of frames in the animation
        interval (int): Time between frames in milliseconds
        title (str | None): Optional overall title shown at the top
        """
        if title:
            self.fig.suptitle(title, fontsize=14)
        anim = FuncAnimation(
            self.fig, 
            self.update, 
            frames=frames,
            interval=interval, 
            blit=True
        )
        plt.show()
        return anim

    def __del__(self):
        """Cleanup method"""
        cleanup_pool()