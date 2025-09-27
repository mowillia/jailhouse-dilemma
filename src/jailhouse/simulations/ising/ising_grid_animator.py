import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from typing import Optional

class IsingAnimator:
    def __init__(self, grid_size=5, J=1.0, T=2.0):
        """
        Initialize the 2D Ising model simulator
        
        Parameters:
        grid_size (int): Size of the square grid
        J (float): Coupling constant (J > 0 for ferromagnetic, J < 0 for antiferromagnetic)
        T (float): Temperature in units where Boltzmann constant k_B = 1
        """
        self.grid_size = grid_size
        self.J = J
        self.T = T
        
        # Initialize random grid with +1 and -1 spins
        self.grid = np.random.choice([-1, 1], size=(grid_size, grid_size))
        
        # Set up the figure and axis
        self.fig, self.ax = plt.subplots()
        # Create custom colormap (blue for -1, red for +1)
        self.colors = ['cornflowerblue', 'lightcoral']
        self.cmap = mcolors.ListedColormap(self.colors)
        
        # Initial plot
        # Convert from [-1,1] to [0,1] for plotting
        plot_grid = (self.grid + 1) / 2
        self.im = self.ax.imshow(plot_grid, cmap=self.cmap)
        
        # Remove axis ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Add title with parameters
        self.ax.set_title(f'2D Ising Model (J={J:.1f}, T={T:.1f})')
        
        # Overlay step counter
        self.step_text = self.ax.text(
            0.02,
            0.95,
            "",
            transform=self.ax.transAxes,
            color="white",
            fontsize=12,
            bbox=dict(facecolor='black', alpha=0.35, pad=3),
            va='top',
        )
    
    def calculate_energy_change(self, i, j):
        """Calculate the energy change if we were to flip the spin at position (i,j)"""
        current_spin = self.grid[i, j]
        # Get neighboring spins (with periodic boundary conditions)
        neighbors = [
            self.grid[(i-1) % self.grid_size, j],  # up
            self.grid[(i+1) % self.grid_size, j],  # down
            self.grid[i, (j-1) % self.grid_size],  # left
            self.grid[i, (j+1) % self.grid_size]   # right
        ]
        
        # Current energy contribution from this spin
        current_energy = -self.J * current_spin * sum(neighbors)
        # Energy if we flipped the spin
        flipped_energy = self.J * current_spin * sum(neighbors)
        
        return flipped_energy - current_energy
    
    def update(self, frame):
        """Update the grid using the Metropolis algorithm"""
        # Randomly select a cell to potentially flip
        i, j = np.random.randint(0, self.grid_size, 2)
        
        # Calculate the energy change if we were to flip this spin
        delta_E = self.calculate_energy_change(i, j)
        
        # Metropolis acceptance criterion
        if delta_E <= 0 or np.random.random() < np.exp(-delta_E / self.T):
            self.grid[i, j] *= -1  # flip the spin
        
        # Convert from [-1,1] to [0,1] for plotting
        plot_grid = (self.grid + 1) / 2
        self.im.set_array(plot_grid)
        self.step_text.set_text(f"Step {frame + 1}")
        return [self.im, self.step_text]
    
    def animate(self, frames=1000, interval=50, title: Optional[str] = None):
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