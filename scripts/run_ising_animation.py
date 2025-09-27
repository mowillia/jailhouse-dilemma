import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from jailhouse.simulations.ising.ising_grid_animator import IsingAnimator

def main():
    # Create Ising model simulator
    # J > 0 for ferromagnetic coupling
    # T controls the temperature (try values around T_c ≈ 2.27 for 2D Ising model)
    animator = IsingAnimator(
        grid_size=5,  # Larger grid to see patterns better
        J=1.0,        # Ferromagnetic coupling
        T=3.0         # Temperature (try varying this!)
    )
    
    # Run animation with more frames and faster updates
    animator.animate(frames=2000, interval=1)

if __name__ == "__main__":
    main()