import sys
from pathlib import Path

# add src to path for local dev
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from jailhouse.simulations.prisoners_dilemma.prisoners_dilemma_grid import PrisonersDilemmaGrid, cleanup_pool
import atexit

def main():
    # Create Prisoner's Dilemma grid simulator
    simulator = PrisonersDilemmaGrid(
        grid_size=5
    )
    
    # Ensure cleanup happens at exit
    atexit.register(cleanup_pool)
    
    # Run animation
    simulator.animate(frames=1, interval=200)

if __name__ == "__main__":
    main()