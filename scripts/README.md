scripts/

CLI entry points for running experiments and simulations.

Commands
- Run experiments using configs/experiments.yaml
  python run_experiments.py

- Print final cooperation fractions (ASCII)
  python final_frac_table.py data/run_YYYYMMDD_HHMMSS/results.json

- Print average auto-correlation (ASCII)
  python autocorr_table.py data/run_YYYYMMDD_HHMMSS/results.json

- Visual simulations
  python run_ising_animation.py
  python run_prisoners_dilemma.py

