configs/

Experiment configuration files.

Default file
- experiments.yaml: Defines global settings and a list of experiments. Fields include
  - num_agents, num_rounds, n_repeats, model, temperature
  - per-experiment: id, name, description, instructions

Usage
- scripts/run_experiments.py reads configs/experiments.yaml by default.
- To use a different file, edit the config path in run_experiments.py or pass it through if you add an argument.

