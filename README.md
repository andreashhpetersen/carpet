# CARPET

**Coarse Action Reinforcement Policy Extraction Tool**

Learns an interpretable MDP abstraction (a binary decision tree with annotated
transition probabilities) from a trained RL policy, without white-box access to
the network.

---

## Quick start

### Single run

Edit `main.py` to select the environment config and any hyperparameters, then:

```bash
python main.py
```

Results are written to `data/results/` and the ensemble manifest is saved under
`data/results/ensembles/`.

### Systematic experiments (parameter sweep)

Sweep configs are defined in `sweep.py` as an indexed list. Each entry specifies
an environment and a set of hyperparameter overrides relative to the environment's
baseline.

**List all configs:**
```bash
python run_sweep.py --list
```

**Run a single config by index:**
```bash
python run_sweep.py --idx 3
```

**Run with a smaller ensemble (faster locally):**
```bash
python run_sweep.py --idx 3 --k 3
```

### On a SLURM cluster

Write a small submission script `run_sweep.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=carpet_sweep
#SBATCH --output=logs/sweep_%a.out
#SBATCH --array=0-17

python run_sweep.py --idx $SLURM_ARRAY_TASK_ID
```

Then submit:
```bash
sbatch run_sweep.sh
```

Each job runs independently and writes its own manifest to `data/results/ensembles/`.

---

## Adding or modifying sweep configs

Open `sweep.py`. Each entry in `CONFIGS` is a tuple:

```python
('descriptive_name', 'env_config_name', {'param': value, ...})
```

- `env_config_name` is passed to `load_config()` — currently one of
  `'random_walk'`, `'bouncing_ball'`, or `'cruise_control'`.
- The overrides dict is merged on top of the defaults in `run_sweep.py`.
  An empty dict `{}` marks a baseline run.
- The index of each entry is its `--idx` value, so **append new configs
  at the end** to avoid shifting existing indices.

Current baseline defaults (defined in `run_sweep.py`):

| Parameter        | Default | Notes                                      |
|------------------|---------|--------------------------------------------|
| `het_thresh`     | 0.1     | Min heterogeneity to attempt a split       |
| `thresh_ratio`   | 0.05    | Min balance ratio between clusters         |
| `entropy_thresh` | 0.05    | Min entropy reduction to commit a split    |
| `n_samples`      | 32      | Next-state samples per observed point      |
| `laplace`        | env     | Laplace smoothing on transition matrix     |
| `max_regions`    | 150     | Hard upper bound on number of leaves       |
| `k` (ensemble)   | 5       | Overrideable via `--k` on the command line |

Note: `n_samples` is the primary runtime bottleneck. Keep it low (16–32)
when exploring locally; use 32–64 for final experiments on the cluster.

---

## Generating the report

After one or more runs have completed:

```bash
python generate_report.py
pdflatex -output-directory data/results data/results/report.tex
```

The report groups runs by ensemble, plots per-round metrics, and includes an
evaluation table. Ensemble manifests must have an `eval` key (written
automatically at the end of each run) to appear in the table.

---

## Environment configs

Environment-specific settings (bounds, model paths, initial predicates, etc.)
live in `configs.py`. To add a new environment, add a config function there and
register it in `load_config()`.

---

## Project layout

```
main.py              Single-run entry point
run_sweep.py         Sweep runner (--idx / --list / --k)
sweep.py             Sweep config definitions
configs.py           Per-environment hyperparameter configs
pipeline.py          run_carpet / run_carpet_fixed / sample_next_states
ensemble.py          Ensemble build, evaluation, manifest I/O
generate_report.py   LaTeX report generation from saved manifests
learning/            Splitting functions (split_on_transition_guided, etc.)
models/              Tree structure and RL model loading
envs/                Environment registration and loading
analysis/            Evaluation metrics
viz/                 Partition plotting
data/results/        Run outputs (manifests, figures, report)
saved_models/        Trained RL policies
notes/               Paper draft and working notes
```
