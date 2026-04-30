"""
Sweep configuration for CARPET experiments.

Each entry in CONFIGS is a (name, env, overrides) triple:
  - name     : human-readable label appended to the run description
  - env      : config name passed to load_config()
  - overrides: dict of carpet_kwargs keys to override from the env's defaults

Run a single config:
    python run_sweep.py --idx 3

List all configs:
    python run_sweep.py --list

On a cluster (SLURM):
    sbatch --array=0-<N> run_sweep.sh
where run_sweep.sh calls: python run_sweep.py --idx $SLURM_ARRAY_TASK_ID
"""

# ---------------------------------------------------------------------------
# Baseline values (for reference — these match the per-env configs)
# ---------------------------------------------------------------------------
# Random Walk:    het_thresh=0.1, thresh_ratio=0.05, entropy_thresh=0.05,
#                 n_samples=32, laplace=0.5, max_regions=150
# Bouncing Ball:  het_thresh=0.1, thresh_ratio=0.05, entropy_thresh=0.05,
#                 n_samples=32, laplace=0.0, max_regions=150
# Cruise Control: het_thresh=0.1, thresh_ratio=0.05, entropy_thresh=0.05,
#                 n_samples=32, laplace=0.1, max_regions=150

# ---------------------------------------------------------------------------
# Sweeps: one parameter at a time, around baseline
# ---------------------------------------------------------------------------

_rw   = 'random_walk'
_bb   = 'bouncing_ball'
_cc   = 'cruise_control'

CONFIGS = [
    # ------------------------------------------------------------------
    # 0-2: het_thresh — primary convergence knob
    # ------------------------------------------------------------------
    ('het_thresh=0.05', _rw, {'het_thresh': 0.05}),
    ('het_thresh=0.10', _rw, {}),                        # baseline
    ('het_thresh=0.20', _rw, {'het_thresh': 0.20}),

    # ------------------------------------------------------------------
    # 3-5: thresh_ratio — balance gate
    # ------------------------------------------------------------------
    ('thresh_ratio=0.02', _rw, {'thresh_ratio': 0.02}),
    ('thresh_ratio=0.05', _rw, {}),                      # baseline
    ('thresh_ratio=0.10', _rw, {'thresh_ratio': 0.10}),

    # ------------------------------------------------------------------
    # 6-8: entropy_thresh — entropy gate
    # ------------------------------------------------------------------
    ('entropy_thresh=0.01', _rw, {'entropy_thresh': 0.01}),
    ('entropy_thresh=0.05', _rw, {}),                    # baseline
    ('entropy_thresh=0.10', _rw, {'entropy_thresh': 0.10}),

    # ------------------------------------------------------------------
    # 9-11: laplace — transition smoothing
    # ------------------------------------------------------------------
    ('laplace=0.0',  _rw, {'laplace': 0.0}),
    ('laplace=0.5',  _rw, {}),                           # baseline
    ('laplace=1.0',  _rw, {'laplace': 1.0}),

    # ------------------------------------------------------------------
    # 12-14: n_samples — most expensive; run last locally
    # ------------------------------------------------------------------
    ('n_samples=16', _rw, {'n_samples': 16}),
    ('n_samples=32', _rw, {}),                           # baseline
    ('n_samples=64', _rw, {'n_samples': 64}),

    # ------------------------------------------------------------------
    # 15-17: max_regions — region budget
    # ------------------------------------------------------------------
    ('max_regions=75',  _rw, {'max_regions': 75}),
    ('max_regions=150', _rw, {}),                        # baseline
    ('max_regions=300', _rw, {'max_regions': 300}),
]

N_CONFIGS = len(CONFIGS)
