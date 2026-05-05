"""
Generate per-round refinement figures for the paper.

Two figures:
  1. Random Walk  — ll, prec_1step, euclidean_ratio vs round (mean ± std band)
  2. Bouncing Ball — same, illustrating non-convergence

Output: notes/imgs/refinement_rw.pdf, notes/imgs/refinement_bb.pdf
"""

import json, glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
    'pdf.fonttype': 42,
})

RUNS_DIR = '../data/results/runs'
OUT_DIR  = 'paper/imgs'
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load per-round CSVs for a given ensemble_id
# ---------------------------------------------------------------------------
def load_ensemble_runs(ensemble_id):
    runs = []
    for meta_path in sorted(glob.glob(f'{RUNS_DIR}/*/meta.json')):
        m = json.load(open(meta_path))
        if m.get('config', {}).get('ensemble_id') != ensemble_id:
            continue
        csv_path = os.path.join(os.path.dirname(meta_path), 'metrics.csv')
        if os.path.exists(csv_path):
            runs.append(pd.read_csv(csv_path))
    return runs


def mean_std_over_rounds(runs, col):
    """
    Align runs by round index, return (rounds, means, stds).
    Runs may have different lengths — pad with NaN.
    """
    max_len = max(len(r) for r in runs)
    matrix = np.full((len(runs), max_len), np.nan)
    for i, r in enumerate(runs):
        if col in r.columns:
            matrix[i, :len(r)] = r[col].values
    rounds = np.arange(1, max_len + 1)
    means = np.nanmean(matrix, axis=0)
    stds  = np.nanstd(matrix,  axis=0)
    return rounds, means, stds


# ---------------------------------------------------------------------------
# Plot one environment
# ---------------------------------------------------------------------------
METRICS = [
    ('ll',              'LL (nats)',       'Log-likelihood',         True,  None),
    ('prec_1step',      'Precision',       '1-step region precision', True,  (0, 1)),
    ('euclidean_ratio', 'Predicted / true baseline', 'Euclidean error ratio', False, None),
]

def plot_env(ensemble_id, env_label, out_path, hline_metrics=None):
    runs = load_ensemble_runs(ensemble_id)
    if not runs:
        print(f'No runs found for {ensemble_id}')
        return

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))
    fig.subplots_adjust(wspace=0.38, left=0.09, right=0.97, top=0.88, bottom=0.18)
    color = '#4C72B0'

    for ax, (col, ylabel, title, higher_better, ylim) in zip(axes, METRICS):
        rounds, means, stds = mean_std_over_rounds(runs, col)

        # Drop trailing NaN (runs that ended early don't pad)
        valid = ~np.isnan(means)
        r, m, s = rounds[valid], means[valid], stds[valid]

        ax.plot(r, m, color=color, linewidth=1.5, marker='o', markersize=3)
        ax.fill_between(r, m - s, m + s, alpha=0.20, color=color)

        if col == 'euclidean_ratio':
            ax.axhline(1.0, color='#cc3333', linewidth=0.9, linestyle='--',
                       label='baseline (ratio=1)')
            ax.legend(loc='upper right', fontsize=7, framealpha=0.7)

        if ylim:
            ax.set_ylim(ylim)

        ax.set_xlabel('Round')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(r)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved {out_path}  (k={len(runs)} members)')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generate figures
# ---------------------------------------------------------------------------

# Most recent ensemble per environment
plot_env(
    ensemble_id='20260417_134139',
    env_label='Random Walk',
    out_path=os.path.join(OUT_DIR, 'refinement_rw.pdf'),
)

plot_env(
    ensemble_id='20260417_104143',
    env_label='Bouncing Ball',
    out_path=os.path.join(OUT_DIR, 'refinement_bb.pdf'),
)

plot_env(
    ensemble_id='20260415_111039',
    env_label='Cruise Control',
    out_path=os.path.join(OUT_DIR, 'refinement_cc.pdf'),
)
