"""
Generate k-step precision and coverage figures for the paper.

Precision figure (kstep_all.pdf)
    Matrix-power vs empirical top-x precision for k = 1..MAX_K, one panel
    per environment.  TOP_X controls how many regions count as a "hit".

Coverage figure (kstep_coverage.pdf)
    For each environment and each k, the mean number of regions needed to
    capture COVERAGE_PROB of the probability mass (greedy highest-first),
    under both T^k and the empirical k-step matrix.

Output: notes/paper/imgs/kstep_all.pdf
        notes/paper/imgs/kstep_coverage.pdf
"""

import json, glob, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
os.chdir(ROOT)   # load_training_data uses relative paths

from configs import load_config
from utils import load_training_data
from ensemble import load_ensemble
from analysis.metrics import kstep_precision, kstep_coverage

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

OUT_DIR = os.path.join(os.path.dirname(__file__), 'paper', 'imgs')
os.makedirs(OUT_DIR, exist_ok=True)

MAX_K        = 8
TOP_X        = 3      # top-x precision; set to e.g. 3 to see top-3 precision
COVERAGE_PROB = 0.9   # probability mass threshold for coverage query

ENVS = [
    ('random_walk',    'Random Walk_20260417_134139.json',   'Random Walk'),
    ('bouncing_ball',  'Bouncing Ball_20260417_104143.json',  'Bouncing Ball'),
    ('cruise_control', 'Cruise Control_20260415_111039.json', 'Cruise Control'),
]

ENSEMBLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'results', 'ensembles')

color_mat = '#4C72B0'
color_emp = '#DD8452'

# ---------------------------------------------------------------------------
# Compute curves for each environment
# ---------------------------------------------------------------------------
prec_results     = []   # for precision figure
coverage_results = []   # for coverage figure

for config_name, manifest_file, label in ENVS:
    config = load_config(config_name)
    obs, _, _, mask = load_training_data(config['model_dir'])
    trees, _ = load_ensemble(os.path.join(ENSEMBLES_DIR, manifest_file))

    all_mat_prec, all_emp_prec = [], []
    all_mat_cov,  all_emp_cov  = [], []

    for tree in trees:
        mat_p, emp_p, n_trans = kstep_precision(
            tree, obs, mask, max_k=MAX_K, top_x=TOP_X,
            laplace=config.get('laplace', 0.0))
        mat_c, emp_c, _       = kstep_coverage(
            tree, obs, mask, max_k=MAX_K, prob=COVERAGE_PROB,
            laplace=config.get('laplace', 0.0))

        all_mat_prec.append(mat_p)
        all_emp_prec.append(emp_p)
        all_mat_cov.append(mat_c)
        all_emp_cov.append(emp_c)

    ks = np.arange(1, MAX_K + 1)

    prec_results.append((
        label, ks,
        np.nanmean(all_mat_prec, axis=0), np.nanstd(all_mat_prec, axis=0),
        np.nanmean(all_emp_prec, axis=0), np.nanstd(all_emp_prec, axis=0),
        n_trans,
    ))
    coverage_results.append((
        label, ks,
        np.nanmean(all_mat_cov,  axis=0), np.nanstd(all_mat_cov,  axis=0),
        np.nanmean(all_emp_cov,  axis=0), np.nanstd(all_emp_cov,  axis=0),
    ))
    print(f'{label}: done')


# ---------------------------------------------------------------------------
# Figure 1 — Precision (top-x)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.8))
fig.subplots_adjust(wspace=0.35, left=0.09, right=0.97, top=0.88, bottom=0.28)

prec_label = f'Top-{TOP_X} precision'

handles = None
for ax, (label, ks, mat_mean, mat_std, emp_mean, emp_std, _) in zip(axes, prec_results):
    h1, = ax.plot(ks, mat_mean, color=color_mat, linewidth=1.5, marker='o',
                  markersize=3, label=r'$\mathbf{T}^k$ (matrix power)')
    ax.fill_between(ks, mat_mean - mat_std, mat_mean + mat_std,
                    alpha=0.18, color=color_mat)

    h2, = ax.plot(ks, emp_mean, color=color_emp, linewidth=1.5, marker='s',
                  markersize=3, linestyle='--', label='Empirical $k$-step')
    ax.fill_between(ks, emp_mean - emp_std, emp_mean + emp_std,
                    alpha=0.18, color=color_emp)

    ax.set_title(label)
    ax.set_xlabel('$k$ (steps ahead)')
    ax.set_ylabel(prec_label)
    ax.set_ylim(0, 1)
    ax.set_xticks(ks)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if handles is None:
        handles = [h1, h2]

fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=7,
           framealpha=0.7, bbox_to_anchor=(0.5, 0.02))

out = os.path.join(OUT_DIR, 'kstep_all.pdf')
fig.savefig(out, bbox_inches='tight')
print(f'Saved {out}')
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Coverage (regions needed for COVERAGE_PROB probability mass)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.8))
fig.subplots_adjust(wspace=0.35, left=0.09, right=0.97, top=0.88, bottom=0.28)

handles = None
for ax, (label, ks, mat_mean, mat_std, emp_mean, emp_std) in zip(axes, coverage_results):
    h1, = ax.plot(ks, mat_mean, color=color_mat, linewidth=1.5, marker='o',
                  markersize=3, label=r'$\mathbf{T}^k$ (matrix power)')
    ax.fill_between(ks, mat_mean - mat_std, mat_mean + mat_std,
                    alpha=0.18, color=color_mat)

    h2, = ax.plot(ks, emp_mean, color=color_emp, linewidth=1.5, marker='s',
                  markersize=3, linestyle='--', label='Empirical $k$-step')
    ax.fill_between(ks, emp_mean - emp_std, emp_mean + emp_std,
                    alpha=0.18, color=color_emp)

    ax.set_title(label)
    ax.set_xlabel('$k$ (steps ahead)')
    ax.set_ylabel(f'Regions for {int(COVERAGE_PROB*100)}\% mass')
    ax.set_xticks(ks)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if handles is None:
        handles = [h1, h2]

fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=7,
           framealpha=0.7, bbox_to_anchor=(0.5, 0.02))

out = os.path.join(OUT_DIR, 'kstep_coverage.pdf')
fig.savefig(out, bbox_inches='tight')
print(f'Saved {out}')
plt.close(fig)
