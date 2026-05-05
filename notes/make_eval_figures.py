"""
Generate evaluation summary figures for the paper.

Picks the most recent ensemble with eval data per environment,
then plots LL, top-1 precision, and Euclidean ratio side by side.

Output: notes/imgs/eval_summary.pdf
"""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'pdf.fonttype': 42,   # embed fonts for pdflatex
})

ENSEMBLES_DIR = '../data/results/ensembles'
OUT_PATH = 'paper/imgs/eval_summary.pdf'

# ---------------------------------------------------------------------------
# Load manifests
# ---------------------------------------------------------------------------
manifests = []
for f in sorted(glob.glob(os.path.join(ENSEMBLES_DIR, '*.json'))):
    d = json.load(open(f))
    if d.get('eval'):
        manifests.append(d)

# Pick most recent ensemble per environment (sorted by ensemble_id = timestamp)
best = {}
for d in manifests:
    env = d['env']
    if env not in best or d['ensemble_id'] > best[env]['ensemble_id']:
        best[env] = d

# Fixed display order
ENV_ORDER = ['Random Walk', 'Bouncing Ball', 'Cruise Control']
ENV_LABELS = ['Random\nWalk', 'Bouncing\nBall', 'Cruise\nControl']
colors = ['#4C72B0', '#DD8452', '#55A868']

# ---------------------------------------------------------------------------
# Extract metrics
# ---------------------------------------------------------------------------
def get_metric(env, key_mean, key_std=None):
    d = best.get(env)
    if d is None:
        return np.nan, np.nan
    ev = d['eval']
    mean = ev.get(key_mean, np.nan)
    std  = ev.get(key_std, 0.0) if key_std else 0.0
    return mean, std

metrics = [
    {
        'title':   'Per-tree log-likelihood',
        'ylabel':  'LL (nats)',
        'key_mean': 'per_tree_ll_mean',
        'key_std':  'per_tree_ll_std',
        'higher_better': True,
    },
    {
        'title':   'Top-1 precision (1-step)',
        'ylabel':  'Precision',
        'key_mean': 'precision_top1',
        'key_std':  None,
        'higher_better': True,
        'ylim': (0, 1),
    },
    {
        'title':   'Euclidean error ratio',
        'ylabel':  'Predicted / baseline',
        'key_mean': 'euclidean_ratio',
        'key_std':  None,
        'higher_better': False,
        'hline': 1.0,
        'hline_label': 'baseline (ratio = 1)',
    },
]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))
fig.subplots_adjust(wspace=0.38, left=0.09, right=0.97, top=0.88, bottom=0.18)

x = np.arange(len(ENV_ORDER))
bar_w = 0.5

for ax, m in zip(axes, metrics):
    means = []
    stds  = []
    for env in ENV_ORDER:
        mn, sd = get_metric(env, m['key_mean'], m.get('key_std'))
        means.append(mn)
        stds.append(sd)

    bars = ax.bar(x, means, width=bar_w, color=colors,
                  yerr=stds if any(s > 0 for s in stds) else None,
                  capsize=3, error_kw={'linewidth': 0.8})

    if 'hline' in m:
        ax.axhline(m['hline'], color='#cc3333', linewidth=0.9,
                   linestyle='--', label=m['hline_label'])
        ax.legend(loc='upper right', framealpha=0.7)

    if 'ylim' in m:
        ax.set_ylim(m['ylim'])

    ax.set_title(m['title'])
    ax.set_ylabel(m['ylabel'])
    ax.set_xticks(x)
    ax.set_xticklabels(ENV_LABELS)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate bars with values
    for bar, mean in zip(bars, means):
        if not np.isnan(mean):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{mean:.2f}', ha='center', va='bottom', fontsize=7)

os.makedirs('imgs', exist_ok=True)
fig.savefig(OUT_PATH, bbox_inches='tight')
print(f'Saved to {OUT_PATH}')

# Print which ensemble was used per env
print('\nEnsembles used:')
for env in ENV_ORDER:
    d = best.get(env)
    if d:
        print(f'  {env}: {d["ensemble_id"]}  (k={d["k"]}, desc={d.get("description", "—")[:60]})')
    else:
        print(f'  {env}: NO DATA')
