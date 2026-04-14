"""
Generate a LaTeX report from CARPET experiment CSVs.

Usage:
    python generate_report.py                  # all envs
    python generate_report.py random_walk      # single env filter

Output:
    data/results/report.tex     — LaTeX document
    data/results/figs/          — one PDF per plot (per env × metric)

Plots per env (runs overlaid):
    - Precision 1-step and 2-step vs round
    - Number of regions vs round
    - Log-likelihood vs round
"""

import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS_DIR = './data/results/runs'
FIG_DIR = './data/results/figs'
REPORT_PATH = './data/results/report.tex'

METRICS = [
    ('prec_1step',  'Precision (1-step)',     'Precision'),
    ('prec_2step',  'Precision (2-step)',     'Precision'),
    ('n_regions',   'Number of regions',      'Regions'),
    ('ll',          'Log-likelihood',         'Log-likelihood'),
]


def load_runs(env_filter=None):
    """Return list of (meta dict, rows list-of-dicts) for all matching runs."""
    runs = []
    for meta_path in sorted(glob(f'{RUNS_DIR}/*/meta.json')):
        with open(meta_path) as f:
            meta = json.load(f)
        if env_filter and meta['env'] != env_filter:
            continue
        csv_path = os.path.join(os.path.dirname(meta_path), 'metrics.csv')
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, newline='') as f:
            rows = list(csv.DictReader(f))
        runs.append((meta, rows))
    return runs


def plot_metric(env_name, runs_with_labels, col, ylabel, fig_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, (meta, rows) in runs_with_labels:
        xs = [int(r['round']) for r in rows if r[col] not in ('', 'None')]
        ys = [float(r[col]) for r in rows if r[col] not in ('', 'None')]
        if not xs:
            continue
        ax.plot(xs, ys, marker='o', markersize=3, label=label)
    ax.set_xlabel('Round')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{env_name} — {ylabel}')
    if len(runs_with_labels) > 1:
        ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)


def make_latex_figure(fig_path, caption):
    rel = os.path.relpath(fig_path, os.path.dirname(REPORT_PATH))
    return (
        r'\begin{figure}[H]' + '\n'
        r'  \centering' + '\n'
        f'  \\includegraphics[width=0.75\\linewidth]{{{rel}}}' + '\n'
        f'  \\caption{{{caption}}}' + '\n'
        r'\end{figure}' + '\n'
    )


def format_run_id(run_id):
    """Convert '20260330_154301' → '2026-03-30 15:43:01' (no special chars)."""
    try:
        return datetime.strptime(run_id, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        return run_id


def escape_latex(s):
    s = s.replace('\\', '\\textbackslash{}')
    for ch in ('_', '%', '&', '#', '$', '{', '}', '^', '~'):
        s = s.replace(ch, '\\' + ch)
    return s


INTRO = r"""
\section*{Summary}

This report documents ongoing work on \textbf{CARPET} (Continuous-state Abstraction via
Region Partitioning and Entropy-based Tree-splitting), a method for learning
tree-structured partitions of RL environment state spaces. The goal is to produce
a compact, interpretable model of the policy's behaviour that can be used for formal
verification in UPPAAL.

\subsection*{Approach}

Starting from an initial action-based partition (one region per distinct policy action),
CARPET iteratively refines the partition by splitting regions whose transition dynamics
are heterogeneous --- i.e.\ states within the same region that transition to systematically
different next regions. Splitting candidates are ranked by a within-region heterogeneity
score (mean total-variation distance from the region mean) and processed in priority order
each round. A sequence of gates filters out splits that are unlikely to be productive
(terminal regions, too few samples, no deterministic grouping, imbalanced clusters,
insufficient entropy reduction).

\subsection*{Metric definitions}

\begin{description}
  \item[Precision (1-step)] Fraction of environment steps at which the tree's
    most likely predicted next region (argmax of the transition row) matches the region
    the model actually ends up in. Measured by running the trained RL policy in the
    real environment. Higher is better; 1.0 means perfect next-region prediction.
  \item[Precision (2-step)] Same as 1-step precision but predicting two steps ahead.
    The predictor uses the same current region for both steps (open-loop).
  \item[Number of regions] The number of leaf nodes in the tree partition after each
    round of splitting. More regions give finer resolution but increase model complexity
    and the risk of sparse transition estimates.
  \item[Log-likelihood] Average log-probability of observed transitions under the tree's
    learned transition model. Computed on held-out training trajectories. Higher (less
    negative) is better; a value of 0 would mean every transition is predicted with
    certainty.
\end{description}

\subsection*{Trials and decisions}

\begin{itemize}
  \item \textbf{Propagate=True vs.\ False.} Splitting was initially propagated recursively
    to sibling regions after each split. This caused Random Walk to terminate in fewer rounds
    but with worse precision. All subsequent runs use \texttt{propagate=False}.
  \item \textbf{Stochastic fallback (tried and reverted).} The \texttt{no\_det\_groups} gate
    blocks regions where no deterministic transition grouping exists --- effectively blocking
    all genuinely stochastic regions. A k=2 clustering fallback in probability space was
    implemented to handle these. On Bouncing Ball it showed early gains but then degraded as
    rounds progressed (over-splitting). On Random Walk it was catastrophic: regions grew from
    7 to 89 in 7 rounds and precision collapsed. The fallback was reverted.
  \item \textbf{Early stopping.} A patience-based log-likelihood early stopping criterion was
    added (\texttt{ll\_patience=10}): training stops when LL has not improved by at least
    \texttt{min\_ll\_improvement} for the specified number of consecutive rounds.
  \item \textbf{Ensemble.} To improve transition estimates, an ensemble of $k=5$ independently
    trained tree partitions is built, each trained on a different random sample of the
    training data. A factored (product-of-marginals) transition model is used to combine them.
    Joint ensemble evaluation (in-support and top-1 precision over intersection regions)
    is implemented but has not yet produced stable results due to memory constraints during
    evaluation.
\end{itemize}

\subsection*{Current status}

Random Walk converges reliably in 8--10 rounds to 21--26 regions with 1-step precision
around 0.57--0.65. Bouncing Ball does not converge: irreducible stochasticity in bounce
and hit zones keeps heterogeneity above the splitting threshold indefinitely, so the region
count grows without bound. Precision on Bouncing Ball is nonetheless high ($\approx$0.83--0.84)
even without convergence, suggesting the partition is already capturing the dominant
transition structure early. The results below reflect individual tree partitioning runs only.

"""


def generate_report(env_filter=None):
    all_runs = load_runs(env_filter)
    if not all_runs:
        print('No runs found.')
        return

    # Group by env
    by_env = defaultdict(list)
    for meta, rows in all_runs:
        by_env[meta['env']].append((meta, rows))

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    sections = []
    for env_name, runs in sorted(by_env.items()):
        body = f'\\section{{{escape_latex(env_name)}}}\n\n'

        # Assign a short letter label to each run
        letters = [chr(ord('A') + i) for i in range(len(runs))]
        runs_with_labels = list(zip(letters, runs))

        # Per-run description list with letter keys
        body += '\\subsection*{Run descriptions}\n\\begin{itemize}\n'
        for letter, (meta, _) in runs_with_labels:
            desc = escape_latex(meta['description'])
            rid = format_run_id(meta['run_id'])
            body += f'  \\item \\textbf{{{letter}}} ({rid}): {desc}\n'
        body += '\\end{itemize}\n\n'

        # One figure per metric
        body += '\\subsection*{Metrics}\n'
        for col, ylabel, short in METRICS:
            fig_path = f'{FIG_DIR}/{env_name}_{col}.pdf'
            plot_metric(env_name, runs_with_labels, col, ylabel, fig_path)
            caption = f'{ylabel} over rounds for \\texttt{{{escape_latex(env_name)}}}.'
            body += make_latex_figure(fig_path, caption) + '\n'

        sections.append(body)

    doc = (
        r'\documentclass{article}' + '\n'
        r'\usepackage{graphicx}' + '\n'
        r'\usepackage{float}' + '\n'
        r'\usepackage{hyperref}' + '\n'
        r'\usepackage{geometry}' + '\n'
        r'\geometry{margin=2.5cm}' + '\n'
        r'\title{CARPET Experiment Results}' + '\n'
        r'\author{}' + '\n'
        r'\date{\today}' + '\n'
        r'\begin{document}' + '\n'
        r'\maketitle' + '\n\n'
    )
    doc += INTRO
    doc += '\n'.join(sections)
    doc += '\n\\end{document}\n'

    with open(REPORT_PATH, 'w') as f:
        f.write(doc)

    print(f'Report written to {REPORT_PATH}')
    print(f'Figures saved to {FIG_DIR}/')
    print(f'Compile with: pdflatex -interaction=nonstopmode -output-directory data/results data/results/report.tex')


if __name__ == '__main__':
    env_filter = sys.argv[1] if len(sys.argv) > 1 else None
    generate_report(env_filter)
