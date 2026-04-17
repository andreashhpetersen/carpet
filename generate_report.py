"""
Generate a LaTeX report from CARPET experiment CSVs.

Usage:
    python generate_report.py                  # all envs
    python generate_report.py random_walk      # single env filter

Output:
    data/results/report.tex     — LaTeX document
    data/results/figs/          — one PDF per plot

Structure per environment:
    - Overview: one plot per metric with each ensemble as a mean ± std band
    - Per-ensemble subsections: individual member curves + ensemble mean
"""

import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from glob import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS_DIR = './data/results/runs'
ENSEMBLES_DIR = './data/results/ensembles'
FIG_DIR = './data/results/figs'
REPORT_PATH = './data/results/report.tex'

ENV_ORDER = ['Random Walk', 'Bouncing Ball', 'Cruise Control']


def load_ensemble_manifests():
    """Return dict mapping ensemble_id → manifest dict (from manifest files)."""
    manifests = {}
    for path in glob(f'{ENSEMBLES_DIR}/*.json'):
        try:
            with open(path) as f:
                m = json.load(f)
            eid = m.get('ensemble_id')
            if eid:
                manifests[eid] = m
        except (json.JSONDecodeError, KeyError):
            pass
    return manifests


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


def group_by_ensemble(runs):
    """
    Group runs by ensemble_id. Returns an ordered list of
    (ensemble_id, members) where members is a list of (meta, rows)
    sorted by member_index. Ensembles are ordered chronologically.
    """
    groups = defaultdict(list)
    for meta, rows in runs:
        eid = meta['config'].get('ensemble_id', meta['run_id'])
        groups[eid].append((meta, rows))

    # Sort members within each ensemble, then sort ensembles chronologically
    ordered = []
    for eid in sorted(groups.keys()):
        members = sorted(groups[eid], key=lambda x: x[0]['config'].get('member_index', 0))
        ordered.append((eid, members))
    return ordered


def get_series(rows, col):
    """Extract (xs, ys) from rows for a given column, skipping missing values."""
    pairs = [(int(r['round']), float(r[col]))
             for r in rows if r.get(col, '') not in ('', 'None')]
    if not pairs:
        return [], []
    xs, ys = zip(*pairs)
    return list(xs), list(ys)


def ensemble_mean_std(members, col):
    """
    Compute ensemble mean ± std across members for a given column.
    Returns (xs, means, stds) on the union of rounds present in all members.
    """
    all_series = [dict(zip(*get_series(rows, col))) for _, rows in members]
    # Use rounds where at least one member has data
    all_xs = sorted(set(x for s in all_series for x in s.keys()))
    means, stds = [], []
    for x in all_xs:
        vals = [s[x] for s in all_series if x in s]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return all_xs, means, stds


def short_label(eid, index):
    """E.g. 'Ens 1 (2026-04-14)'"""
    try:
        dt = datetime.strptime(eid, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d')
        return f'Ens {index + 1} ({dt})'
    except ValueError:
        return f'Ens {index + 1}'


# ---------------------------------------------------------------------------
# Overview plots: one band per ensemble
# ---------------------------------------------------------------------------

def _apply_band_style(ax, xs, means, stds, color, label):
    ys_lo = [m - s for m, s in zip(means, stds)]
    ys_hi = [m + s for m, s in zip(means, stds)]
    ax.plot(xs, means, marker='o', markersize=3, color=color, label=label)
    ax.fill_between(xs, ys_lo, ys_hi, alpha=0.18, color=color)


def plot_overview_metric(env_name, ensembles_with_labels, col, ylabel, fig_path):
    """One plot: each ensemble = mean ± std band."""
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (label, members) in enumerate(ensembles_with_labels):
        color = prop_cycle[i % len(prop_cycle)]
        xs, means, stds = ensemble_mean_std(members, col)
        if not xs:
            continue
        _apply_band_style(ax, xs, means, stds, color, label)
    ax.set_xlabel('Round')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{env_name} — {ylabel} (ensemble overview)')
    if len(ensembles_with_labels) > 1:
        ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)


def plot_overview_precision(env_name, ensembles_with_labels, fig_path):
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for col, ylabel, ax in [
        ('prec_1step', 'Precision (1-step)', axes[0]),
        ('prec_2step', 'Precision (2-step)', axes[1]),
    ]:
        for i, (label, members) in enumerate(ensembles_with_labels):
            color = prop_cycle[i % len(prop_cycle)]
            xs, means, stds = ensemble_mean_std(members, col)
            if not xs:
                continue
            _apply_band_style(ax, xs, means, stds, color, label)
        ax.set_xlabel('Round')
        ax.set_ylabel('Precision')
        ax.set_title(f'{env_name} — {ylabel}')
        if len(ensembles_with_labels) > 1:
            ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)


def plot_overview_euclidean(env_name, ensembles_with_labels, fig_path):
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, (ax_err, ax_ratio) = plt.subplots(1, 2, figsize=(11, 4))
    for i, (label, members) in enumerate(ensembles_with_labels):
        color = prop_cycle[i % len(prop_cycle)]
        xs_p, means_p, stds_p = ensemble_mean_std(members, 'euclidean_error')
        xs_t, means_t, stds_t = ensemble_mean_std(members, 'euclidean_true')
        xs_r, means_r, stds_r = ensemble_mean_std(members, 'euclidean_ratio')
        if xs_p:
            _apply_band_style(ax_err, xs_p, means_p, stds_p, color, f'{label} pred')
        if xs_t:
            ax_err.plot(xs_t, means_t, marker='o', markersize=3, color=color,
                        linestyle='--', label=f'{label} true')
            lo = [m - s for m, s in zip(means_t, stds_t)]
            hi = [m + s for m, s in zip(means_t, stds_t)]
            ax_err.fill_between(xs_t, lo, hi, alpha=0.1, color=color)
        if xs_r:
            _apply_band_style(ax_ratio, xs_r, means_r, stds_r, color, label)
    ax_err.set_xlabel('Round')
    ax_err.set_ylabel('Distance')
    ax_err.set_title(f'{env_name} — Euclidean error')
    if ax_err.get_legend_handles_labels()[0]:
        ax_err.legend(fontsize=8, loc='best')
    ax_ratio.axhline(1.0, color='grey', linewidth=0.8, linestyle=':')
    ax_ratio.set_xlabel('Round')
    ax_ratio.set_ylabel('Ratio (pred / true)')
    ax_ratio.set_title(f'{env_name} — Euclidean ratio')
    if len(ensembles_with_labels) > 1 and ax_ratio.get_legend_handles_labels()[0]:
        ax_ratio.legend(fontsize=9, loc='best')
    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Detail plots: individual member curves + ensemble mean, for one ensemble
# ---------------------------------------------------------------------------

def plot_detail_metric(env_name, ens_label, members, col, ylabel, fig_path):
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(6, 4))
    for j, (meta, rows) in enumerate(members):
        color = prop_cycle[j % len(prop_cycle)]
        xs, ys = get_series(rows, col)
        if not xs:
            continue
        ax.plot(xs, ys, marker='o', markersize=3, color=color,
                alpha=0.55, label=f'Member {j + 1}')
    # Ensemble mean
    xs_m, means, _ = ensemble_mean_std(members, col)
    if xs_m:
        ax.plot(xs_m, means, color='black', linewidth=2, linestyle='--', label='Mean')
    ax.set_xlabel('Round')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{env_name} — {ylabel} ({ens_label})')
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)


def plot_detail_precision(env_name, ens_label, members, fig_path):
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for col, ylabel, ax in [
        ('prec_1step', 'Precision (1-step)', axes[0]),
        ('prec_2step', 'Precision (2-step)', axes[1]),
    ]:
        for j, (meta, rows) in enumerate(members):
            color = prop_cycle[j % len(prop_cycle)]
            xs, ys = get_series(rows, col)
            if not xs:
                continue
            ax.plot(xs, ys, marker='o', markersize=3, color=color,
                    alpha=0.55, label=f'Member {j + 1}')
        xs_m, means, _ = ensemble_mean_std(members, col)
        if xs_m:
            ax.plot(xs_m, means, color='black', linewidth=2, linestyle='--', label='Mean')
        ax.set_xlabel('Round')
        ax.set_ylabel('Precision')
        ax.set_title(f'{env_name} — {ylabel}')
        ax.legend(fontsize=9, loc='best')
    fig.suptitle(ens_label, fontsize=10, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)


def plot_detail_euclidean(env_name, ens_label, members, fig_path):
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, (ax_err, ax_ratio) = plt.subplots(1, 2, figsize=(11, 4))
    for j, (meta, rows) in enumerate(members):
        color = prop_cycle[j % len(prop_cycle)]
        xs_p, ys_p = get_series(rows, 'euclidean_error')
        xs_t, ys_t = get_series(rows, 'euclidean_true')
        xs_r, ys_r = get_series(rows, 'euclidean_ratio')
        if xs_p:
            ax_err.plot(xs_p, ys_p, marker='o', markersize=3, color=color,
                        alpha=0.55, label=f'M{j+1} pred')
        if xs_t:
            ax_err.plot(xs_t, ys_t, marker='o', markersize=3, color=color,
                        linestyle='--', alpha=0.55, label=f'M{j+1} true')
        if xs_r:
            ax_ratio.plot(xs_r, ys_r, marker='o', markersize=3, color=color,
                          alpha=0.55, label=f'Member {j + 1}')
    # Ensemble means
    xs_m, means_p, _ = ensemble_mean_std(members, 'euclidean_error')
    if xs_m:
        ax_err.plot(xs_m, means_p, color='black', linewidth=2, linestyle='--', label='Mean pred')
    xs_m, means_t, _ = ensemble_mean_std(members, 'euclidean_true')
    if xs_m:
        ax_err.plot(xs_m, means_t, color='black', linewidth=2, linestyle=':', label='Mean true')
    xs_m, means_r, _ = ensemble_mean_std(members, 'euclidean_ratio')
    if xs_m:
        ax_ratio.plot(xs_m, means_r, color='black', linewidth=2, linestyle='--', label='Mean')
    ax_err.set_xlabel('Round')
    ax_err.set_ylabel('Distance')
    ax_err.set_title(f'{env_name} — Euclidean error')
    if ax_err.get_legend_handles_labels()[0]:
        ax_err.legend(fontsize=8, loc='best')
    ax_ratio.axhline(1.0, color='grey', linewidth=0.8, linestyle=':')
    ax_ratio.set_xlabel('Round')
    ax_ratio.set_ylabel('Ratio (pred / true)')
    ax_ratio.set_title(f'{env_name} — Euclidean ratio')
    if ax_ratio.get_legend_handles_labels()[0]:
        ax_ratio.legend(fontsize=9, loc='best')
    fig.suptitle(ens_label, fontsize=10, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def make_latex_figure(fig_path, caption, width=0.75):
    rel = os.path.relpath(fig_path, os.path.dirname(REPORT_PATH))
    return (
        r'\begin{figure}[H]' + '\n'
        r'  \centering' + '\n'
        f'  \\includegraphics[width={width}\\linewidth]{{{rel}}}' + '\n'
        f'  \\caption{{{caption}}}' + '\n'
        r'\end{figure}' + '\n'
    )


def format_run_id(run_id):
    try:
        return datetime.strptime(run_id, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        return run_id


def escape_latex(s):
    s = s.replace('\\', '\\textbackslash{}')
    for ch in ('_', '%', '&', '#', '$', '{', '}', '^', '~'):
        s = s.replace(ch, '\\' + ch)
    return s


def make_eval_table(ensembles, ens_labels, ens_descriptions):
    """
    Render a wide LaTeX table of ensemble evaluation results.
    Rows = ensembles; columns = all eval metrics.
    Ensembles without an 'eval' key are shown with dashes.
    """
    header = (
        r'\begin{table}[H]' + '\n'
        r'  \centering' + '\n'
        r'  \small' + '\n'
        r'  \begin{tabular}{l c r@{$\,\pm\,$}l r r r r l}' + '\n'
        r'  \toprule' + '\n'
        r'  Ensemble & $k$ & \multicolumn{2}{c}{Tree LL} & Joint LL & Joint Perp.'
        r' & In-supp & Top-1 & Eucl.\ pred\,/\,true (ratio) \\' + '\n'
        r'  \midrule' + '\n'
    )
    rows = ''
    for label, (eid, members) in zip(ens_labels, ensembles):
        k = len(members)
        first_meta = members[0][0]
        # Try to load eval from manifest (ens_descriptions dict has per-eid manifest data)
        ev = first_meta.get('_eval')  # injected below
        if ev:
            eucl = (f'${ev["euclidean_pred"]:.4f}$\\,/\\,'
                    f'${ev["euclidean_true"]:.4f}$ '
                    f'(${ev["euclidean_ratio"]:.3f}$)')
            row = (
                f'  {escape_latex(label)} & {k}'
                f' & ${ev["per_tree_ll_mean"]:.3f}$ & ${ev["per_tree_ll_std"]:.3f}$'
                f' & ${ev["joint_ll"]:.3f}$'
                f' & ${ev["joint_perp"]:.2f}$'
                f' & ${ev["precision_in_support"]:.3f}$'
                f' & ${ev["precision_top1"]:.3f}$'
                f' & {eucl}'
                r' \\' + '\n'
            )
        else:
            row = f'  {escape_latex(label)} & {k} & \\multicolumn{{8}}{{c}}{{---}} \\\\\n'
        rows += row

    footer = (
        r'  \bottomrule' + '\n'
        r'  \end{tabular}' + '\n'
        r'  \caption{Ensemble evaluation results. Tree LL: mean $\pm$ std of per-member'
        r' log-likelihood. Joint LL/Perp: ensemble-level log-likelihood and perplexity.'
        r' In-supp: fraction of steps where the actual next intersection region has'
        r' non-zero score. Top-1: fraction where the argmax predicted intersection region'
        r' matches the actual. Eucl.\ pred/true (ratio): score-weighted / true-region'
        r' Euclidean error and their ratio.}' + '\n'
        r'\end{table}' + '\n'
    )
    return header + rows + footer


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
  \item[Euclidean error (predicted)] For each observed transition $s_t \to s_{t+1}$, the
    tree predicts the next region as the argmax of its transition row. Points are sampled
    uniformly from the observed training states in that predicted region, and the mean
    Euclidean distance to the actual next state $s_{t+1}$ is recorded. Lower is better.
    Unlike precision, this metric is robust to small region misclassifications: predicting
    a neighbouring region incurs a small error rather than a binary miss.
  \item[Euclidean error (true)] The same computation, but sampling from the \emph{actual}
    next region that $s_{t+1}$ belongs to. This is the irreducible spatial error given the
    current partition resolution --- the best any predictor could achieve. It is expected to
    decrease as the partition becomes finer.
  \item[Euclidean ratio] Predicted error divided by true error. A ratio of 1.0 means the
    correct region was predicted; values above 1.0 indicate how much further off the
    prediction was relative to the partition's own spatial resolution.
\end{description}

\subsection*{Ensemble method}

To improve transition estimates, an ensemble of $k$ independently trained tree partitions
is built. Each member tree is trained on a different random sample of the collected
training trajectories, providing diversity in the learned partitions.

\paragraph{Intersection regions.}
A state is characterised by its \emph{intersection region}: the tuple
$(r_1, r_2, \ldots, r_k)$ of region labels, one per member tree. The intersection
implicitly defines a finer-grained partition of the state space than any single tree.

\paragraph{Transition model.}
Rather than enumerating the exponentially large intersection-region space, the ensemble
transition model scores training points directly. Given a current intersection region
$(r_1, \ldots, r_k)$, the score of a candidate next training point $x_j$ is
\[
  \mathrm{score}(x_j) = \prod_{i=1}^{k} T_i[r_i,\, L_{ji}]
\]
where $T_i$ is the transition matrix of tree $i$ and $L_{ji}$ is the region label of
$x_j$ in tree $i$. Working in log space, this is a sum of $k$ scalar lookups ---
$O(n_{\text{points}} \times k)$ per query with no combinatorial blow-up.

\paragraph{Prediction.}
The predicted next state is $x^* = \arg\max_j \mathrm{score}(x_j)$. Its intersection
region tuple $(L_{1j^*}, \ldots, L_{kj^*})$ is the predicted next intersection region.

\paragraph{Log-likelihood.}
The probability of the actual next intersection region is proportional to the total score
mass landing on training points that share that intersection tuple:
\[
  P(\text{actual region} \mid \text{current}) =
    \frac{\sum_{j \in \mathcal{R}_{\text{next}}} \mathrm{score}(x_j)}
         {\sum_{j} \mathrm{score}(x_j)}
\]
where $\mathcal{R}_{\text{next}}$ is the set of training points in the actual next
intersection region. The ensemble LL is the average log of this probability over all
observed transitions.

\paragraph{Euclidean error.}
The ensemble euclidean error is the score-weighted mean distance from all training points
to the actual next state --- the expected distance under the ensemble's full predictive
distribution, not just the argmax.

\subsection*{Trials and decisions}

\begin{itemize}
  \item \textbf{Propagate=True vs.\ False.} Splitting was initially propagated recursively
    to sibling regions after each split. This caused Random Walk to terminate in fewer rounds
    but with worse precision. All subsequent runs use \texttt{propagate=False}.
  \item \textbf{Stochastic fallback (tried and reverted).} The \texttt{no\_det\_groups} gate
    blocks regions where no deterministic transition grouping exists --- effectively blocking
    all genuinely stochastic regions. A $k{=}2$ clustering fallback in probability space was
    implemented to handle these. On Bouncing Ball it showed early gains but then degraded as
    rounds progressed (over-splitting). On Random Walk it was catastrophic: regions grew from
    7 to 89 in 7 rounds and precision collapsed. The fallback was reverted.
  \item \textbf{Early stopping.} A patience-based log-likelihood stopping criterion was
    added (\texttt{ll\_patience=10}): training stops when LL has not improved by at least
    \texttt{min\_ll\_improvement} for the specified number of consecutive rounds.
  \item \textbf{Outer-product ensemble (abandoned).} An earlier ensemble transition model
    computed the full outer product of per-tree marginals, producing a distribution over
    all intersection tuples with nonzero probability. This caused memory exhaustion at
    evaluation time and was replaced by the direct scoring approach described above.
\end{itemize}

\subsection*{Current status}

Random Walk converges reliably in 8--10 rounds to 21--26 regions with 1-step precision
around 0.57--0.65. Bouncing Ball does not converge: irreducible stochasticity in bounce
and hit zones keeps heterogeneity above the splitting threshold indefinitely, so the region
count grows without bound. Precision on Bouncing Ball is nonetheless high ($\approx$0.83--0.84)
even without convergence, suggesting the partition is already capturing the dominant
transition structure early. The results below reflect individual tree partitioning runs only;
ensemble evaluation results will be added as they become available.

"""


def generate_report(env_filter=None):
    all_runs = load_runs(env_filter)
    ens_manifests = load_ensemble_manifests()
    if not all_runs:
        print('No runs found.')
        return

    by_env = defaultdict(list)
    for meta, rows in all_runs:
        by_env[meta['env']].append((meta, rows))

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    known = [e for e in ENV_ORDER if e in by_env]
    extra = sorted(e for e in by_env if e not in ENV_ORDER)
    env_names_ordered = known + extra

    sections = []
    for env_name in env_names_ordered:
        runs = by_env[env_name]
        ensembles = group_by_ensemble(runs)  # [(eid, members), ...]
        n_ens = len(ensembles)
        body = f'\\section{{{escape_latex(env_name)}}}\n\n'

        env_safe = env_name.replace(' ', '_')

        # Build short labels for each ensemble; inject eval + description from manifests
        ens_labels = [short_label(eid, i) for i, (eid, _) in enumerate(ensembles)]
        for eid, members in ensembles:
            manifest = ens_manifests.get(eid, {})
            members[0][0]['_eval'] = manifest.get('eval')
            members[0][0]['_description'] = manifest.get('description', '')
        ensembles_with_labels = list(zip(ens_labels, [m for _, m in ensembles]))

        # ------------------------------------------------------------------
        # Ensemble index table
        # ------------------------------------------------------------------
        body += '\\subsection*{Ensembles}\n'
        body += '\\begin{itemize}\n'
        for label, (eid, members) in zip(ens_labels, ensembles):
            first_meta = members[0][0]
            k = len(members)
            desc = escape_latex(first_meta.get('description', ''))
            # Try to pull a cleaner description from the config description field if present
            body += f'  \\item \\textbf{{{escape_latex(label)}}} ({k} members): {desc}\n'
        body += '\\end{itemize}\n\n'

        # Ensemble evaluation table
        body += '\\subsection*{Ensemble evaluation}\n\n'
        body += make_eval_table(ensembles, ens_labels, ens_manifests) + '\n\n'

        # ------------------------------------------------------------------
        # Overview figures (all ensembles as bands) — only if >1 ensemble
        # ------------------------------------------------------------------
        if n_ens > 1:
            body += '\\subsection*{Overview (all ensembles)}\n\n'

            ov_ll = f'{FIG_DIR}/{env_safe}_overview_ll.pdf'
            plot_overview_metric(env_name, ensembles_with_labels, 'll',
                                 'Log-likelihood', ov_ll)
            body += make_latex_figure(
                ov_ll,
                f'Log-likelihood over rounds — all ensembles (mean $\\pm$ std).',
            ) + '\n'

            ov_prec = f'{FIG_DIR}/{env_safe}_overview_precision.pdf'
            plot_overview_precision(env_name, ensembles_with_labels, ov_prec)
            body += make_latex_figure(
                ov_prec,
                f'Precision (1- and 2-step) — all ensembles (mean $\\pm$ std).',
                width=0.9,
            ) + '\n'

            ov_euc = f'{FIG_DIR}/{env_safe}_overview_euclidean.pdf'
            plot_overview_euclidean(env_name, ensembles_with_labels, ov_euc)
            body += make_latex_figure(
                ov_euc,
                f'Euclidean error and ratio — all ensembles (mean $\\pm$ std).',
                width=0.9,
            ) + '\n'

            ov_reg = f'{FIG_DIR}/{env_safe}_overview_n_regions.pdf'
            plot_overview_metric(env_name, ensembles_with_labels, 'n_regions',
                                 'Number of regions', ov_reg)
            body += make_latex_figure(
                ov_reg,
                f'Number of regions over rounds — all ensembles (mean $\\pm$ std).',
            ) + '\n'

        # ------------------------------------------------------------------
        # Per-ensemble detail subsections
        # ------------------------------------------------------------------
        for label, (eid, members) in zip(ens_labels, ensembles):
            eid_safe = eid.replace(':', '-')
            body += f'\\subsection{{{escape_latex(label)}}}\n\n'
            desc = members[0][0].get('_description', '')
            if desc:
                body += escape_latex(desc) + '\n\n'

            det_ll = f'{FIG_DIR}/{env_safe}_{eid_safe}_ll.pdf'
            plot_detail_metric(env_name, label, members, 'll',
                               'Log-likelihood', det_ll)
            body += make_latex_figure(
                det_ll,
                f'{escape_latex(label)} — log-likelihood per member.',
            ) + '\n'

            det_prec = f'{FIG_DIR}/{env_safe}_{eid_safe}_precision.pdf'
            plot_detail_precision(env_name, label, members, det_prec)
            body += make_latex_figure(
                det_prec,
                f'{escape_latex(label)} — precision (1- and 2-step) per member.',
                width=0.9,
            ) + '\n'

            det_euc = f'{FIG_DIR}/{env_safe}_{eid_safe}_euclidean.pdf'
            plot_detail_euclidean(env_name, label, members, det_euc)
            body += make_latex_figure(
                det_euc,
                f'{escape_latex(label)} — euclidean error and ratio per member.',
                width=0.9,
            ) + '\n'

            det_reg = f'{FIG_DIR}/{env_safe}_{eid_safe}_n_regions.pdf'
            plot_detail_metric(env_name, label, members, 'n_regions',
                               'Number of regions', det_reg)
            body += make_latex_figure(
                det_reg,
                f'{escape_latex(label)} — number of regions per member.',
            ) + '\n'

        sections.append(body)

    doc = (
        r'\documentclass{article}' + '\n'
        r'\usepackage{graphicx}' + '\n'
        r'\usepackage{float}' + '\n'
        r'\usepackage{booktabs}' + '\n'
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
