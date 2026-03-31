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

CSV_DIR = './data/results/csv'
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
    for json_path in sorted(glob(f'{CSV_DIR}/*.json')):
        with open(json_path) as f:
            meta = json.load(f)
        if env_filter and meta['env'] != env_filter:
            continue
        csv_path = json_path.replace('.json', '.csv')
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, newline='') as f:
            rows = list(csv.DictReader(f))
        runs.append((meta, rows))
    return runs


def plot_metric(env_name, runs, col, ylabel, fig_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for meta, rows in runs:
        xs = [int(r['round']) for r in rows if r[col] not in ('', 'None')]
        ys = [float(r[col]) for r in rows if r[col] not in ('', 'None')]
        if not xs:
            continue
        label = meta['description'][:60]  # truncate long labels
        ax.plot(xs, ys, marker='o', markersize=3, label=label)
    ax.set_xlabel('Round')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{env_name} — {ylabel}')
    if len(runs) > 1:
        ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)


def make_latex_figure(fig_path, caption):
    rel = os.path.relpath(fig_path, os.path.dirname(REPORT_PATH))
    return (
        r'\begin{figure}[ht]' + '\n'
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

        # Per-run description table
        body += '\\subsection*{Run descriptions}\n\\begin{itemize}\n'
        for meta, _ in runs:
            desc = escape_latex(meta['description'])
            rid = format_run_id(meta['run_id'])
            body += f'  \\item {rid}: {desc}\n'
        body += '\\end{itemize}\n\n'

        # One figure per metric
        body += '\\subsection*{Metrics}\n'
        for col, ylabel, short in METRICS:
            fig_path = f'{FIG_DIR}/{env_name}_{col}.pdf'
            plot_metric(env_name, runs, col, ylabel, fig_path)
            caption = f'{ylabel} over rounds for \\texttt{{{escape_latex(env_name)}}}.'
            body += make_latex_figure(fig_path, caption) + '\n'

        sections.append(body)

    doc = (
        r'\documentclass{article}' + '\n'
        r'\usepackage{graphicx}' + '\n'
        r'\usepackage{hyperref}' + '\n'
        r'\usepackage{geometry}' + '\n'
        r'\geometry{margin=2.5cm}' + '\n'
        r'\title{CARPET Experiment Results}' + '\n'
        r'\author{}' + '\n'
        r'\date{\today}' + '\n'
        r'\begin{document}' + '\n'
        r'\maketitle' + '\n\n'
    )
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
