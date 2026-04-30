"""
Runner for CARPET sweep experiments.

Usage:
    python run_sweep.py --idx 3              # run config at index 3
    python run_sweep.py --list               # print all configs
    python run_sweep.py --idx 0 --k 3       # override ensemble size

On a SLURM cluster:
    sbatch --array=0-17 run_sweep.sh
"""

import argparse
import numpy as np

from configs import load_config
from envs.load import load_env
from models.policy import load_or_train_model
from models.tree import TreeObserver
from learning.splitting import split_on_action, split_on_reachability
from ensemble import (build_ensemble, build_label_matrix, evaluate_ensemble,
                      estimate_precision_ensemble, save_eval_results)
from utils import load_training_data, ResultsLogger
from sweep import CONFIGS, N_CONFIGS


def run_config(idx, k=5):
    name, env_name, overrides = CONFIGS[idx]

    config = load_config(env_name)

    # Carpet kwargs: start from config defaults, apply overrides.
    carpet_kwargs = dict(
        het_thresh      = config.get('het_thresh', 0.1),
        n_samples       = 32,
        estimation_runs = config['estimation_runs'],
        laplace         = config.get('laplace', 0.0),
        n_dims          = config['n_dims'],
        model_name      = config['model_name'],
        propagate       = False,
        max_regions     = 150,
        min_ll_improvement = 0.01,
        ll_patience     = 10,
        thresh_ratio    = 0.05,
        entropy_thresh  = 0.05,
    )
    carpet_kwargs.update(overrides)

    model_name   = config['model_name']
    env_id       = config['env_id']
    model_path   = config['model_path']
    model_dir    = config['model_dir']
    n_timesteps  = config['n_timesteps']
    bounds       = np.array(config['bounds'])
    initial_preds = config['initial_preds']
    mark_terminal = config['mark_terminal']
    reachability_split = config.get('reachability_split', False)
    n_acts       = config['n_acts']

    env   = load_env(env_id)
    model = load_or_train_model(env, model_path, n_timesteps=n_timesteps)
    obs, acts, _, mask = load_training_data(model_dir)

    description = f'sweep idx={idx}: {name}'

    with ResultsLogger(model_dir, model_name) as logger:
        logger.log(f'Sweep config [{idx}]: {name}')
        logger.log(f'Overrides: {overrides}')

        def make_tree():
            t = TreeObserver(n_dims=config['n_dims'], n_acts=n_acts,
                             bounds=bounds, initial_preds=initial_preds)
            if initial_preds is None:
                t.initialize_single_region()
            if mark_terminal:
                t.mark_terminal_states(obs, mask)
            if reachability_split:
                split_on_reachability(t, obs, mask, bounds)
            return t

        trees, manifest_path = build_ensemble(
            k=k,
            make_tree=make_tree,
            env=env, model=model, logger=logger, model_dir=model_dir,
            env_name=model_name,
            description=description,
            resume_manifest=None,
            **carpet_kwargs,
        )

        logger.section('Ensemble evaluation')
        all_obs_ens, label_matrix = build_label_matrix(trees, obs, mask)
        per_tree, (joint_ll, joint_perp, n_zero, n_total) = evaluate_ensemble(
            trees, all_obs_ens, label_matrix, obs, mask)

        lls   = [s[0] for s in per_tree]
        perps = [s[1] for s in per_tree]
        logger.log(f'Per-tree LL:         mean={np.mean(lls):.4f}  std={np.std(lls):.4f}')
        logger.log(f'Per-tree perplexity: mean={np.mean(perps):.4f}  std={np.std(perps):.4f}')
        logger.log(f'Joint ensemble LL:   {joint_ll:.4f} | perplexity: {joint_perp:.4f} | zero-prob: {n_zero}/{n_total}')

        in_support, top1, ens_euclidean, ens_euclidean_true, ens_euclidean_ratio = (
            estimate_precision_ensemble(
                trees, all_obs_ens, label_matrix, env, model,
                n_runs=config['estimation_runs']))
        logger.log(f'Ensemble precision — in-support: {in_support:.4f},  top-1: {top1:.4f}')
        logger.log(f'Euclidean error — predicted: {ens_euclidean:.4f}, '
                   f'true: {ens_euclidean_true:.4f}, ratio: {ens_euclidean_ratio:.4f}')

        save_eval_results(manifest_path, {
            'sweep_idx':          idx,
            'sweep_name':         name,
            'overrides':          overrides,
            'per_tree_ll_mean':   float(np.mean(lls)),
            'per_tree_ll_std':    float(np.std(lls)),
            'per_tree_perp_mean': float(np.mean(perps)),
            'per_tree_perp_std':  float(np.std(perps)),
            'joint_ll':           float(joint_ll),
            'joint_perp':         float(joint_perp),
            'n_zero':             int(n_zero),
            'n_total':            int(n_total),
            'precision_in_support': float(in_support),
            'precision_top1':       float(top1),
            'euclidean_pred':     float(ens_euclidean),
            'euclidean_true':     float(ens_euclidean_true),
            'euclidean_ratio':    float(ens_euclidean_ratio),
        })


def list_configs():
    print(f'{"idx":>4}  {"env":<15}  {"name"}')
    print('-' * 50)
    for i, (name, env, overrides) in enumerate(CONFIGS):
        marker = '(baseline)' if not overrides else ''
        print(f'{i:>4}  {env:<15}  {name}  {marker}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx',  type=int,  default=None, help='Config index to run')
    parser.add_argument('--k',    type=int,  default=5,    help='Ensemble size')
    parser.add_argument('--list', action='store_true',     help='List all configs and exit')
    args = parser.parse_args()

    if args.list:
        list_configs()
    elif args.idx is None:
        parser.error('Provide --idx or --list')
    elif not (0 <= args.idx < N_CONFIGS):
        parser.error(f'--idx must be in 0..{N_CONFIGS - 1}')
    else:
        run_config(args.idx, k=args.k)
