import numpy as np
from configs import load_config

from envs.load import load_env
from learning.splitting import split_on_action, split_on_reachability
from models.policy import load_or_train_model
from models.tree import TreeObserver
from ensemble import (build_ensemble, load_ensemble, build_label_matrix,
                      evaluate_ensemble, estimate_precision_ensemble,
                      save_eval_results)
from pipeline import run_carpet, run_carpet_fixed, sample_next_states

from utils import load_training_data, ResultsLogger, RunLogger


if __name__ == '__main__':

    # Set to a manifest path to skip training and load a complete ensemble.
    load_manifest = None
    # load_manifest = './data/results/ensembles/Cruise Control_20260415_111039.json'

    # Set to a partial manifest path to resume an interrupted ensemble build.
    # resume_manifest = './data/results/ensembles/Random Walk_20260415_112954.json'
    resume_manifest = None

    # load config
    config = load_config('random_walk')
    # config = load_config('bouncing_ball')
    # config = load_config('cruise_control')

    model_name = config['model_name']
    env_id = config['env_id']
    model_path = config['model_path']
    model_dir = config['model_dir']
    n_timesteps = config['n_timesteps']
    bounds = np.array(config['bounds'])
    resolution = config['resolution']
    n_runs = config['n_runs']
    initial_preds = config['initial_preds']
    pad_to_size = config['pad_to_size']
    mark_terminal = config['mark_terminal']
    estimation_runs = config['estimation_runs']
    rounds = config['rounds']
    n_dims = config['n_dims']
    n_acts = config['n_acts']
    laplace = config.get('laplace', 0.0)
    reachability_split = config.get('reachability_split', False)

    # load environment and model, and generate training data
    env = load_env(env_id)
    model = load_or_train_model(
        env, model_path, n_timesteps=500_000
    )
    obs, acts, _, mask = load_training_data(model_dir)

    with ResultsLogger(model_dir, model_name) as logger:

        het_thresh = config.get('het_thresh', 0.1)
        propagate = False
        k = 5

        max_regions = 150
        min_ll_improvement = 0.01
        ll_patience = 10

        carpet_kwargs = dict(
            het_thresh=het_thresh,
            n_samples=32,
            estimation_runs=estimation_runs,
            laplace=laplace,
            n_dims=n_dims,
            model_name=model_name,
            propagate=propagate,
            max_regions=max_regions,
            min_ll_improvement=min_ll_improvement,
            ll_patience=ll_patience,
        )

        def make_tree():
            t = TreeObserver(n_dims=n_dims, n_acts=n_acts, bounds=bounds,
                             initial_preds=initial_preds)
            if initial_preds is None:
                t.initialize_single_region()
            if mark_terminal:
                t.mark_terminal_states(obs, mask)
            if reachability_split:
                split_on_reachability(t, obs, mask, bounds)
            # split_on_action(t, obs, acts, mask, thresh=0.99, ratio_thresh=0.98)
            return t

        if load_manifest is not None:
            trees, _ = load_ensemble(load_manifest)
            manifest_path = load_manifest
            logger.log(f'Loaded ensemble from {manifest_path} ({len(trees)} members)')
        else:
            ensemble_description = (
                'No split on action'
            )
            trees, manifest_path = build_ensemble(
                k=k,
                make_tree=make_tree,
                env=env, model=model, logger=logger, model_dir=model_dir,
                env_name=model_name,
                description=ensemble_description,
                resume_manifest=resume_manifest,
                **carpet_kwargs,
            )

        # Ensemble evaluation
        logger.section('Ensemble evaluation')
        # for tree in trees:
        #     tree.set_transition_scores(obs, mask, n_step=1, laplace=laplace)

        all_obs_ens, label_matrix = build_label_matrix(trees, obs, mask)

        per_tree, (joint_ll, joint_perp, n_zero, n_total) = evaluate_ensemble(
            trees, all_obs_ens, label_matrix, obs, mask)

        lls = [s[0] for s in per_tree]
        perps = [s[1] for s in per_tree]
        logger.log(f'Per-tree LL:         mean={np.mean(lls):.4f}  std={np.std(lls):.4f}')
        logger.log(f'Per-tree perplexity: mean={np.mean(perps):.4f}  std={np.std(perps):.4f}')
        logger.log(f'Joint ensemble LL:   {joint_ll:.4f} | perplexity: {joint_perp:.4f} | zero-prob: {n_zero}/{n_total}')

        in_support, top1, ens_euclidean, ens_euclidean_true, ens_euclidean_ratio = (
            estimate_precision_ensemble(
                trees, all_obs_ens, label_matrix, env, model, n_runs=estimation_runs))
        logger.log(f'Ensemble precision — in-support: {in_support:.4f},  top-1: {top1:.4f}')
        logger.log(f'Ensemble euclidean error — predicted: {ens_euclidean:.4f}, '
                   f'true: {ens_euclidean_true:.4f}, ratio: {ens_euclidean_ratio:.4f}')

        save_eval_results(manifest_path, {
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

        # Single-run alternative (comment out build_ensemble above and use this):
        # run_logger = RunLogger(env_name=model_name, config_dict=carpet_kwargs,
        #                        description='Single run')
        # run_carpet(tree, env, model, logger, model_dir,
        #            run_logger=run_logger, **carpet_kwargs)
        # run_logger.close()
