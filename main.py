import numpy as np
import gymnasium as gym
import uppaal_gym
from collections import defaultdict
from itertools import combinations
from configs import load_config

from analysis.metrics import estimate_precision_model, estimate_precision_tree, evaluate
from envs.load import load_env
from learning.splitting import split_on_action
from models.policy import load_or_train_model
from models.tree import TreeObserver
from pipeline import run_carpet, run_carpet_fixed, sample_next_states
from viz.plotting import plot_tree_partition

from utils import pad_to_array, save_training_data, load_training_data, ResultsLogger, CSVLogger


if __name__ == '__main__':

    # load config
    # config = load_config('random_walk')
    config = load_config('bouncing_ball')
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

    # load environment and model, and generate training data
    env = load_env(env_id)
    model = load_or_train_model(
        env, model_path, n_timesteps=500_000
    )
    obs, acts, _, mask = load_training_data(model_dir)

    tree = TreeObserver(
        n_dims=n_dims, n_acts=n_acts,
        bounds=bounds, initial_preds=initial_preds
    )

    if mark_terminal:
        tree.mark_terminal_states(obs, mask)

    with ResultsLogger(model_dir, model_name) as logger:

        # learn initial action mapping
        split_on_action(tree, obs, acts, mask, thresh=0.99, ratio_thresh=0.98)
        tree.reorder_leaf_labels()

        logger.section('Initial action mapping')
        logger.log(f'Regions: {tree.n_leaves}')

        if n_dims == 2:
            plot_tree_partition(
                tree,
                title=f"{model_name} - Initial action mapping",
                draw_boundaries=True,
                points=obs, acts=acts, mask=mask,
                save_dir='./data/figs/'
            )

        het_thresh = config.get('het_thresh', 0.1)
        propagate = False

        run_description = (
            f"Stochastic fallback when no det groups - propagate={propagate}. "
            f"het_thresh={het_thresh}, laplace={laplace}."
        )
        csv_logger = CSVLogger(
            env_name=model_name,
            config_dict={
                'het_thresh': het_thresh,
                'laplace': laplace,
                'propagate': propagate,
                'n_samples': 32,
                'estimation_runs': estimation_runs,
            },
            description=run_description,
        )

        run_carpet(
            tree, env, model, logger, model_dir,
            het_thresh=het_thresh,
            n_samples=32,
            estimation_runs=estimation_runs,
            laplace=laplace,
            n_dims=n_dims,
            model_name=model_name,
            save_dir='./data/figs/',
            propagate=propagate,
            csv_logger=csv_logger,
        )
        csv_logger.close()
        # run_carpet_fixed(
        #     tree, env, model, logger, model_dir,
        #     rounds=rounds,
        #     n_samples=32,
        #     estimation_runs=estimation_runs,
        #     laplace=laplace,
        #     n_dims=n_dims,
        #     model_name=model_name,
        #     save_dir='./data/figs/',
        # )
