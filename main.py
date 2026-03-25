import numpy as np
import gymnasium as gym
import uppaal_gym
from collections import defaultdict
from itertools import combinations
from configs import load_config

from analysis.metrics import estimate_precision_model, estimate_precision_tree, evaluate
from envs.load import load_env
from learning.splitting import split_on_action, split_on_transition, split_on_transition_tv, split_on_transition_unified
from models.policy import load_or_train_model
from models.tree import TreeObserver, State
from viz.plotting import plot_tree_partition

from utils import pad_to_array, normalize_to_prob, save_training_data, load_training_data



def sample_next_states(tree : TreeObserver, env, obs, mask, n_samples=16):
    """
    Return a list of size `n_regions` where each element is a list of `State`
    objects generated from observed points in `obs` and annotated with their
    next states as sampled `n_samples` times from `env`
    """

    print(f'sample next states\n')

    n_regions = tree.n_leaves
    region2states = [[] for _ in range(n_regions)]

    for x in obs[mask]:
        region = tree.get(x)
        state = State(x, region=region.label)

        next_states = []
        ps = None
        for i in range(n_samples):
            ns, _, _ = env.unwrapped.step_from(x, region.action)
            next_states.append(ns)

            # break if transition is deterministic
            if np.array_equal(ns, ps):
                break
            else:
                ps = ns

        next_states = np.array(next_states)
        state.next_states = next_states
        state.next_regions = tree.get_labels(next_states)
        state.make_transition_probability_vector(n_regions)

        region2states[region.label].append(state)

    return region2states


def get_transitions(tree, obs, mask, n_step=1):
    """Get the transitions observed in the data, annotated with their current and
    next region according to the tree."""
    transitions = []
    for run, run_mask in zip(obs, mask):
        run_obs = run[run_mask]
        labels = tree.get_labels(run_obs)
        transitions.append(np.vstack(
            [labels[i:len(labels) - n_step + i] for i in range(n_step + 1)]
        ).T)

    return np.concatenate(transitions)


if __name__ == '__main__':

    # load config
    # config = load_config('random_walk')
    # config = load_config('bouncing_ball')
    config = load_config('cruise_control')

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
    # rounds = 2
    n_dims = config['n_dims']
    n_acts = config['n_acts']

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

    # learn initial action mapping
    split_on_action(tree, obs, acts, mask, thresh=0.99, ratio_thresh=0.98)
    tree.reorder_leaf_labels()

    if n_dims == 2:
        plot_tree_partition(
            tree,
            title=f"{model_name} - Initial action mapping",
            draw_boundaries=True,
            points=obs, acts=acts, mask=mask,
            save_dir='./data/figs/'
        )

    # prepare observations and make splits according to the NEXT action taken
    # tree.split_for_action(obs, mask)
    # tree.set_transition_scores(obs, mask)

    # plot_tree_partition(
    #     tree,
    #     title="Split for next action",
    #     draw_boundaries=True,
    #     # points=obs, acts=acts, mask=mask,
    # )

    for i in range(rounds):
        print(f"\nRound {i+1}")

        # state sampling
        region2states = sample_next_states(tree, env, obs, mask, n_samples=32)

        # splitting — swap between the two to compare:
        #   split_on_transition:    deterministic-groups heuristic (original)
        #   split_on_transition_tv: probability-space clustering with TV + accuracy gates
        # split_on_transition_tv(region2states, tree)
        split_on_transition(region2states, tree)
        # split_on_transition_unified(region2states, tree)
        print(f'Regions: {tree.n_leaves}')

        # update transition scores
        obs, acts, _, mask = load_training_data(model_dir)
        tree.set_transition_scores(obs, mask)

        # evaluate
        evaluate(tree, obs, mask)

        # check precision
        T1 = np.array([l.T for l in tree.leaves()])
        _, reg_precision_model = estimate_precision_model(T1, tree, env, model, n_runs=estimation_runs)
        _, reg_precision_tree  = estimate_precision_tree(T1, tree, env, model, n_runs=estimation_runs)
        print(f'Precision (1 step, model acting): {reg_precision_model}')
        print(f'Precision (1 step, tree acting):  {reg_precision_tree}')

        # make n step transition matrix and check n step precision
        n_step = 2
        T = np.zeros((tree.n_leaves,) * (n_step + 1), dtype=np.int32)

        transitions = get_transitions(tree, obs, mask, n_step=n_step)
        for t in transitions:
            T[t[0],t[1],t[2]] += 1
        T = normalize_to_prob(T, axis=n_step)

        _, reg_precision_model = estimate_precision_model(T, tree, env, model, n_step=n_step, n_runs=estimation_runs)
        _, reg_precision_tree  = estimate_precision_tree(T, tree, env, model, n_step=n_step, n_runs=estimation_runs)
        print(f'Precision ({n_step} step, model acting): {reg_precision_model}')
        print(f'Precision ({n_step} step, tree acting):  {reg_precision_tree}')

        # plot the current tree
        if i > -1 and n_dims == 2:
            plot_tree_partition(
                tree, draw_boundaries=False,
                # points=obs, acts=acts, mask=mask,
                title=f"{model_name} unified_Round {i+1}", save_dir='./data/figs/'
            )
