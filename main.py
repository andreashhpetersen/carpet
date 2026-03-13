import numpy as np
import gymnasium as gym
import uppaal_gym
from collections import defaultdict
from itertools import combinations
from configs import load_config

from analysis.metrics import estimate_precision_n_step, evaluate
from envs.load import load_env
from learning.splitting import split_on_action, split_on_transition
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
            if ns == ps:
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

    return np.concat(transitions)


if __name__ == '__main__':

    # load config
    config = load_config('random_walk')
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
    rounds = config['rounds']
    n_dims = config['n_dims']
    n_acts = config['n_acts']

    # load environment and model, and generate training data
    env = load_env(env_id)
    model = load_or_train_model(env, model_path, n_timesteps=n_timesteps)
    obs, acts, _, mask = load_training_data(model_dir)

    tree = TreeObserver(
        n_dims=n_dims, n_acts=n_acts,
        bounds=bounds, initial_preds=initial_preds
    )

    if mark_terminal:
        tree.mark_terminal_states(obs, mask)

    # learn initial action mapping
    split_on_action(tree, obs, acts, mask, thresh=0.9)
    tree.reorder_leaf_labels()

    plot_tree_partition(
        tree,
        title="Initial action mapping",
        draw_boundaries=True,
        points=obs, acts=acts, mask=mask,
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

        # splitting
        split_on_transition(region2states, tree)
        print(f'Regions: {tree.n_leaves}')

        # update transition scores
        obs, acts, _, mask = load_training_data(model_dir)
        tree.set_transition_scores(obs, mask)

        # evaluate
        evaluate(tree, obs, mask)

        # check precision
        T1 = np.array([l.T for l in tree.leaves()])
        _, reg_precision = estimate_precision_n_step(T1, tree, env, model)
        print(f'Precision (1 step): {reg_precision}')

        # make n step transition matrix and check n step precision
        n_step = 2
        T = np.zeros((tree.n_leaves,) * (n_step + 1), dtype=np.int32)

        transitions = get_transitions(tree, obs, mask, n_step=n_step)
        for t in transitions:
            T[t[0],t[1],t[2]] += 1
        T = normalize_to_prob(T, axis=n_step)

        _, reg_precision = estimate_precision_n_step(T, tree, env, model, n_step=n_step)
        print(f'Precision ({n_step} step): {reg_precision}')

        # plot the current tree
        if i > 8:
            plot_tree_partition(
                tree, draw_boundaries=False,
                points=obs, acts=acts, mask=mask,
                title=f"Round {i+1}"
            )
            import ipdb; ipdb.set_trace()
