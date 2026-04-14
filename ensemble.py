"""
Ensemble of CARPET tree partitionings.

Usage
-----
Build an ensemble of k trees, each trained independently via run_carpet:

    trees = build_ensemble(
        k=5,
        make_tree=lambda: my_initialized_tree_factory(),
        env=env, model=model, logger=logger, model_dir=model_dir,
        env_name='Bouncing Ball',
        **carpet_kwargs,
    )

Each member tree gets its own RunLogger and subfolder under data/results/runs/.
An ensemble manifest is saved to data/results/ensembles/{env}_{id}.json.

Inference
---------
Given a trained ensemble, assign a state to its intersection region and compute
a factored transition distribution:

    region_tuple = ensemble_region(trees, state)
    dist = ensemble_transition(trees, region_tuple)
"""

import json
import os
import numpy as np
from datetime import datetime

from analysis.metrics import evaluate
from pipeline import run_carpet
from utils import RunLogger


def build_ensemble(k, make_tree, env, model, logger, model_dir, env_name,
                   **carpet_kwargs):
    """
    Train k independent trees using run_carpet.

    Parameters
    ----------
    k : int
        Number of ensemble members.
    make_tree : callable
        Zero-argument function returning a freshly initialised TreeObserver
        (including initial action split). Called once per member.
    env, model, logger, model_dir
        Passed through to run_carpet.
    env_name : str
        Used for RunLogger subfolder naming and ensemble manifest.
    **carpet_kwargs
        Passed through to run_carpet and stored in the manifest.

    Returns
    -------
    trees : list of TreeObserver
        The k trained trees.
    manifest_path : str
        Path to the saved ensemble manifest JSON.
    """
    ensemble_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    member_run_ids = []
    trees = []

    for i in range(k):
        logger.section(f'Ensemble member {i + 1}/{k}')

        tree = make_tree()

        run_logger = RunLogger(
            env_name=env_name,
            config_dict={**carpet_kwargs, 'ensemble_id': ensemble_id, 'member_index': i},
            description=f'Ensemble {ensemble_id} — member {i + 1}/{k}',
        )

        run_carpet(tree, env, model, logger, model_dir,
                   run_logger=run_logger, **carpet_kwargs)
        run_logger.close()

        member_run_ids.append(run_logger.run_id)
        trees.append(tree)

        tree_path = os.path.join(run_logger.run_dir, 'tree.joblib')
        tree.save(tree_path)

    os.makedirs('./data/results/ensembles', exist_ok=True)
    manifest = {
        'ensemble_id': ensemble_id,
        'env': env_name,
        'k': k,
        'members': [{'index': i, 'run_id': rid} for i, rid in enumerate(member_run_ids)],
        'config': carpet_kwargs,
    }
    manifest_path = f'./data/results/ensembles/{env_name}_{ensemble_id}.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.log(f'Ensemble manifest saved to {manifest_path}')
    return trees, manifest_path


def load_ensemble(manifest_path):
    """
    Load a previously built ensemble from disk.

    Reads the manifest JSON and loads each member tree from the
    tree.joblib file in its run subfolder.

    Parameters
    ----------
    manifest_path : str
        Path to the ensemble manifest JSON (e.g.
        'data/results/ensembles/random_walk_20260331_120000.json').

    Returns
    -------
    trees : list of TreeObserver
    manifest : dict
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    from models.tree import TreeObserver

    trees = []
    for member in manifest['members']:
        run_dir = f"./data/results/runs/{manifest['env']}_{member['run_id']}"
        tree_path = os.path.join(run_dir, 'tree.joblib')
        trees.append(TreeObserver.load(tree_path))

    return trees, manifest


def ensemble_region(trees, state):
    """
    Return the intersection region tuple for a state across all trees.

    Parameters
    ----------
    trees : list of TreeObserver
    state : np.ndarray, shape (n_dims,)

    Returns
    -------
    tuple of int  — (r1, r2, ..., rk), one region label per tree
    """
    return tuple(tree.get(state).label for tree in trees)


def ensemble_transition(trees, region_tuple):
    """
    Compute a factored transition distribution over intersection regions.

    Uses the product-of-marginals model: assumes independence across trees.
    Enumerates all intersection regions observed in the product support,
    normalises to 1 (filtering geometric phantoms implicitly via the
    outer product — any zero-probability entry in any tree drops the whole
    combination to zero).

    Parameters
    ----------
    trees : list of TreeObserver
    region_tuple : tuple of int
        Current intersection region, one label per tree.

    Returns
    -------
    dist : dict mapping intersection tuple → probability
        Sparse — only non-zero entries are included.
    """
    # Per-tree transition distributions from the current region
    marginals = []
    for tree, r in zip(trees, region_tuple):
        row = tree.T[r]               # shape (n_leaves_i,)
        nonzero = np.where(row > 0)[0]
        marginals.append({int(j): float(row[j]) for j in nonzero})

    # Compute the outer product over all trees (only non-zero combinations)
    dist = {(): 1.0}
    for marginal in marginals:
        new_dist = {}
        for prefix, p_prefix in dist.items():
            for r_next, p_next in marginal.items():
                new_dist[prefix + (r_next,)] = p_prefix * p_next
        dist = new_dist

    # Normalise (product of already-normalised marginals sums to 1 in theory,
    # but floating-point drift may occur)
    total = sum(dist.values())
    if total > 0:
        dist = {k: v / total for k, v in dist.items()}

    return dist


def evaluate_ensemble(trees, obs, mask):
    """
    Evaluate the ensemble on training data.

    Per-tree: runs `evaluate` independently on each tree.
    Joint: computes LL under the product-of-marginals ensemble transition model.
    The ensemble transition for each step is looked up from a cache keyed on
    the current intersection tuple to avoid redundant outer-product computations.

    Parameters
    ----------
    trees : list of TreeObserver
    obs : np.ndarray, shape (n_runs, T, n_dims)
    mask : np.ndarray, shape (n_runs, T), bool

    Returns
    -------
    per_tree : list of (ll, perplexity, n_zero, n_total) — one per tree
    joint    : (ll, perplexity, n_zero, n_total)         — ensemble joint model
    """
    per_tree = [evaluate(tree, obs, mask) for tree in trees]

    total_ll = 0.0
    n_transitions = 0
    n_zero = 0

    for run, run_mask in zip(obs, mask):
        run_obs = run[run_mask]
        tuples = [ensemble_region(trees, x) for x in run_obs]

        for i in range(len(tuples) - 1):
            current = tuples[i]
            nxt = tuples[i + 1]

            # P(nxt | current) = product of each tree's marginal probability.
            # No need to materialise the full outer-product distribution — we only
            # ever need the probability of this specific observed next tuple.
            prob = 1.0
            for tree, r_cur, r_nxt in zip(trees, current, nxt):
                prob *= float(tree.T[r_cur, r_nxt])

            if prob == 0.0:
                n_zero += 1
            else:
                total_ll += np.log(prob)
                n_transitions += 1

    total = n_transitions + n_zero
    avg_ll = total_ll / n_transitions if n_transitions > 0 else float('-inf')
    perplexity = np.exp(-avg_ll)

    return per_tree, (avg_ll, perplexity, n_zero, total)


def estimate_precision_ensemble(trees, env, model, n_runs=100):
    """
    Estimate ensemble precision when the real model is acting.

    For each step:
      - Maps current state to intersection tuple via ensemble_region.
      - Gets the predicted distribution via ensemble_transition.
      - Steps the environment with the model's action.
      - Maps next state to its intersection tuple.
      - Records whether the actual tuple is in the support (in-support precision)
        and whether it is the single most likely tuple (top-1 precision).

    Parameters
    ----------
    trees : list of TreeObserver
    env   : gym environment
    model : RL model with predict()
    n_runs : int

    Returns
    -------
    in_support_prec : float — fraction of steps where actual next tuple has nonzero prob
    top1_prec       : float — fraction of steps where actual next tuple is the argmax
    """
    n_support = 0
    n_top1 = 0
    n_total = 0
    transition_cache = {}

    for _ in range(n_runs):
        obs, _ = env.reset()
        done = False

        while not done:
            current = ensemble_region(trees, obs)

            if current not in transition_cache:
                transition_cache[current] = ensemble_transition(trees, current)
            dist = transition_cache[current]

            action, _ = model.predict(obs, deterministic=True)
            nobs, _, done, _, _ = env.step(action)

            nxt = ensemble_region(trees, nobs)

            if dist.get(nxt, 0.0) > 0:
                n_support += 1

            if dist and max(dist, key=dist.get) == nxt:
                n_top1 += 1

            n_total += 1
            obs = nobs

    in_support = n_support / n_total if n_total > 0 else 0.0
    top1 = n_top1 / n_total if n_total > 0 else 0.0
    return in_support, top1
